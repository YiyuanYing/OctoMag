"""
磁体路径跟踪控制实验记录版。

功能概览：
- 通过 OpenCV 摄像头画面检测小磁体中心位置。
- 使用 PySide6 提供实时显示、选择起点、绘制路径、停止和磁铁模式切换。
- 使用带预瞄点的 PID 控制，让磁体沿用户绘制路径运动。
- 通过串口 COM4、115200 波特率向下位机发送 a0-a5 六通道电流。
- 路径开始后自动录像，停止时保存 CSV 日志和 X/Y 误差曲线图。

运行环境：
- 建议使用 conda 环境 grad_mag，Python 3.13。
- 需要 pip 包：opencv-python、numpy、pyserial、PySide6、matplotlib。

输出文件：
- traj_*.avi：实验过程录像。
- data_*.csv：时间、目标点、实际位置、误差、磁铁编号和电流记录。
- errors_*.png：X/Y 方向误差曲线。

注意事项：
- 默认摄像头索引为 1，如设备不同请修改 cv2.VideoCapture(1)。
- 默认串口为 COM4，如下位机端口不同请修改 SERIAL_PORT。
"""

import sys
import cv2
import numpy as np
import serial
import time
import math
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton,
    QLabel, QVBoxLayout, QWidget
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# 串口参数
SERIAL_PORT = "COM4"
BAUDRATE = 115200

class OverlayLabel(QLabel):
    """自定义标签，用于接收鼠标事件并转发给主窗口"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
    def mousePressEvent(self, event):
        if self.main_window:
            self.main_window.handle_mouse_press(event)
    def mouseMoveEvent(self, event):
        if self.main_window:
            self.main_window.handle_mouse_move(event)
    def mouseReleaseEvent(self, event):
        if self.main_window:
            self.main_window.handle_mouse_release(event)

class MagneticControl(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("6极电磁铁磁体控制 - PID 工业级")
        self.resize(1000, 800)

        # 摄像头初始化
        self.cap = cv2.VideoCapture(1)

        # --- GUI组件 ---
        self.image_label = OverlayLabel(self)
        self.btn_start = QPushButton("选择起点")
        self.btn_draw = QPushButton("绘制路径")
        self.btn_stop = QPushButton("停止")
        self.btn_mode = QPushButton("Mode:1-3Mags")  # 初始为1-3号模式
        self.btn_mode.setCheckable(True)
        self.btn_mode.setChecked(True)

        # 连接按钮事件
        self.btn_start.clicked.connect(self.select_start)
        self.btn_draw.clicked.connect(self.enable_path_drawing)
        self.btn_stop.clicked.connect(self.stop_motion)
        self.btn_mode.clicked.connect(self.toggle_mode)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_draw)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_mode)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # --- 控制参数（可调节） ---
        self.Kp = 12
        self.Ki = 0.02
        self.Kd = 0.2

        self.timer_interval = 30    # 定时器周期 (ms)
        self.max_current = 90       # 单磁铁最大电流
        self.max_delta = 10         # 电流变化速率限制
        self.preview_steps = 3      # 预瞄目标点步数

        # 工作模式
        self.use_123_only = True    # True: 只用1-3号电磁铁；False: 用全部6号

        # 状态变量
        self.start_point = None
        self.magnet_center = None
        self.traj_line = []
        self.full_path = []
        self.current_target_index = 0

        # 绘制路径相关
        self.drawing_path = False
        self.is_right_drawing = False
        self.path_points = []
        self.way_point = None

        # 录像与记录
        self.recording = False
        self.video_writer = None
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fps = 1000.0 / self.timer_interval  # ~33FPS
        self.frame_size = None
        self.motion_start_time = None

        # PID内部变量
        self.int_error_x = 0.0
        self.int_error_y = 0.0
        self.last_meas_x = 0.0
        self.last_meas_y = 0.0

        # 日志数据
        self.log = []

        # 串口初始化（未连接时进入干运行模式）
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
        except:
            self.ser = None
            print("⚠ 串口未连接，仅GUI运行")

        # 定时器启动（参见 Qt 文档【6†L2178-L2186】）
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self.timer_interval)

    def select_start(self):
        """选择当前磁体位置作为起点"""
        if self.magnet_center:
            self.start_point = self.magnet_center
            self.traj_line = [self.start_point]
            self.full_path = []
            self.current_target_index = 0
            self.int_error_x = 0.0
            self.int_error_y = 0.0
            self.last_meas_x, self.last_meas_y = self.magnet_center

    def enable_path_drawing(self):
        """进入路径绘制模式（右键绘制曲线）"""
        self.drawing_path = True
        self.path_points = []
        self.way_point = None

    def stop_motion(self):
        """停止运动：录像停止、发送0电流、保存数据并重置状态"""
        if self.recording and self.video_writer:
            self.video_writer.release()
        self.recording = False

        # 发送零电流
        self.send_currents([0]*6)

        # 保存误差数据和绘制图表
        self.save_data()

        # 重置运动状态
        self.full_path = []
        self.current_target_index = 0
        self.motion_start_time = None
        self.traj_line = []

    def toggle_mode(self):
        """切换磁铁使用模式：1-3号或全6号"""
        self.use_123_only = not self.use_123_only
        if self.use_123_only:
            self.btn_mode.setText("Mode:1-3Mags")
        else:
            self.btn_mode.setText("Mode:1-6Mags")

    def handle_mouse_press(self, event):
        """处理图像标签上的鼠标按下事件"""
        if self.drawing_path and event.button() == Qt.RightButton:
            self.is_right_drawing = True
            x = int(event.position().x())
            y = int(event.position().y())
            self.way_point = (x, y)
            self.path_points = [self.way_point]

    def handle_mouse_move(self, event):
        """处理鼠标移动事件（累加绘制路径点）"""
        if self.is_right_drawing:
            x = int(event.position().x())
            y = int(event.position().y())
            self.path_points.append((x, y))

    def handle_mouse_release(self, event):
        """处理鼠标释放事件：完成路径绘制"""
        if self.is_right_drawing:
            x = int(event.position().x())
            y = int(event.position().y())
            self.path_points.append((x, y))
            self.is_right_drawing = False
            self.drawing_path = False
            self.generate_full_path()

    def generate_full_path(self):
        """根据起点和绘制路径生成完整的路径序列"""
        if self.start_point is None or self.way_point is None:
            return
        x0, y0 = self.start_point
        x1, y1 = self.way_point
        full = []
        steps = 30
        for i in range(steps):
            t = i / (steps - 1)
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            full.append((x, y))
        full.extend(self.path_points)
        self.full_path = full
        self.current_target_index = 0

        # 开始录像
        if self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_size = (width, height)
            filename = f"traj_{time.strftime('%Y%m%d_%H%M%S')}.avi"
            self.video_writer = cv2.VideoWriter(filename, self.fourcc, self.fps, self.frame_size)
            self.recording = True

        self.motion_start_time = time.time()

    def update_frame(self):
        """定时获取摄像头图像，绘制界面并执行控制"""
        ret, frame = self.cap.read()
        if not ret:
            return

        # 检测磁体中心（通过简单的二值化和轮廓找中心）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.magnet_center = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w//2; cy = y + h//2
            self.magnet_center = (cx, cy)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # 绘制起点和实际轨迹（红色）
        if self.start_point:
            cv2.circle(frame, self.start_point, 6, (0, 0, 255), -1)
        for i in range(1, len(self.traj_line)):
            cv2.line(frame, self.traj_line[i-1], self.traj_line[i], (0, 0, 255), 2)

        # 绘制规划的完整路径（绿色）
        for i in range(1, len(self.full_path)):
            cv2.line(frame,
                     tuple(map(int, self.full_path[i-1])),
                     tuple(map(int, self.full_path[i])),
                     (0, 255, 0), 2)

        # 显示当前模式
        mode_text = "1-3Mags" if self.use_123_only else "1-6Mags"
        cv2.putText(frame, f"Mode:{mode_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 执行控制循环
        if self.start_point and self.magnet_center and len(self.full_path) > 0:
            self.control_magnet()

        # 写入视频帧
        if self.recording and self.video_writer:
            self.video_writer.write(frame)

        # Qt 显示
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def control_magnet(self):
        """PID 控制 + 单磁铁选择 + 限幅限速 + 发送电流"""
        # 预瞄目标点索引
        idx = min(self.current_target_index + self.preview_steps, len(self.full_path) - 1)
        tx, ty = self.full_path[idx]
        cx, cy = self.magnet_center
        # 计算误差（注意 OpenCV 图像坐标 y 反向）
        err_x = tx - cx
        err_y = -(ty - cy)

        # 更新积分量
        dt = self.timer_interval / 1000.0
        self.int_error_x += err_x * dt
        self.int_error_y += err_y * dt
        # 限制积分避免饱和（抗积分风加）【11†L1156-L1164】
        self.int_error_x = max(min(self.int_error_x, self.max_current), -self.max_current)
        self.int_error_y = max(min(self.int_error_y, self.max_current), -self.max_current)

        # 计算速度（测量法微分）
        vx = (cx - self.last_meas_x) / dt
        vy = (cy - self.last_meas_y) / dt
        self.last_meas_x = cx
        self.last_meas_y = cy

        # PID 输出
        ux = self.Kp * err_x + self.Ki * self.int_error_x - self.Kd * vx
        uy = self.Kp * err_y + self.Ki * self.int_error_y - self.Kd * vy

        # 检查是否到达当前目标点
        dist = math.hypot(err_x, err_y)
        if dist < 5:
            if self.current_target_index >= len(self.full_path) - 1:
                # 最终点到达，停止控制
                self.send_currents([0]*6)
                self.recording = False
                return
            else:
                self.current_target_index += 1

        # 计算各电磁铁方向投影（1-3号为例，6号模式时再加后三个方向）
        directions = [
            (-math.sqrt(2)/2,  math.sqrt(2)/2),  # a0
            (0, -1),                             # a1
            (math.sqrt(3)/2, 0.5)                # a2
        ]
        if not self.use_123_only:
            directions.extend([
                (math.sqrt(2)/2, -math.sqrt(2)/2),  # a3
                (0, 1),                             # a4
                (-math.sqrt(3)/2, -0.5)             # a5
            ])

        # 计算投影贡献
        contributions = [ux*dx + uy*dy for (dx, dy) in directions]
        max_idx = int(np.argmax(contributions))
        val = contributions[max_idx]
        target_current = int(round(max(0, min(self.max_current, val))))

        # 构造6个电流输出（仅一个非零）
        currents = [0]*6
        currents[max_idx] = target_current

        # 电流变化限幅（±max_delta）
        if not hasattr(self, 'last_currents'):
            self.last_currents = [0]*6
        for i in range(6):
            delta = currents[i] - self.last_currents[i]
            if delta > self.max_delta:
                currents[i] = self.last_currents[i] + self.max_delta
            elif delta < -self.max_delta:
                currents[i] = self.last_currents[i] - self.max_delta
            # 再次限幅
            currents[i] = max(0, min(self.max_current, currents[i]))
        self.last_currents = currents.copy()

        # 发送电流命令
        self.send_currents(currents)

        # 记录轨迹和日志
        self.traj_line.append(self.magnet_center)
        t = time.time() - (self.motion_start_time or time.time())
        err_x_final = self.full_path[-1][0] - cx
        err_y_final = -(self.full_path[-1][1] - cy)
        self.log.append((t, tx, ty, cx, cy, err_x_final, err_y_final, max_idx, currents[max_idx]))

    def send_currents(self, currents):
        """通过串口发送6通道电流值命令"""
        cmd = (f"a0:{int(currents[0]):02d},"
               f"a1:{int(currents[1]):02d},"
               f"a2:{int(currents[2]):02d},"
               f"a3:{int(currents[3]):02d},"
               f"a4:{int(currents[4]):02d},"
               f"a5:{int(currents[5]):02d}\r\n")
        print(cmd.strip())
        if self.ser:
            self.ser.write(cmd.encode())

    def save_data(self):
        """保存误差数据为CSV并绘制误差曲线"""
        if not self.log:
            return
        import csv
        fname = f"data_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(fname, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time','tx','ty','cx','cy','err_x','err_y','magnet','current'])
            for row in self.log:
                writer.writerow(row)
        data = np.array(self.log)
        times = data[:,0].astype(float)
        err_xs = data[:,5].astype(float)
        err_ys = data[:,6].astype(float)
        plt.figure(figsize=(6,5))
        plt.subplot(2,1,1)
        plt.plot(times, err_xs, label='X误差')
        plt.xlabel('时间(s)'); plt.ylabel('误差(px)'); plt.legend()
        plt.subplot(2,1,2)
        plt.plot(times, err_ys, label='Y误差', color='orange')
        plt.xlabel('时间(s)'); plt.ylabel('误差(px)'); plt.legend()
        plt.tight_layout()
        plt.savefig(f"errors_{time.strftime('%Y%m%d_%H%M%S')}.png")
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MagneticControl()
    window.show()
    sys.exit(app.exec())
