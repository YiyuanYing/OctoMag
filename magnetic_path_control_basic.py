"""
磁体路径跟踪控制基础版。

功能概览：
- 通过 OpenCV 摄像头画面检测小磁体中心位置。
- 使用 PySide6 提供实时显示、选择起点、绘制路径、停止和磁铁模式切换。
- 使用 PID 控制把路径跟踪误差转换为 6 路电磁铁电流命令。
- 通过串口 COM4、115200 波特率向下位机发送 a0-a5 六通道电流。

运行环境：
- 建议使用 conda 环境 grad_mag，Python 3.13。
- 需要 pip 包：opencv-python、numpy、pyserial、PySide6、matplotlib。

注意事项：
- 默认摄像头索引为 1，如设备不同请修改 cv2.VideoCapture(1)。
- 默认串口为 COM4，如下位机端口不同请修改 SERIAL_PORT。
- 该版本主要用于实时控制和调试，误差曲线函数不会自动保存实验数据。
"""

import sys
import cv2
import numpy as np
import serial
import time
import math
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QWidget
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt

# ================== 串口配置 ==================
SERIAL_PORT = "COM4"        # 串口号，可根据实际调整
BAUDRATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
except:
    ser = None
    print("⚠ 串口未连接，仅在终端打印电流命令")

# ================== PID 参数 ==================
Kp = 8   # 比例增益（可调）
Ki = 0.02  # 积分增益（较小，防止积分风up）
Kd = 0.2   # 微分增益（提供阻尼与前瞻性）
# 最大电流与斜率限制
MAX_I = 80
MAX_DELTA_I = 10  # 每帧电流变化限幅

# ================== 主窗口类 ==================
class MagneticControl(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("6极电磁铁磁体控制 - 工业级方案")
        self.resize(1000, 800)

        # 摄像头
        self.cap = cv2.VideoCapture(1)  # 摄像头索引，默认1
        # 界面组件
        self.image_label = QLabel()
        self.btn_start = QPushButton("选择起点")
        self.btn_draw = QPushButton("绘制路径")
        self.btn_stop = QPushButton("停止")
        self.btn_mode = QPushButton("模式：1-3号磁铁")
        self.lbl_currents = QLabel("电流: a0=0 a1=0 a2=0 a3=0 a4=0 a5=0")

        # 按钮绑定
        self.btn_start.clicked.connect(self.select_start)
        self.btn_draw.clicked.connect(self.enable_path_drawing)
        self.btn_stop.clicked.connect(self.stop_motion)
        self.btn_mode.clicked.connect(self.toggle_mode)

        # 布局设置
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_draw)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_mode)
        btn_layout.addWidget(self.lbl_currents)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 状态变量
        self.start_point = None            # 起点
        self.magnet_center = None          # 当前磁体中心
        self.traj_line = []               # 记录磁体轨迹

        # 路径绘制相关
        self.drawing_path = False
        self.is_right_drawing = False
        self.path_points = []             # 鼠标绘制的曲线路径点
        self.way_point = None             # 右键按下时的途径点（曲线起点）
        self.full_path = []               # 完整跟踪路径点序列
        self.current_target_index = 0     # 当前跟踪的路径点索引

        # PID控制变量
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.last_time = time.time()
        self.last_x = 0.0
        self.last_y = 0.0

        # 磁铁模式（默认只用1-3号磁铁）
        self.use_123_only = True

        # 上一帧输出电流（用于斜率限制）
        self.last_currents = [0]*6

        # 定时器：控制循环约30ms刷新
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ================== 图像与控制更新 ==================
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.magnet_center = None
        if contours:
            # 取最大轮廓为磁体
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w//2; cy = y + h//2
            self.magnet_center = (cx, cy)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # 绘制起点
        if self.start_point:
            cv2.circle(frame, self.start_point, 6, (0, 0, 255), -1)
        # 绘制轨迹
        for i in range(len(self.traj_line)-1):
            cv2.line(frame, self.traj_line[i], self.traj_line[i+1], (0, 0, 255), 1)
        # 绘制完整路径
        if len(self.full_path) > 1:
            for i in range(1, len(self.full_path)):
                p0 = tuple(map(int, self.full_path[i-1]))
                p1 = tuple(map(int, self.full_path[i]))
                cv2.line(frame, p0, p1, (0, 255, 0), 2)

        # 显示当前模式文字
        mode_text = "模式：1-3号磁铁" if self.use_123_only else "模式：全部6号磁铁"
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

        # 执行控制
        if self.start_point and self.magnet_center and len(self.full_path) > 0:
            self.control_magnet()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(img))

    # ================== 选择起点 ==================
    def select_start(self):
        if self.magnet_center:
            self.start_point = self.magnet_center
            self.traj_line = [self.start_point]
            self.full_path = []
            self.current_target_index = 0
            self.integral_x = 0.0
            self.integral_y = 0.0
            self.last_time = time.time()
            self.last_x, self.last_y = self.start_point
            # 重置电流显示
            self.lbl_currents.setText("电流: a0=0 a1=0 a2=0 a3=0 a4=0 a5=0")

    # ================== 启用路径绘制 ==================
    def enable_path_drawing(self):
        # 开始绘制路径
        self.drawing_path = True
        self.path_points = []
        self.way_point = None
        self.btn_draw.setStyleSheet("background-color: green;")
        # 重置之前的路径
        self.full_path = []
        self.current_target_index = 0

    # ================== 停止运动 ==================
    def stop_motion(self):
        # 立即清除路径并置零输出
        self.full_path = []
        self.current_target_index = 0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.send_currents([0]*6)
        self.lbl_currents.setText("电流: a0=0 a1=0 a2=0 a3=0 a4=0 a5=0")
        print("已停止运动：电流清零、路径重置、PID状态复位。")

    # ================== 模式切换 ==================
    def toggle_mode(self):
        self.use_123_only = not self.use_123_only
        if self.use_123_only:
            self.btn_mode.setText("模式：1-3号磁铁")
            self.btn_mode.setStyleSheet("background-color: #e67e22; color: white;")
        else:
            self.btn_mode.setText("模式：全部6号磁铁")
            self.btn_mode.setStyleSheet("background-color: #3498db; color: white;")
        print("当前模式:", "1-3号磁铁" if self.use_123_only else "全部6号磁铁")

    # ================== 鼠标事件处理 ==================
    def mousePressEvent(self, event):
        if self.drawing_path and event.button() == Qt.RightButton:
            self.is_right_drawing = True
            pos = event.position().toPoint()
            self.way_point = (pos.x(), pos.y())
            self.path_points = [self.way_point]

    def mouseMoveEvent(self, event):
        if self.is_right_drawing:
            pos = event.position().toPoint()
            self.path_points.append((pos.x(), pos.y()))

    def mouseReleaseEvent(self, event):
        if self.is_right_drawing:
            pos = event.position().toPoint()
            self.path_points.append((pos.x(), pos.y()))
            self.is_right_drawing = False
            self.drawing_path = False
            self.btn_draw.setStyleSheet("")
            self.generate_full_path()

    # ================== 构造完整路径 ==================
    def generate_full_path(self):
        if self.start_point is None or self.way_point is None:
            return

        full_path = []
        x0, y0 = self.start_point
        x1, y1 = self.way_point
        # 第1段：起点到途径点1的直线插值
        segments = 30
        for i in range(segments):
            t = i / segments
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            full_path.append((x, y))
        # 第2段：添加鼠标绘制的曲线段
        full_path.extend(self.path_points)

        self.full_path = full_path
        self.current_target_index = 0
        print("路径已设置，点数量：", len(full_path))

    # ================== 控制主循环 ==================
    def control_magnet(self):
        # 跟踪结束，则输出0电流
        if self.current_target_index >= len(self.full_path):
            self.send_currents([0] * 6)
            return

        tx, ty = self.full_path[self.current_target_index]
        cx, cy = self.magnet_center

        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 0.01
        self.last_time = now

        # 计算误差（注意OpenCV坐标y反转）
        error_x = tx - cx
        error_y = -(ty - cy)

        # 实际速度（用于微分项）
        vx = (cx - self.last_x) / dt
        vy = (cy - self.last_y) / dt
        self.last_x, self.last_y = cx, cy

        # 积分更新 (条件积分以防积分饱和)
        # 只有当输出未饱和时才累积积分，可选策略之一
        # 这里简单限定积分不超过阈值
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        # 积分限幅（防止积分累加过大）
        I_max = 1000.0  # 积分限幅可调
        self.integral_x = max(-I_max, min(I_max, self.integral_x))
        self.integral_y = max(-I_max, min(I_max, self.integral_y))

        # PID 输出
        ux = Kp * error_x + Ki * self.integral_x - Kd * vx
        uy = Kp * error_y + Ki * self.integral_y - Kd * vy

        distance = math.hypot(error_x, error_y)
        # 距离阈值内则认为到达当前子目标
        TOL = 10  # 像素容差，可调
        if distance < TOL:
            self.current_target_index += 1
            return

        # 磁铁单位方向（1-6号）
        all_dirs = [
            (-math.sqrt(2)/2,  math.sqrt(2)/2),  # a0
            (0, -1),                              # a1
            (math.sqrt(3)/2,  0.5),               # a2
            (math.sqrt(2)/2, -math.sqrt(2)/2),    # a3
            (0, 1),                               # a4
            (-math.sqrt(3)/2, -0.5)               # a5
        ]
        if self.use_123_only:
            dirs = all_dirs[:3]
        else:
            dirs = all_dirs

        # 计算各磁铁贡献量（可以为负，但后续只用正值）
        contributions = []
        for dx, dy in dirs:
            contributions.append(ux*dx + uy*dy)

        # 选出贡献最大的磁铁
        max_idx = int(np.argmax(contributions))
        max_val = contributions[max_idx]
        # 只输出正向贡献，否则输出0
        if max_val < 0:
            max_val = 0
        # 限幅到最大电流
        I_cmd = min(MAX_I, max_val)

        # 构造完整6通道电流列表（空余置0）
        currents = [0]*6
        currents[max_idx] = int(I_cmd)

        # 斜率限制（每帧最大ΔI）
        for i in range(6):
            delta = currents[i] - self.last_currents[i]
            if delta > MAX_DELTA_I:
                currents[i] = self.last_currents[i] + MAX_DELTA_I
            elif delta < -MAX_DELTA_I:
                currents[i] = self.last_currents[i] - MAX_DELTA_I

        # 发送电流指令
        self.send_currents(currents)

        # 更新轨迹记录
        self.traj_line.append(self.magnet_center)
        # 更新上次输出
        self.last_currents = currents.copy()

        # 更新电流显示
        self.lbl_currents.setText(
            f"电流: a0={currents[0]} a1={currents[1]} a2={currents[2]} a3={currents[3]} a4={currents[4]} a5={currents[5]}"
        )

    # ================== 发送串口指令 ==================
    def send_currents(self, currents):
        # 确保发送6通道格式
        cmd = (
            f"a0:{currents[0]:02d},a1:{currents[1]:02d},"
            f"a2:{currents[2]:02d},a3:{currents[3]:02d},"
            f"a4:{currents[4]:02d},a5:{currents[5]:02d}\r\n"
        )
        print(cmd.strip())
        if ser:
            ser.write(cmd.encode())

    # ================== 误差绘图 ==================
    def generate_error_plot(self):
        # 控制完成后可调用该函数绘制误差曲线
        if not self.full_path:
            return
        target_x, target_y = self.full_path[-1]
        # 计算轨迹到终点的距离误差
        error_list = [math.hypot(pt[0]-target_x, pt[1]-target_y) for pt in self.traj_line]
        t = np.arange(len(error_list)) * 0.03
        plt.figure()
        plt.plot(t, error_list)
        plt.xlabel("时间 (s)")
        plt.ylabel("到终点距离误差 (像素)")
        plt.title("跟踪误差曲线")
        plt.grid(True)
        plt.show()

# ================== 主程序入口 ==================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MagneticControl()
    window.show()
    sys.exit(app.exec())
