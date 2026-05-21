import sys
import time
import math

import cv2
import serial
import serial.tools.list_ports
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MicroClusterManualControlGUI(QMainWindow):
    """六线圈微纳集群手动控制 GUI。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("梯度线圈仪控制-微纳集群")

        # 六路线圈的空间方向，后续可按实际接线顺序修改。
        self.coil_angles_deg = [0, 60, 120, 180, 240, 300]
        self.coil_dirs = [
            (math.cos(math.radians(theta)), math.sin(math.radians(theta)))
            for theta in self.coil_angles_deg
        ]

        # 串口与摄像头资源。
        self.ser = None
        self.cap = None

        # 控制状态。
        self.motion_mode = "idle"
        self.rotate_direction = "CCW"
        self.translation_angle_deg = 0
        self.rotate_translate_angle_deg = 0
        self.pulse_coil_index = 0
        self.last_currents = [0, 0, 0, 0, 0, 0]
        self.idle_zero_sent = True
        self.motion_start_time = time.time()
        self.ramp_stop_start_time = None
        self.ramp_stop_start_currents = [0, 0, 0, 0, 0, 0]

        # 默认安全参数。
        self.max_current = 90
        self.max_delta = 10

        # 摄像头截图与录像状态。
        self.latest_frame = None
        self.recording = False
        self.video_writer = None
        self.video_filename = None

        # 串口实时命令日志节流。
        self.last_command_log_time = 0.0

        self.current_labels = []
        self.log_lines = []

        self.build_ui()
        self.update_mode_display()
        self.refresh_ports()

        # 线圈控制与摄像头刷新使用相互独立的定时器。
        self.control_timer = QTimer(self)
        self.control_timer.timeout.connect(self.control_loop)
        self.control_timer.start(30)

        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera_frame)

    def build_ui(self):
        """构建主界面布局。"""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(12)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.addWidget(self.build_serial_ui())
        left_layout.addWidget(self.build_emergency_ui())
        left_layout.addWidget(self.build_translation_ui())
        left_layout.addWidget(self.build_rotation_ui())
        left_layout.addWidget(self.build_smooth_ui())
        left_layout.addWidget(self.build_combined_ui())
        left_layout.addWidget(self.build_pulse_ui())
        left_layout.addStretch(1)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        right_layout.addWidget(self.build_camera_ui())
        right_layout.addWidget(self.build_monitor_ui(), 1)

        main_layout.addLayout(left_layout, 0)
        main_layout.addLayout(right_layout, 1)

    def build_serial_ui(self):
        """串口连接区。"""
        group = QGroupBox("串口连接区")
        layout = QGridLayout(group)

        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setMinimumWidth(140)
        self.port_combo.setCurrentText("COM7")

        refresh_button = QPushButton("刷新串口")
        connect_button = QPushButton("连接串口")
        disconnect_button = QPushButton("断开串口")
        self.serial_status_label = QLabel("未连接")

        refresh_button.clicked.connect(self.refresh_ports)
        connect_button.clicked.connect(self.connect_serial)
        disconnect_button.clicked.connect(self.disconnect_serial)

        layout.addWidget(QLabel("串口"), 0, 0)
        layout.addWidget(self.port_combo, 0, 1)
        layout.addWidget(refresh_button, 0, 2)
        layout.addWidget(connect_button, 1, 0)
        layout.addWidget(disconnect_button, 1, 1)
        layout.addWidget(self.serial_status_label, 1, 2)

        return group

    def build_emergency_ui(self):
        """急停按钮区，放在左侧靠上位置方便快速点击。"""
        self.emergency_stop_button = QPushButton("急停 / EMERGENCY STOP")
        self.emergency_stop_button.setMinimumHeight(52)
        self.emergency_stop_button.setStyleSheet(
            "background-color: red; color: white; font-weight: bold; font-size: 16px;"
        )
        self.emergency_stop_button.clicked.connect(self.emergency_stop)
        return self.emergency_stop_button

    def build_camera_ui(self):
        """小尺寸摄像头预览区。"""
        group = QGroupBox("摄像头预览区")
        layout = QVBoxLayout(group)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(720, 540)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("相机未打开")
        self.camera_label.setStyleSheet("background-color: black; color: white; border: 1px solid gray;")

        controls = QHBoxLayout()
        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(0, 20)
        self.camera_index_spin.setValue(0)

        open_button = QPushButton("打开相机")
        close_button = QPushButton("关闭相机")
        snapshot_button = QPushButton("保存截图")
        start_record_button = QPushButton("开始录像")
        stop_record_button = QPushButton("停止录像")
        self.camera_status_label = QLabel("相机：未打开")

        open_button.clicked.connect(self.open_camera)
        close_button.clicked.connect(self.close_camera)
        snapshot_button.clicked.connect(self.save_snapshot)
        start_record_button.clicked.connect(self.start_recording)
        stop_record_button.clicked.connect(self.stop_recording)

        controls.addWidget(QLabel("index"))
        controls.addWidget(self.camera_index_spin)
        controls.addWidget(open_button)
        controls.addWidget(close_button)

        record_controls = QHBoxLayout()
        record_controls.addWidget(snapshot_button)
        record_controls.addWidget(start_record_button)
        record_controls.addWidget(stop_record_button)

        layout.addWidget(self.camera_label, alignment=Qt.AlignHCenter)
        layout.addLayout(controls)
        layout.addLayout(record_controls)
        layout.addWidget(self.camera_status_label)

        return group

    def build_translation_ui(self):
        """平动控制区。"""
        group = QGroupBox("平动控制区")
        layout = QGridLayout(group)

        self.translation_current_spin = QSpinBox()
        self.translation_current_spin.setRange(0, 999)
        self.translation_current_spin.setValue(30)

        forward_button = QPushButton("前进")
        backward_button = QPushButton("后退")
        left_button = QPushButton("左移")
        right_button = QPushButton("右移")
        stop_button = QPushButton("停止")

        forward_button.clicked.connect(lambda: self.set_translate(90))
        backward_button.clicked.connect(lambda: self.set_translate(270))
        left_button.clicked.connect(lambda: self.set_translate(180))
        right_button.clicked.connect(lambda: self.set_translate(0))
        stop_button.clicked.connect(self.stop_all)

        layout.addWidget(QLabel("平动电流"), 0, 0)
        layout.addWidget(self.translation_current_spin, 0, 1, 1, 3)
        layout.addWidget(forward_button, 1, 1)
        layout.addWidget(left_button, 2, 0)
        layout.addWidget(stop_button, 2, 1)
        layout.addWidget(right_button, 2, 2)
        layout.addWidget(backward_button, 3, 1)

        return group

    def build_rotation_ui(self):
        """旋转控制区。"""
        group = QGroupBox("旋转控制区")
        layout = QGridLayout(group)

        cw_button = QPushButton("顺时针旋转")
        ccw_button = QPushButton("逆时针旋转")
        stop_button = QPushButton("停止旋转")

        cw_button.clicked.connect(lambda: self.set_rotate("CW"))
        ccw_button.clicked.connect(lambda: self.set_rotate("CCW"))
        stop_button.clicked.connect(self.stop_all)

        self.frequency_spin = QDoubleSpinBox()
        self.frequency_spin.setRange(0.01, 20.0)
        self.frequency_spin.setDecimals(2)
        self.frequency_spin.setSingleStep(0.1)
        self.frequency_spin.setValue(1.0)
        self.frequency_spin.setSuffix(" Hz")
        self.frequency_spin.valueChanged.connect(self.update_mode_display)

        self.rotation_amplitude_spin = QSpinBox()
        self.rotation_amplitude_spin.setRange(0, 999)
        self.rotation_amplitude_spin.setValue(30)

        self.bias_current_spin = QSpinBox()
        self.bias_current_spin.setRange(0, 999)
        self.bias_current_spin.setValue(45)

        self.max_current_spin = QSpinBox()
        self.max_current_spin.setRange(0, 999)
        self.max_current_spin.setValue(90)
        self.max_current_spin.valueChanged.connect(self.update_safety_limits)

        self.max_delta_spin = QSpinBox()
        self.max_delta_spin.setRange(1, 999)
        self.max_delta_spin.setValue(10)
        self.max_delta_spin.valueChanged.connect(self.update_safety_limits)

        layout.addWidget(cw_button, 0, 0)
        layout.addWidget(ccw_button, 0, 1)
        layout.addWidget(stop_button, 0, 2)
        layout.addWidget(QLabel("频率"), 1, 0)
        layout.addWidget(self.frequency_spin, 1, 1, 1, 2)
        layout.addWidget(QLabel("旋转幅值"), 2, 0)
        layout.addWidget(self.rotation_amplitude_spin, 2, 1, 1, 2)
        layout.addWidget(QLabel("偏置电流"), 3, 0)
        layout.addWidget(self.bias_current_spin, 3, 1, 1, 2)
        layout.addWidget(QLabel("最大电流"), 4, 0)
        layout.addWidget(self.max_current_spin, 4, 1, 1, 2)
        layout.addWidget(QLabel("max_delta"), 5, 0)
        layout.addWidget(self.max_delta_spin, 5, 1, 1, 2)

        return group

    def build_smooth_ui(self):
        """平滑控制区：缓启动与缓停止。"""
        group = QGroupBox("平滑控制区")
        layout = QGridLayout(group)

        self.ramp_enable_checkbox = QCheckBox("启用缓启动/缓停止")
        self.ramp_enable_checkbox.stateChanged.connect(self.update_mode_display)

        self.ramp_start_time_spin = QDoubleSpinBox()
        self.ramp_start_time_spin.setRange(0.0, 10.0)
        self.ramp_start_time_spin.setDecimals(2)
        self.ramp_start_time_spin.setSingleStep(0.1)
        self.ramp_start_time_spin.setValue(1.0)
        self.ramp_start_time_spin.setSuffix(" s")
        self.ramp_start_time_spin.valueChanged.connect(self.update_mode_display)

        self.ramp_stop_time_spin = QDoubleSpinBox()
        self.ramp_stop_time_spin.setRange(0.0, 10.0)
        self.ramp_stop_time_spin.setDecimals(2)
        self.ramp_stop_time_spin.setSingleStep(0.1)
        self.ramp_stop_time_spin.setValue(0.5)
        self.ramp_stop_time_spin.setSuffix(" s")
        self.ramp_stop_time_spin.valueChanged.connect(self.update_mode_display)

        layout.addWidget(self.ramp_enable_checkbox, 0, 0, 1, 2)
        layout.addWidget(QLabel("缓启动时间"), 1, 0)
        layout.addWidget(self.ramp_start_time_spin, 1, 1)
        layout.addWidget(QLabel("缓停止时间"), 2, 0)
        layout.addWidget(self.ramp_stop_time_spin, 2, 1)

        return group

    def build_combined_ui(self):
        """组合控制区。"""
        group = QGroupBox("组合控制区")
        layout = QGridLayout(group)

        forward_button = QPushButton("旋转 + 前进")
        backward_button = QPushButton("旋转 + 后退")
        left_button = QPushButton("旋转 + 左移")
        right_button = QPushButton("旋转 + 右移")

        forward_button.clicked.connect(lambda: self.set_rotate_translate(90))
        backward_button.clicked.connect(lambda: self.set_rotate_translate(270))
        left_button.clicked.connect(lambda: self.set_rotate_translate(180))
        right_button.clicked.connect(lambda: self.set_rotate_translate(0))

        self.move_overlay_spin = QSpinBox()
        self.move_overlay_spin.setRange(0, 999)
        self.move_overlay_spin.setValue(10)

        layout.addWidget(forward_button, 0, 0)
        layout.addWidget(backward_button, 0, 1)
        layout.addWidget(left_button, 1, 0)
        layout.addWidget(right_button, 1, 1)
        layout.addWidget(QLabel("平动叠加强度"), 2, 0)
        layout.addWidget(self.move_overlay_spin, 2, 1)

        return group

    def build_pulse_ui(self):
        """单线圈点动测试区。"""
        group = QGroupBox("单线圈点动测试区")
        layout = QGridLayout(group)

        self.pulse_current_spin = QSpinBox()
        self.pulse_current_spin.setRange(0, 999)
        self.pulse_current_spin.setValue(30)

        layout.addWidget(QLabel("点动电流"), 0, 0)
        layout.addWidget(self.pulse_current_spin, 0, 1, 1, 2)

        for i in range(6):
            button = QPushButton(f"a{i} 点动")
            button.pressed.connect(lambda idx=i: self.start_pulse(idx))
            button.released.connect(self.stop_pulse)
            layout.addWidget(button, 1 + i // 3, i % 3)

        return group

    def build_monitor_ui(self):
        """实时监控区。"""
        group = QGroupBox("实时监控区")
        layout = QVBoxLayout(group)

        status_layout = QGridLayout()
        self.mode_status_label = QLabel("当前模式：停止 / Idle")
        self.direction_status_label = QLabel("方向：-")
        self.frequency_status_label = QLabel("频率：-")
        self.record_status_label = QLabel("录像：未开始")
        self.ramp_status_label = QLabel("缓启动/缓停止：关闭")

        status_layout.addWidget(self.mode_status_label, 0, 0)
        status_layout.addWidget(self.direction_status_label, 0, 1)
        status_layout.addWidget(self.frequency_status_label, 1, 0)
        status_layout.addWidget(self.record_status_label, 1, 1)
        status_layout.addWidget(self.ramp_status_label, 2, 0, 1, 2)

        current_layout = QGridLayout()
        for i in range(6):
            label = QLabel(f"a{i}: 00")
            label.setMinimumWidth(70)
            self.current_labels.append(label)
            current_layout.addWidget(label, i // 3, i % 3)

        self.verbose_command_log_checkbox = QCheckBox("显示实时串口命令")

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(220)

        layout.addLayout(status_layout)
        layout.addLayout(current_layout)
        layout.addWidget(self.verbose_command_log_checkbox)
        layout.addWidget(QLabel("输出日志"))
        layout.addWidget(self.log_edit, 1)

        return group

    def refresh_ports(self):
        """刷新串口列表，同时保留手动输入的 COM 号。"""
        try:
            current_text = self.port_combo.currentText().strip() or "COM7"
            ports = [port.device for port in serial.tools.list_ports.comports()]

            self.port_combo.blockSignals(True)
            self.port_combo.clear()
            if current_text not in ports:
                self.port_combo.addItem(current_text)
            for port in ports:
                self.port_combo.addItem(port)
            self.port_combo.setCurrentText(current_text)
            self.port_combo.blockSignals(False)

            self.log(f"串口列表已刷新：{', '.join(ports) if ports else '未发现串口'}")
        except Exception as exc:
            self.log(f"刷新串口失败：{exc}")

    def connect_serial(self):
        """连接串口，失败时仅写日志。"""
        port = self.port_combo.currentText().strip() or "COM7"
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()

            self.ser = serial.Serial(port=port, baudrate=115200, timeout=0.1)
            self.serial_status_label.setText(f"已连接 {port}")
            self.log(f"串口已连接：{port}, baudrate=115200")
        except Exception as exc:
            self.ser = None
            self.serial_status_label.setText("未连接")
            self.log(f"串口连接失败：{port}, {exc}")

    def disconnect_serial(self):
        """断开串口。"""
        try:
            if self.ser and self.ser.is_open:
                self.last_currents = [0, 0, 0, 0, 0, 0]
                self.send_currents([0, 0, 0, 0, 0, 0])
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.log("串口已断开")
        except Exception as exc:
            self.log(f"断开串口失败：{exc}")
        finally:
            self.ser = None
            self.serial_status_label.setText("未连接")

    def open_camera(self):
        """打开摄像头，失败时仅写日志。"""
        index = self.camera_index_spin.value()
        try:
            self.close_camera(silent=True)
            self.cap = cv2.VideoCapture(index)
            if not self.cap or not self.cap.isOpened():
                self.cap = None
                self.camera_status_label.setText("相机：打开失败")
                self.camera_label.setText("相机打开失败")
                self.log(f"相机打开失败：index={index}")
                return

            self.camera_timer.start(30)
            self.camera_status_label.setText(f"相机：已打开 index={index}")
            self.log(f"相机已打开：index={index}")
        except Exception as exc:
            self.cap = None
            self.camera_status_label.setText("相机：打开失败")
            self.camera_label.setText("相机打开失败")
            self.log(f"相机打开异常：index={index}, {exc}")

    def close_camera(self, silent=False):
        """关闭摄像头并释放资源。"""
        try:
            was_open = bool(self.cap and self.cap.isOpened())
            if self.recording:
                self.stop_recording()
            self.camera_timer.stop()
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = None
            self.latest_frame = None
            self.camera_label.clear()
            self.camera_label.setText("相机未打开")
            self.camera_status_label.setText("相机：未打开")
            if was_open and not silent:
                self.log("相机已关闭")
        except Exception as exc:
            self.log(f"关闭相机失败：{exc}")

    def update_camera_frame(self):
        """读取并显示一帧摄像头画面。"""
        try:
            if not self.cap or not self.cap.isOpened():
                return

            ret, frame = self.cap.read()
            if not ret:
                return

            self.latest_frame = frame.copy()
            if self.recording and self.video_writer:
                self.video_writer.write(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.camera_label.setPixmap(pixmap)
        except Exception as exc:
            self.log(f"相机刷新失败：{exc}")

    def save_snapshot(self):
        """保存当前摄像头画面为 PNG。"""
        try:
            if not self.cap or not self.cap.isOpened():
                self.log("保存截图失败：相机未打开")
                return
            if self.latest_frame is None:
                self.log("保存截图失败：没有可用画面")
                return

            filename = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            if cv2.imwrite(filename, self.latest_frame):
                self.log(f"保存截图：{filename}")
            else:
                self.log(f"保存截图失败：{filename}")
        except Exception as exc:
            self.log(f"保存截图异常：{exc}")

    def start_recording(self):
        """开始保存摄像头录像。"""
        try:
            if not self.cap or not self.cap.isOpened():
                self.log("开始录像失败：相机未打开")
                return
            if self.recording:
                self.log(f"录像已在进行：{self.video_filename}")
                return

            if self.latest_frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    self.log("开始录像失败：没有可用画面")
                    return
                self.latest_frame = frame.copy()

            height, width = self.latest_frame.shape[:2]
            filename = f"video_{time.strftime('%Y%m%d_%H%M%S')}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
            if not writer.isOpened():
                writer.release()
                self.log(f"开始录像失败：{filename}")
                return

            self.video_writer = writer
            self.video_filename = filename
            self.recording = True
            self.log(f"开始录像：{filename}")
            self.update_mode_display()
        except Exception as exc:
            self.recording = False
            self.video_writer = None
            self.video_filename = None
            self.log(f"开始录像异常：{exc}")
            self.update_mode_display()

    def stop_recording(self):
        """停止录像并释放 VideoWriter。"""
        try:
            filename = self.video_filename
            if self.video_writer:
                self.video_writer.release()
            self.video_writer = None
            self.video_filename = None
            was_recording = self.recording
            self.recording = False
            if was_recording:
                self.log(f"停止录像：{filename}")
            self.update_mode_display()
        except Exception as exc:
            self.recording = False
            self.video_writer = None
            self.video_filename = None
            self.log(f"停止录像异常：{exc}")
            self.update_mode_display()

    def set_translate(self, angle_deg):
        """进入平动模式；再次点击同一方向则停止。"""
        if self.motion_mode == "translate" and self.translation_angle_deg == angle_deg:
            self.log(f"再次点击平动方向：angle={angle_deg} deg，停止输出")
            self.stop_all()
            return

        self.motion_mode = "translate"
        self.translation_angle_deg = angle_deg
        self.motion_start_time = time.time()
        self.idle_zero_sent = False
        self.ramp_stop_start_time = None
        self.log(f"进入平动模式：angle={angle_deg} deg")
        self.update_mode_display()

    def set_rotate(self, direction):
        """进入旋转模式；再次点击同一方向则停止。"""
        if self.motion_mode == "rotate" and self.rotate_direction == direction:
            text = "顺时针" if direction == "CW" else "逆时针"
            self.log(f"再次点击{text}旋转，停止输出")
            self.stop_all()
            return

        self.motion_mode = "rotate"
        self.rotate_direction = direction
        self.motion_start_time = time.time()
        self.idle_zero_sent = False
        self.ramp_stop_start_time = None
        text = "顺时针" if direction == "CW" else "逆时针"
        self.log(f"进入旋转模式：{text}")
        self.update_mode_display()

    def set_rotate_translate(self, angle_deg):
        """进入旋转叠加平动模式；再次点击同一方向则停止。"""
        if self.motion_mode == "rotate_translate" and self.rotate_translate_angle_deg == angle_deg:
            self.log(f"再次点击组合控制方向：angle={angle_deg} deg，停止输出")
            self.stop_all()
            return

        self.motion_mode = "rotate_translate"
        self.rotate_translate_angle_deg = angle_deg
        self.motion_start_time = time.time()
        self.idle_zero_sent = False
        self.ramp_stop_start_time = None
        self.log(f"进入组合控制模式：旋转 + 平动 angle={angle_deg} deg")
        self.update_mode_display()

    def start_pulse(self, coil_index):
        """按下单线圈点动按钮。"""
        self.motion_mode = "pulse"
        self.pulse_coil_index = coil_index
        self.motion_start_time = time.time()
        self.idle_zero_sent = False
        self.ramp_stop_start_time = None
        self.log(f"点动开始：a{coil_index}")
        self.update_mode_display()

    def stop_pulse(self):
        """松开点动按钮后立即停止输出。

        点动用于测试单路线圈接线，松手即断输出更符合安全直觉，因此这里不走缓停止。
        """
        self.log("点动停止")
        self.stop_all(force_immediate=True)

    def stop_all(self, force_immediate=False):
        """停止所有输出；普通停止可走缓停止，强制停止立即归零。"""
        if (
            not force_immediate
            and self.motion_mode not in ("idle", "ramp_stop")
            and self.is_ramp_enabled()
            and self.ramp_stop_time_spin.value() > 0
        ):
            self.start_ramp_stop()
            return

        self.motion_mode = "idle"
        self.idle_zero_sent = True
        self.ramp_stop_start_time = None
        self.last_currents = [0, 0, 0, 0, 0, 0]
        self.send_currents([0, 0, 0, 0, 0, 0])
        self.update_current_display([0, 0, 0, 0, 0, 0])
        self.log("停止输出：六路零电流")
        self.update_mode_display()

    def emergency_stop(self):
        """急停：强制立即归零，不走缓停止，不关闭串口和摄像头。"""
        self.motion_mode = "idle"
        self.idle_zero_sent = True
        self.ramp_stop_start_time = None
        self.ramp_stop_start_currents = [0, 0, 0, 0, 0, 0]
        self.last_currents = [0, 0, 0, 0, 0, 0]
        self.send_currents([0, 0, 0, 0, 0, 0])
        self.update_current_display([0, 0, 0, 0, 0, 0])
        self.log("急停触发：六路电流已归零")
        self.update_mode_display()

    def start_ramp_stop(self):
        """进入缓停止状态，从当前输出逐步衰减到 0。"""
        self.motion_mode = "ramp_stop"
        self.ramp_stop_start_time = time.time()
        self.ramp_stop_start_currents = self.last_currents.copy()
        self.idle_zero_sent = False
        self.log("开始缓停止")
        self.update_mode_display()

    def control_loop(self):
        """周期性生成并发送电流。"""
        try:
            if self.motion_mode == "idle":
                if not self.idle_zero_sent:
                    self.stop_all()
                return

            if self.motion_mode == "ramp_stop":
                currents = self.generate_ramp_stop_currents()
                self.send_currents(currents)
                self.update_current_display(currents)
                return

            if self.motion_mode == "translate":
                target = self.generate_translation_currents()
            elif self.motion_mode == "rotate":
                target = self.generate_rotation_currents()
            elif self.motion_mode == "rotate_translate":
                target = self.generate_rotate_translate_currents()
            elif self.motion_mode == "pulse":
                target = self.generate_pulse_currents()
            else:
                target = [0, 0, 0, 0, 0, 0]

            target = self.apply_start_ramp(target)
            currents = self.apply_current_slew_limit(target)
            self.send_currents(currents)
            self.update_current_display(currents)
        except Exception as exc:
            self.log(f"控制循环异常：{exc}")

    def generate_translation_currents(self):
        """根据目标方向选择点积最大的单个线圈通电。"""
        phi = math.radians(self.translation_angle_deg)
        ux = math.cos(phi)
        uy = math.sin(phi)

        contributions = [
            ux * dx + uy * dy
            for dx, dy in self.coil_dirs
        ]
        max_idx = max(range(len(contributions)), key=lambda i: contributions[i])

        current_value = min(self.translation_current_spin.value(), self.max_current_spin.value())
        currents = [0, 0, 0, 0, 0, 0]
        currents[max_idx] = current_value
        return currents

    def generate_rotation_currents(self):
        """生成六相正弦旋转电流。"""
        t = time.time() - self.motion_start_time
        omega = 2.0 * math.pi * self.frequency_spin.value()
        phase = -omega * t if self.rotate_direction == "CW" else omega * t

        bias = self.bias_current_spin.value()
        amplitude = self.rotation_amplitude_spin.value()

        currents = []
        for theta_deg in self.coil_angles_deg:
            theta = math.radians(theta_deg)
            value = bias + amplitude * math.cos(phase - theta)
            currents.append(value)
        return self.clamp_currents(currents)

    def generate_rotate_translate_currents(self):
        """生成旋转电流叠加平动偏置。"""
        t = time.time() - self.motion_start_time
        omega = 2.0 * math.pi * self.frequency_spin.value()
        phase = -omega * t if self.rotate_direction == "CW" else omega * t

        bias = self.bias_current_spin.value()
        rot_amp = self.rotation_amplitude_spin.value()
        move_amp = self.move_overlay_spin.value()
        phi = math.radians(self.rotate_translate_angle_deg)

        currents = []
        for theta_deg in self.coil_angles_deg:
            theta = math.radians(theta_deg)
            value = (
                bias
                + rot_amp * math.cos(phase - theta)
                + move_amp * math.cos(phi - theta)
            )
            currents.append(value)
        return self.clamp_currents(currents)

    def generate_pulse_currents(self):
        """生成单线圈点动电流。"""
        current_value = min(self.pulse_current_spin.value(), self.max_current_spin.value())
        currents = [0, 0, 0, 0, 0, 0]
        if 0 <= self.pulse_coil_index < 6:
            currents[self.pulse_coil_index] = current_value
        return currents

    def apply_start_ramp(self, currents):
        """缓启动：从 idle 进入运动模式后按时间比例逐渐放大目标电流。"""
        if not self.is_ramp_enabled():
            return currents

        ramp_time = self.ramp_start_time_spin.value()
        if ramp_time <= 0:
            return currents

        elapsed_time = time.time() - self.motion_start_time
        ramp_factor = min(1.0, max(0.0, elapsed_time / ramp_time))
        return [value * ramp_factor for value in currents]

    def generate_ramp_stop_currents(self):
        """缓停止：从进入缓停瞬间的电流逐渐衰减到 0。"""
        if self.ramp_stop_start_time is None:
            self.motion_mode = "idle"
            self.idle_zero_sent = True
            self.last_currents = [0, 0, 0, 0, 0, 0]
            self.update_mode_display()
            return [0, 0, 0, 0, 0, 0]

        ramp_time = self.ramp_stop_time_spin.value()
        elapsed_time = time.time() - self.ramp_stop_start_time
        ramp_factor = 0.0 if ramp_time <= 0 else max(0.0, 1.0 - elapsed_time / ramp_time)
        currents = [
            max(0, min(self.max_current_spin.value(), int(round(value * ramp_factor))))
            for value in self.ramp_stop_start_currents
        ]

        if ramp_factor <= 0:
            currents = [0, 0, 0, 0, 0, 0]
            self.motion_mode = "idle"
            self.idle_zero_sent = True
            self.ramp_stop_start_time = None
            self.ramp_stop_start_currents = [0, 0, 0, 0, 0, 0]
            self.last_currents = [0, 0, 0, 0, 0, 0]
            self.log("缓停止完成")
            self.update_mode_display()
        else:
            self.last_currents = currents.copy()

        return currents

    def apply_current_slew_limit(self, currents):
        """对六路电流做变化率限制。"""
        self.update_safety_limits()
        limited = []
        for i, target in enumerate(currents):
            target = max(0, min(self.max_current, int(round(target))))
            delta = target - self.last_currents[i]

            if delta > self.max_delta:
                value = self.last_currents[i] + self.max_delta
            elif delta < -self.max_delta:
                value = self.last_currents[i] - self.max_delta
            else:
                value = target

            value = max(0, min(self.max_current, int(round(value))))
            limited.append(value)

        self.last_currents = limited.copy()
        return limited

    def send_currents(self, currents):
        """按旧协议通过串口发送 6 通道电流值命令。"""
        try:
            safe_currents = [
                max(0, min(self.max_current_spin.value(), int(round(value))))
                for value in currents
            ]
            cmd = (
                f"a0:{safe_currents[0]:02d},"
                f"a1:{safe_currents[1]:02d},"
                f"a2:{safe_currents[2]:02d},"
                f"a3:{safe_currents[3]:02d},"
                f"a4:{safe_currents[4]:02d},"
                f"a5:{safe_currents[5]:02d}\r\n"
            )

            if self.ser and self.ser.is_open:
                try:
                    self.ser.write(cmd.encode())
                except Exception as exc:
                    self.log(f"串口发送失败：{exc}")
                    self.serial_status_label.setText("未连接")
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                    self.ser = None

            if hasattr(self, "verbose_command_log_checkbox") and self.verbose_command_log_checkbox.isChecked():
                now = time.time()
                if now - self.last_command_log_time > 0.5:
                    self.log(f"发送：{cmd.strip()}")
                    self.last_command_log_time = now
        except Exception as exc:
            self.log(f"生成串口命令失败：{exc}")

    def update_current_display(self, currents):
        """更新 a0-a5 当前输出电流显示。"""
        for i, value in enumerate(currents):
            if i < len(self.current_labels):
                self.current_labels[i].setText(f"a{i}: {int(value):02d}")

    def log(self, text):
        """追加日志，并限制最大行数。"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_lines.append(f"[{timestamp}] {text}")
        if len(self.log_lines) > 300:
            self.log_lines = self.log_lines[-300:]

        if hasattr(self, "log_edit"):
            self.log_edit.setPlainText("\n".join(self.log_lines))
            self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())

    def update_safety_limits(self):
        """同步 GUI 中的安全参数。"""
        if hasattr(self, "max_current_spin"):
            self.max_current = self.max_current_spin.value()
        if hasattr(self, "max_delta_spin"):
            self.max_delta = self.max_delta_spin.value()

    def is_ramp_enabled(self):
        """读取缓启动/缓停止开关状态。"""
        return hasattr(self, "ramp_enable_checkbox") and self.ramp_enable_checkbox.isChecked()

    def angle_to_name(self, angle_deg):
        """将方向角转换为界面显示文字。"""
        names = {
            0: "右移",
            90: "前进",
            180: "左移",
            270: "后退",
        }
        return names.get(int(angle_deg), f"{angle_deg}°")

    def update_mode_display(self, *args):
        """更新实时监控区中的模式、方向、频率、录像和平滑状态。"""
        if not hasattr(self, "mode_status_label"):
            return

        mode_names = {
            "idle": "停止 / Idle",
            "translate": "平动",
            "rotate": "旋转",
            "rotate_translate": "旋转 + 平动",
            "pulse": "单线圈点动",
            "ramp_stop": "缓停止",
        }
        mode_text = mode_names.get(self.motion_mode, self.motion_mode)

        if self.motion_mode == "translate":
            direction_text = self.angle_to_name(self.translation_angle_deg)
        elif self.motion_mode == "rotate":
            direction_text = "顺时针" if self.rotate_direction == "CW" else "逆时针"
        elif self.motion_mode == "rotate_translate":
            rotate_text = "顺时针" if self.rotate_direction == "CW" else "逆时针"
            direction_text = f"{rotate_text} + {self.angle_to_name(self.rotate_translate_angle_deg)}"
        elif self.motion_mode == "pulse":
            direction_text = f"a{self.pulse_coil_index} 点动"
        elif self.motion_mode == "ramp_stop":
            direction_text = "输出衰减中"
        else:
            direction_text = "-"

        frequency_text = f"{self.frequency_spin.value():.2f} Hz" if hasattr(self, "frequency_spin") else "-"
        record_text = "录制中" if self.recording else "未开始"
        if self.is_ramp_enabled():
            ramp_text = (
                f"开启，启动 {self.ramp_start_time_spin.value():.2f}s / "
                f"停止 {self.ramp_stop_time_spin.value():.2f}s"
            )
        else:
            ramp_text = "关闭"

        self.mode_status_label.setText(f"当前模式：{mode_text}")
        self.direction_status_label.setText(f"方向：{direction_text}")
        self.frequency_status_label.setText(f"频率：{frequency_text}")
        self.record_status_label.setText(f"录像：{record_text}")
        self.ramp_status_label.setText(f"缓启动/缓停止：{ramp_text}")

    def clamp_currents(self, currents):
        """将电流限制在 [0, max_current]。"""
        self.update_safety_limits()
        return [
            max(0, min(self.max_current, int(round(value))))
            for value in currents
        ]

    def closeEvent(self, event):
        """关闭窗口时安全释放资源。"""
        try:
            self.control_timer.stop()
        except Exception:
            pass

        try:
            self.camera_timer.stop()
        except Exception:
            pass

        try:
            if self.recording:
                self.stop_recording()
        except Exception:
            pass

        try:
            self.motion_mode = "idle"
            self.idle_zero_sent = True
            self.last_currents = [0, 0, 0, 0, 0, 0]
            self.send_currents([0, 0, 0, 0, 0, 0])
        except Exception:
            pass

        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.ser = None
        except Exception:
            pass

        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = None
        except Exception:
            pass

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MicroClusterManualControlGUI()
    window.show()
    sys.exit(app.exec())
