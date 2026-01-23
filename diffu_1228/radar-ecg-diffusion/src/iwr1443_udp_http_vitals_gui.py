"""IWR1443 生命体征(UDP) + HTTP 上报（GUI版）

- CLI 串口用于下发 cfg/启动雷达。
- 雷达数据通过 UDP 接收（通常对应 TI Demo 的 UDP 输出）。
- GUI 展示呼吸/心跳/相位/距离像，并按帧把当前点通过 HTTP POST 上报后端。
"""

import sys
import time
import numpy as np
import serial
import requests
import json
from queue import Queue, Empty
from dataclasses import dataclass, asdict
from datetime import datetime
import socket

from vitals_common import (
    MAGIC_WORD,
    LENGTH_MAGIC_WORD,
    RadarConfig,
    HttpSenderThread,
    parse_frame_bytes,
    IDX_PHASE,
    IDX_BREATH_WAVE,
    IDX_HEART_WAVE,
    IDX_HEART_RATE_FFT,
    IDX_HEART_RATE_PEAK,
    IDX_BREATH_RATE_FFT,
    IDX_CONF_BREATH,
    IDX_CONF_HEART,
    IDX_ENERGY_BREATH,
    IDX_ENERGY_HEART,
    IDX_MOTION,
)


from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QGroupBox, QGridLayout)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
import pyqtgraph as pg

PLOT_DISPLAY_LENGTH = 128

# mmWave VitalSigns Demo（本项目使用的解析方式）常见帧结构（按字节偏移读取）：
# [Header 40B] [TLV1 Header 8B] [DebugData 128B] [TLV2 Header 8B] [RangeProfile 4*N] [Padding]
# 其中 DebugData 以 float32 数组形式存放多种生命体征/置信度/能量等字段。

# --- 后端配置 ---
BACKEND_URL = "http://10.29.211.136:8080/api/ti6843/vital/data/data"
DEVICE_ID = "TI1443_01"

@dataclass
class VitalSignsPayload:
    """
    定义发送给后端的数据包结构
    """
    deviceId: str
    breathRate: int
    heartRate: int
    status: str
    breathWavePoint: float
    heartWavePoint: float
    timestamp: str

    def to_json_dict(self):
        return asdict(self)


DataSenderThread = HttpSenderThread


class RadarThread(QThread):
    packet_received = pyqtSignal(dict)

    def __init__(self, data_port, cli_port, config_obj, cfg_file_path):
        super().__init__()
        self.data_port = data_port      # 现在代表 UDP 端口号（例如 9000）
        self.cli_port = cli_port        # 仍然是雷达 CLI 串口（例如 COM10）
        self.cfg = config_obj
        self.cfg_file_path = cfg_file_path
        self.running = False

        self._udp_sock = None
        self._cli = None

    def run(self):
        print(f"--> 打开 CLI 串口: {self.cli_port}")
        print(f"--> 打开 UDP 数据端口: {self.data_port}")

        try:
            # ===== CLI 串口（下发 cfg，保持不变）=====
            self._cli = serial.Serial(self.cli_port, 115200, timeout=1)

            # ===== UDP Socket（替代 COM25）=====
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_sock.bind(('0.0.0.0', int(self.data_port)))
            self._udp_sock.settimeout(0.5)

            print("--> CLI 串口 & UDP Socket 打开成功")

            # 停止雷达
            self._cli.write(b'sensorStop\n')
            time.sleep(1.0)

            # 下发 cfg
            try:
                with open(self.cfg_file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('%'):
                            continue
                        print(f"    发送指令: {line}")
                        self._cli.write((line + '\n').encode())
                        time.sleep(0.05)

                time.sleep(1.0)
                self._cli.write(b'sensorStart\n')
                time.sleep(0.5)
                self._cli.write(b'frameStart\n')
                print("--> 发送 frameStart 完成")

            except Exception as e:
                print(f"!!! 读取/发送配置文件失败: {e}")
                return

            self.running = True
            buffer = bytearray()

            print(f"--> 开始监听 UDP 数据 (预期包长: {self.cfg.payload_size} 字节)...")

            while self.running:
                try:
                    # UDP 收包（替代 data_ser.read）
                    try:
                        data, addr = self._udp_sock.recvfrom(4096)
                        buffer.extend(data)
                    except socket.timeout:
                        continue

                    # ===== 下面这一段：完全沿用你原来的帧同步/截帧逻辑 =====
                    while True:
                        idx = buffer.find(MAGIC_WORD)
                        if idx == -1:
                            if len(buffer) > 2 * self.cfg.payload_size:
                                buffer = buffer[-LENGTH_MAGIC_WORD:]
                            break

                        required_len = idx + self.cfg.payload_size
                        if len(buffer) < required_len:
                            break

                        frame_data = buffer[idx: required_len]
                        buffer = buffer[required_len:]

                        parsed = self.parse_frame(frame_data)
                        if parsed:
                            self.packet_received.emit(parsed)

                except Exception as e:
                    print(f"UDP Loop Error: {e}")
                    time.sleep(0.1)

        except Exception as e:
            print(f"RadarThread Error: {e}")

        finally:
            # 线程退出时清理资源
            try:
                if self._udp_sock:
                    self._udp_sock.close()
            except:
                pass
            try:
                if self._cli:
                    self._cli.close()
            except:
                pass

    def parse_frame(self, data):
        return parse_frame_bytes(data, self.cfg)

    def stop(self):
        self.running = False
        self.wait()


class VitalSignsGUI(QMainWindow):
    def __init__(self, data_port, cli_port, cfg_path):
        super().__init__()
        self.setWindowTitle("Vital Signs Demo - Final")
        self.resize(1200, 800)

        self.cfg = RadarConfig(cfg_path)
        if not self.cfg.valid: return

        self.init_ui()

        # Buffers
        self.breath_buf = np.zeros(PLOT_DISPLAY_LENGTH)
        self.heart_buf = np.zeros(PLOT_DISPLAY_LENGTH)
        self.phase_buf = np.zeros(PLOT_DISPLAY_LENGTH)
        self.heart_cm = 0.0
        self.breath_cm = 0.0
        self.range_bin_val_avg = 0.0
        self.prev_breath_val = 0.0
        self.prev_heart_val = 0.0
        self.dataPlotThresh = 50.0

        self.sender_thread = DataSenderThread(BACKEND_URL, queue_maxsize=10)
        self.sender_thread.start()
        self.thread = RadarThread(data_port, cli_port, self.cfg, cfg_path)
        self.thread.packet_received.connect(self.update_data)
        self.thread.start()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QGridLayout(central)

        # Stats Area
        stats_box = QGroupBox("Vital Signs Monitor")
        stats_layout = QHBoxLayout()
        stats_box.setLayout(stats_layout)
        self.style_normal = "background-color: white; color: black; font-size: 20px; border: 1px solid gray;"
        self.style_alert = "background-color: red; color: white; font-size: 20px; border: 1px solid gray;"
        self.lbl_breath = QLabel("BR: 0")
        self.lbl_heart = QLabel("HR: 0")
        self.lbl_motion = QLabel("Motion")
        self.lbl_debug = QLabel("CM (B/H): 0.0/0.0")
        for l in [self.lbl_breath, self.lbl_heart, self.lbl_motion]:
            l.setStyleSheet(self.style_normal)
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stats_layout.addWidget(l)
        stats_layout.addWidget(self.lbl_debug)
        layout.addWidget(stats_box, 0, 0, 1, 2)

        # Plots
        pg.setConfigOptions(antialias=True)
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground('w')
        layout.addWidget(self.win, 1, 0, 1, 2)

        p1 = self.win.addPlot(title="Breathing Waveform", row=0, col=0)
        p1.showGrid(x=True, y=True)
        self.curve_breath = p1.plot(pen=pg.mkPen('b', width=2))
        p1.setYRange(-1.5, 1.5)
        p1.setMouseEnabled(x=False, y=False)

        p2 = self.win.addPlot(title="Heart Waveform", row=1, col=0)
        p2.showGrid(x=True, y=True)
        self.curve_heart = p2.plot(pen=pg.mkPen('r', width=2))
        p2.setYRange(-2, 2)
        p2.setMouseEnabled(x=False, y=False)

        p3 = self.win.addPlot(title="Phase (Displacement)", row=0, col=1)
        p3.showGrid(x=True, y=True)
        self.curve_phase = p3.plot(pen=pg.mkPen('k', width=2))
        p3.setYRange(-4, 4)
        p3.setMouseEnabled(x=False, y=False)

        p4 = self.win.addPlot(title="Range Profile", row=1, col=1)
        p4.showGrid(x=True, y=True)
        self.curve_range = p4.plot(pen=pg.mkPen('g', width=2))
        p4.setYRange(0, 150000)
        p4.setMouseEnabled(x=False, y=False)

    def update_data(self, data):
        stats = data['stats']
        val_phase = stats[IDX_PHASE]
        val_breath = stats[IDX_BREATH_WAVE]
        val_heart = stats[IDX_HEART_WAVE]
        conf_heart_curr = stats[IDX_CONF_HEART]
        conf_breath_curr = stats[IDX_CONF_BREATH]
        energy_breath = stats[IDX_ENERGY_BREATH]
        motion_flag = stats[IDX_MOTION]

        # Smooth
        alpha = 0.5
        self.heart_cm = alpha * conf_heart_curr + (1 - alpha) * self.heart_cm
        self.breath_cm = alpha * conf_breath_curr + (1 - alpha) * self.breath_cm
        curr_range_max = np.max(data['range_profile']) if len(data['range_profile']) > 0 else 0
        self.range_bin_val_avg = 0.1 * curr_range_max + 0.9 * self.range_bin_val_avg

        # Update Buffers
        self.breath_buf = np.roll(self.breath_buf, -1)
        self.breath_buf[-1] = val_breath
        self.heart_buf = np.roll(self.heart_buf, -1)
        self.heart_buf[-1] = val_heart
        self.phase_buf = np.roll(self.phase_buf, -1)
        self.phase_buf[-1] = val_phase

        plot_breath = self.breath_buf - np.mean(self.breath_buf)
        if abs(plot_breath[-1] - self.prev_breath_val) > self.dataPlotThresh:
            plot_breath[-1] = self.prev_breath_val
        else:
            self.prev_breath_val = plot_breath[-1]

        plot_heart = self.heart_buf
        if abs(plot_heart[-1] - self.prev_heart_val) > self.dataPlotThresh:
            plot_heart[-1] = self.prev_heart_val
        else:
            self.prev_heart_val = plot_heart[-1]

        plot_phase = self.phase_buf - np.mean(self.phase_buf)

        # Decision Logic
        hr_fft = stats[IDX_HEART_RATE_FFT]
        hr_peak = stats[IDX_HEART_RATE_PEAK]
        diff_est = abs(hr_fft - hr_peak)
        if (self.heart_cm > 0.25) or (diff_est < 10):
            hr_display = hr_fft
        else:
            hr_display = hr_peak
        br_display = stats[IDX_BREATH_RATE_FFT]

        # Set Data to Plots
        self.curve_breath.setData(plot_breath)
        self.curve_heart.setData(plot_heart)
        self.curve_phase.setData(plot_phase)
        self.curve_range.setData(self.cfg.rangeAxis, data['range_profile'])

        # UI Updates
        range_thresh = 50
        energy_thresh = 0.1

        is_valid = True

        if (self.range_bin_val_avg < range_thresh) or (energy_breath < energy_thresh):
            self.lbl_breath.setText("BR: --")
            self.lbl_breath.setStyleSheet(self.style_alert)
            is_valid = False
        else:
            self.lbl_breath.setText(f"BR: {br_display:.1f}")
            self.lbl_breath.setStyleSheet(self.style_normal)

        if (self.range_bin_val_avg < range_thresh) or (self.heart_cm < 0.1):
            self.lbl_heart.setText("HR: --")
            self.lbl_heart.setStyleSheet(self.style_alert)
            is_valid = False
        else:
            self.lbl_heart.setText(f"HR: {hr_display:.1f}")
            self.lbl_heart.setStyleSheet(self.style_normal)

        if motion_flag > 0:
            status_text = "Motion Detected"
            self.lbl_motion.setText(status_text)
            self.lbl_motion.setStyleSheet(self.style_alert)
        else:
            status_text = "Stationary"
            self.lbl_motion.setText(status_text)
            self.lbl_motion.setStyleSheet(self.style_normal)

        self.lbl_debug.setText(f"CM(B/H): {self.breath_cm:.2f} / {self.heart_cm:.2f}")

        current_time_float = time.time()
        iso_timestamp = datetime.now().isoformat()


        # 准备数据
        final_br = int(round(br_display)) if is_valid else 0
        final_hr = int(round(hr_display)) if is_valid else 0
        final_br_wave = round(float(plot_breath[-1]), 2)
        final_hr_wave = round(float(plot_heart[-1]), 2)

        payload_obj = VitalSignsPayload(
            deviceId=DEVICE_ID,
            breathRate=final_br,
            heartRate=final_hr,
            status=status_text,
            breathWavePoint=final_br_wave,
            heartWavePoint=final_hr_wave,
            timestamp=iso_timestamp
        )

        self.sender_thread.add_data(payload_obj)

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        if self.sender_thread:
            self.sender_thread.stop()
        event.accept()


if __name__ == "__main__":
    DATA_PORT = 9000          # ← 原 COM25 改为 UDP 端口
    CLI_PORT = 'COM10'        # ← 雷达 CLI 串口不变
    CFG_FILE = r'D:\PG\Product development\数据处理\profile_2d_VitalSigns_20fps.cfg'

    app = QApplication(sys.argv)
    window = VitalSignsGUI(DATA_PORT, CLI_PORT, CFG_FILE)
    window.show()
    sys.exit(app.exec())
