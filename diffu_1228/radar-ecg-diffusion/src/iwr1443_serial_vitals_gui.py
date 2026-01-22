"""IWR1443 生命体征(串口) GUI

这是最基础的可视化脚本：
- CLI 串口：下发 cfg/启动雷达。
- Data 串口：921600bps 读取二进制帧。
- 解析 vitalSigns Debug TLV 与 Range Profile，并在 GUI 中显示。

不包含 HTTP 上报、也不包含 ECG 重建推理。
"""

import sys
import time
import struct
import numpy as np
import serial
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QGroupBox, QGridLayout)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
import pyqtgraph as pg

# --- 全局常量 ---
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
LENGTH_MAGIC_WORD = 8
LENGTH_HEADER = 40
LENGTH_TLV_HEADER = 8
LENGTH_DEBUG_DATA = 128
MMWDEMO_SEGMENT_LEN = 32
PLOT_DISPLAY_LENGTH = 128

# mmWave VitalSigns Demo 常见帧结构（解析见 RadarThread.parse_frame 的说明）：
# [Header 40B] [TLV Header 8B] [DebugData 128B] [TLV Header 8B] [RangeProfile 4*N] [Padding]

# 数据索引
IDX_PHASE = 4
IDX_BREATH_WAVE = 5
IDX_HEART_WAVE = 6
IDX_HEART_RATE_FFT = 7
IDX_HEART_RATE_PEAK = 10
IDX_BREATH_RATE_FFT = 11
IDX_CONF_BREATH = 14
IDX_CONF_HEART = 16
IDX_ENERGY_BREATH = 19
IDX_ENERGY_HEART = 20
IDX_MOTION = 21


class RadarConfig:
    def __init__(self, cfg_path):
        self.valid = False
        self.numRangeBinProcessed = 0
        self.rangeStartMeters = 0.2
        self.rangeEndMeters = 1.0
        self.rangeAxis = np.array([])
        self.payload_size = 0
        self.parse(cfg_path)

    def parse(self, cfg_path):
        try:
            with open(cfg_path, 'r') as f:
                lines = f.readlines()

            adc_samples = 256
            dig_out_rate = 2500
            freq_slope = 60
            start_freq = 77

            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] == 'profileCfg':
                    start_freq = float(parts[2])
                    freq_slope = float(parts[8])
                    adc_samples = int(parts[10])
                    dig_out_rate = int(parts[11])
                elif parts[0] == 'vitalSignsCfg':
                    self.rangeStartMeters = float(parts[1])
                    self.rangeEndMeters = float(parts[2])

            num_range_bins = 1
            while num_range_bins < adc_samples:
                num_range_bins *= 2

            range_max = (3e8 * dig_out_rate * 1e3) / (2 * freq_slope * 1e12)
            range_bin_size = range_max / num_range_bins

            start_idx = int(self.rangeStartMeters / range_bin_size)
            end_idx = int(self.rangeEndMeters / range_bin_size)
            self.numRangeBinProcessed = end_idx - start_idx + 1
            self.rangeAxis = np.arange(start_idx, end_idx + 1) * range_bin_size

            # 数据包结构 (与MATLAB一致):
            # [Header 40B] [TLV_Header 8B] [DebugData 128B] [TLV_Header 8B] [RangeProfile 4*N bytes]
            total = LENGTH_HEADER + LENGTH_TLV_HEADER + LENGTH_DEBUG_DATA + \
                    LENGTH_TLV_HEADER + (4 * self.numRangeBinProcessed)
            remainder = total % MMWDEMO_SEGMENT_LEN
            if remainder != 0:
                total += (MMWDEMO_SEGMENT_LEN - remainder)

            self.payload_size = int(total)
            self.valid = True
            print(f"Config: RangeBins={self.numRangeBinProcessed}, PayloadSize={self.payload_size}")

        except Exception as e:
            print(f"Config Error: {e}")


class RadarThread(QThread):
    packet_received = pyqtSignal(dict)

    def __init__(self, data_port, cli_port, config_obj, cfg_file_path):
        super().__init__()
        self.data_port = data_port
        self.cli_port = cli_port
        self.cfg = config_obj
        self.cfg_file_path = cfg_file_path
        self.running = False

    def run(self):
        print(f"--> 尝试打开串口: CLI={self.cli_port}, Data={self.data_port}")
        try:
            cli = serial.Serial(self.cli_port, 115200, timeout=1)
            data_ser = serial.Serial(self.data_port, 921600, timeout=1)
            print("--> 串口打开成功！")

            print("--> 发送 sensorStop...")
            cli.write(b'sensorStop\n')
            time.sleep(1.0)
            data_ser.reset_input_buffer()

            print(f"--> 正在发送配置文件: {self.cfg_file_path}")
            try:
                with open(self.cfg_file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('%'):
                            continue
                        print(f"    发送指令: {line}")
                        cli.write((line + '\n').encode())
                        time.sleep(0.05)

                print("--> 配置发送完毕！等待雷达启动...")
                time.sleep(1.0)
                cli.write(b'sensorStart\n')
                time.sleep(0.5)
                cli.write(b'frameStart\n')
                print("--> 发送 frameStart 完成")

            except Exception as e:
                print(f"!!! 读取/发送配置文件失败: {e}")
                return

            self.running = True
            buffer = bytearray()
            print(f"--> 开始监听数据 (预期包长: {self.cfg.payload_size} 字节)...")
            last_print_time = time.time()

            while self.running:
                try:
                    waiting = data_ser.in_waiting
                    if waiting > 0:
                        chunk = data_ser.read(waiting)
                        buffer.extend(chunk)
                        if time.time() - last_print_time > 1.0:
                            last_print_time = time.time()
                    else:
                        time.sleep(0.005)
                        continue

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
                    print(f"Loop Error: {e}")
                    time.sleep(0.1)

        except Exception as e:
            print(f"Serial Error: {e}")

    def parse_frame(self, data):
        """
        解析数据帧 - 按照MATLAB官方代码的顺序
        
        数据包结构:
        [Header 40B] [TLV1_Header 8B] [RangeProfile] [TLV2_Header 8B] [DebugData 128B] [Padding]
        
        但MATLAB读取方式是直接用字节偏移，Debug数据在TLV1之后：
        OFFSET = 48 (Header + TLV_Header)
        stats索引从 OFFSET 开始计算
        """
        # 方法1：按MATLAB的字节偏移方式直接读取
        # MATLAB: OFFSET = LENGTH_HEADER_BYTES - LENGTH_MAGIC_WORD_BYTES + LENGTH_TLV_MESSAGE_HEADER_BYTES
        #                = 40 - 8 + 8 = 40
        # MATLAB: 实际偏移 = OFFSET + LENGTH_TLV_MESSAGE_HEADER_BYTES = 48
        
        # Debug数据紧跟在第一个TLV Header之后
        ptr = LENGTH_HEADER + LENGTH_TLV_HEADER  # 48
        stats_data = data[ptr: ptr + LENGTH_DEBUG_DATA]  # 读取128字节的调试数据
        stats = struct.unpack(f'<{LENGTH_DEBUG_DATA // 4}f', stats_data)  # 32个float
        
        # Range Profile在Debug数据之后
        ptr += LENGTH_DEBUG_DATA  # 48 + 128 = 176
        ptr += LENGTH_TLV_HEADER  # 176 + 8 = 184 (跳过第二个TLV Header)
        
        rp_len = 4 * self.cfg.numRangeBinProcessed  # 每个bin 4字节 (2个int16: real + imag)
        rp_data = data[ptr: ptr + rp_len]
        rp_raw = np.frombuffer(rp_data, dtype=np.int16)
        rp_real = rp_raw[0::2].astype(np.float64)
        rp_imag = rp_raw[1::2].astype(np.float64)
        rp_abs = np.sqrt(rp_real ** 2 + rp_imag ** 2)
        
        return {'stats': stats, 'range_profile': rp_abs}

    def stop(self):
        self.running = False
        self.wait()


class VitalSignsGUI(QMainWindow):
    def __init__(self, data_port, cli_port, cfg_path):
        super().__init__()
        self.setWindowTitle("Vital Signs Demo - Final")
        self.resize(1200, 800)

        self.cfg = RadarConfig(cfg_path)
        if not self.cfg.valid:
            print("Invalid Config")
            return

        self.init_ui()

        # Buffer
        self.breath_buf = np.zeros(PLOT_DISPLAY_LENGTH)
        self.heart_buf = np.zeros(PLOT_DISPLAY_LENGTH)
        self.phase_buf = np.zeros(PLOT_DISPLAY_LENGTH)

        # Smooth
        self.heart_cm = 0.0
        self.breath_cm = 0.0
        self.range_bin_val_avg = 0.0

        # Glitch Filter
        self.prev_breath_val = 0.0
        self.prev_heart_val = 0.0
        self.dataPlotThresh = 50.0

        self.thread = RadarThread(data_port, cli_port, self.cfg, cfg_path)
        self.thread.packet_received.connect(self.update_data)
        self.thread.start()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QGridLayout(central)

        # 1. Stats
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

        # 2. Plots
        pg.setConfigOptions(antialias=True)
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground('w')
        layout.addWidget(self.win, 1, 0, 1, 2)

        # Breathing
        p1 = self.win.addPlot(title="Breathing Waveform", row=0, col=0)
        p1.showGrid(x=True, y=True)
        self.curve_breath = p1.plot(pen=pg.mkPen('b', width=2))
        p1.setXRange(0, PLOT_DISPLAY_LENGTH, padding=0)
        p1.setYRange(-1.5, 1.5)
        p1.getViewBox().disableAutoRange(axis=pg.ViewBox.XAxis)  # 锁定 X 轴
        p1.setMouseEnabled(x=False, y=False)

        # Heart
        p2 = self.win.addPlot(title="Heart Waveform", row=1, col=0)
        p2.showGrid(x=True, y=True)
        self.curve_heart = p2.plot(pen=pg.mkPen('r', width=2))
        p2.setXRange(0, PLOT_DISPLAY_LENGTH, padding=0)
        p2.setYRange(-2, 2)
        p2.getViewBox().disableAutoRange(axis=pg.ViewBox.XAxis)  # 锁定 X 轴
        p2.setMouseEnabled(x=False, y=False)

        # Phase
        p3 = self.win.addPlot(title="Phase (Displacement)", row=0, col=1)
        p3.showGrid(x=True, y=True)
        self.curve_phase = p3.plot(pen=pg.mkPen('k', width=2))
        p3.setXRange(0, PLOT_DISPLAY_LENGTH, padding=0)
        p3.setYRange(-4, 4)
        p3.getViewBox().disableAutoRange(axis=pg.ViewBox.XAxis)  # 锁定 X 轴
        p3.setMouseEnabled(x=False, y=False)

        # Range
        p4 = self.win.addPlot(title="Range Profile", row=1, col=1)
        p4.showGrid(x=True, y=True)
        p4.setLabel('bottom', 'Distance (m)')
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

        # Set Data
        self.curve_breath.setData(plot_breath)
        self.curve_heart.setData(plot_heart)
        self.curve_phase.setData(plot_phase)
        self.curve_range.setData(self.cfg.rangeAxis, data['range_profile'])

        # UI Updates
        range_thresh = 50
        energy_thresh = 0.1

        if (self.range_bin_val_avg < range_thresh) or (energy_breath < energy_thresh):
            self.lbl_breath.setText("BR: --")
            self.lbl_breath.setStyleSheet(self.style_alert)
        else:
            self.lbl_breath.setText(f"BR: {br_display:.1f}")
            self.lbl_breath.setStyleSheet(self.style_normal)

        if (self.range_bin_val_avg < range_thresh) or (self.heart_cm < 0.1):
            self.lbl_heart.setText("HR: --")
            self.lbl_heart.setStyleSheet(self.style_alert)
        else:
            self.lbl_heart.setText(f"HR: {hr_display:.1f}")
            self.lbl_heart.setStyleSheet(self.style_normal)

        if motion_flag > 0:
            self.lbl_motion.setText("Motion Detected")
            self.lbl_motion.setStyleSheet(self.style_alert)
        else:
            self.lbl_motion.setText("Stationary")
            self.lbl_motion.setStyleSheet(self.style_normal)

        self.lbl_debug.setText(f"CM(B/H): {self.breath_cm:.2f} / {self.heart_cm:.2f}")

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        event.accept()


if __name__ == "__main__":
    DATA_PORT = 'COM14'
    CLI_PORT = 'COM13'
    CFG_FILE = r'D:\\mmWave\\IWR1443-Radar-Connection\\profile_2d_VitalSigns_20fps.cfg'

    app = QApplication(sys.argv)
    window = VitalSignsGUI(DATA_PORT, CLI_PORT, CFG_FILE)
    window.show()
    sys.exit(app.exec())