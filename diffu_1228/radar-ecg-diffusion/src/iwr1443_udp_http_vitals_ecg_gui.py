"""IWR1443 生命体征(UDP) + 扩散ECG重建 + HTTP逐点上报（GUI版）

功能概览
- 通过串口(CLI)下发 cfg，雷达数据通过 UDP 接收（替代 data 串口）。
- 实时显示呼吸/心跳/相位/距离像，并用扩散模型重建 ECG（滑窗推理）。
- 将当前时刻的 BR/HR/波形点(含 ECG 最后一个点)通过 HTTP POST 上报后端。

备注
- 本脚本属于“逐点上报”的版本：每收到一帧就上报 1 个点；不做整段 128 点的时间同步发送。
"""

import sys
import os
import time
import struct
import numpy as np
import serial
import requests
import json
import socket
from queue import Queue, Empty
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加项目路径以导入本地模块（避免从别的工作目录启动时 import 失败）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from iwr1443_realtime_ecg_diffusion_gui import (  # noqa: E402
    ECGReconstructor,
    InferenceThread,
    MODEL_INPUT_LENGTH,
)

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

# 解析说明：
# - stats(DebugData) 是 32 个 float32，下面 IDX_* 是对应索引。
# - range_profile 为 complex int16 序列转换后的幅度（abs）。

# --- 后端配置 ---
BACKEND_URL = "http://10.29.211.136:8080/api/ti6843/vital/data/data"
DEVICE_ID = "TI1443_01"

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
    ecgWavePoint: float
    timestamp: str

    def to_json_dict(self):
        return asdict(self)


class DataSenderThread(QThread):
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.data_queue = Queue(maxsize=10)
        self.running = True

    def add_data(self, payload: VitalSignsPayload):
        if not self.data_queue.full():
            self.data_queue.put(payload)

    def get_log_time(self):
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def run(self):
        print(f"[{self.get_log_time()}] [SYSTEM] HTTP 发送线程启动 -> 目标: {self.url}")

        while self.running:
            try:
                payload_obj = self.data_queue.get(timeout=1.0)

                json_data = payload_obj.to_json_dict()
                headers = {'Content-Type': 'application/json'}

                try:
                    start_ts = time.time()
                    response = requests.post(
                        self.url,
                        json=json_data,
                        headers=headers,
                        timeout=0.5
                    )
                    cost_time = (time.time() - start_ts) * 1000

                    if response.status_code == 200:
                        print(f"[{self.get_log_time()}] [HTTP SUCCESS] ✅ 耗时:{cost_time:.0f}ms | Code:200 | "
                              f"HR:{payload_obj.heartRate} BR:{payload_obj.breathRate} "
                              f"Wave(H/B/ECG):{payload_obj.heartWavePoint}/{payload_obj.breathWavePoint}/{getattr(payload_obj, 'ecgWavePoint', 0.0)}")
                    else:
                        print(f"[{self.get_log_time()}] [HTTP FAIL] ❌ 服务器返回: {response.status_code} | "
                              f"响应: {response.text[:50]}")

                except requests.exceptions.Timeout:
                    print(f"[{self.get_log_time()}] [HTTP ERROR] ⏳ 请求超时")
                except requests.exceptions.ConnectionError:
                    print(f"[{self.get_log_time()}] [HTTP ERROR] 🔌 连接失败")
                except Exception as e:
                    print(f"[{self.get_log_time()}] [HTTP ERROR] ⚠️ 未知错误: {e}")

            except Empty:
                continue
            except Exception as e:
                print(f"[{self.get_log_time()}] [THREAD ERROR] {e}")

    def stop(self):
        self.running = False
        self.wait()


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
        self._udp_sock = None
        self._cli = None

    def run(self):
        print(f"--> 打开 CLI 串口: {self.cli_port}")
        print(f"--> 打开 UDP 数据端口: {self.data_port}")
        try:
            # 打开 CLI 串口用于下发配置
            self._cli = serial.Serial(self.cli_port, 115200, timeout=1)
            # 打开 UDP Socket 用于接收数据
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_sock.bind(('0.0.0.0', int(self.data_port)))
            self._udp_sock.settimeout(0.5)
            print("--> CLI 串口 & UDP Socket 打开成功")
            self._cli.write(b'sensorStop\n')
            time.sleep(1.0)
            try:
                with open(self.cfg_file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('%'): continue
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
            last_print_time = time.time()
            while self.running:
                try:
                    try:
                        data, addr = self._udp_sock.recvfrom(4096)
                        buffer.extend(data)
                    except socket.timeout:
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
                    print(f"UDP Loop Error: {e}")
                    time.sleep(0.1)
        except Exception as e:
            print(f"RadarThread Error: {e}")
        finally:
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
        """解析单帧二进制数据（UDP 接收版本）。"""
        ptr = LENGTH_HEADER + LENGTH_TLV_HEADER  # 48
        stats_data = data[ptr: ptr + LENGTH_DEBUG_DATA]
        stats = struct.unpack(f'<{LENGTH_DEBUG_DATA // 4}f', stats_data)

        ptr += LENGTH_DEBUG_DATA
        ptr += LENGTH_TLV_HEADER
        
        rp_len = 4 * self.cfg.numRangeBinProcessed
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
        if not self.cfg.valid: return

        self.init_ui()

        # ECG Inference Setup
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "diffusion_model_final_128_0109.pth")
        self.ecg_reconstructor = ECGReconstructor(model_path, device='cuda', ddim_steps=20)
        self.inference_thread = InferenceThread(self.ecg_reconstructor)
        self.inference_thread.inference_done.connect(self.update_ecg_plot)
        self.inference_thread.start()

        self.inference_data_buf = deque(maxlen=MODEL_INPUT_LENGTH)
        self.inference_new_points = 0
        self.slide_step = 64
        self.latest_ecg_point = 0.0

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

        self.sender_thread = DataSenderThread(BACKEND_URL)
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

        p5 = self.win.addPlot(title="Reconstructed ECG (Diffusion)", row=2, col=0, colspan=2)
        p5.showGrid(x=True, y=True)
        self.curve_ecg = p5.plot(pen=pg.mkPen('r', width=2))
        # p5.setYRange(-2, 2) # Adjust based on normalized ECG range
        p5.setMouseEnabled(x=False, y=False)

    def update_data(self, data):
        stats = data['stats']
        val_phase = stats[IDX_PHASE]
        val_breath = stats[IDX_BREATH_WAVE]
        val_heart = stats[IDX_HEART_WAVE]
        conf_heart_curr = stats[IDX_CONF_HEART]
        conf_breath_curr = stats[IDX_CONF_BREATH]
        energy_breath = stats[IDX_ENERGY_BREATH]
        motion_flag = stats[IDX_MOTION]
        
        # Accumulate data for inference
        self.inference_data_buf.append(val_breath)
        self.inference_new_points += 1
        
        if len(self.inference_data_buf) >= MODEL_INPUT_LENGTH and self.inference_new_points >= self.slide_step:
            if hasattr(self, 'inference_thread'):
                chunk = np.array(list(self.inference_data_buf))
                self.inference_thread.add_data(chunk, self.slide_step)
                self.inference_new_points = 0

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
            ecgWavePoint=float(self.latest_ecg_point),
            timestamp=iso_timestamp
        )

        self.sender_thread.add_data(payload_obj)

    def update_ecg_plot(self, ecg_data, latency, step):
        self.curve_ecg.setData(ecg_data)
        try:
            if ecg_data is not None and len(ecg_data) > 0:
                self.latest_ecg_point = float(ecg_data[-1])
        except Exception:
            pass

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        if self.inference_thread:
            self.inference_thread.stop()
        if self.sender_thread:
            self.sender_thread.stop()
        event.accept()


if __name__ == "__main__":
    DATA_PORT = 9000
    CLI_PORT = 'COM13'
    CFG_FILE = r'D:\mmWave\IWR1443-Radar-Connection\profile_2d_VitalSigns_20fps.cfg'

    app = QApplication(sys.argv)
    window = VitalSignsGUI(DATA_PORT, CLI_PORT, CFG_FILE)
    window.show()
    sys.exit(app.exec())