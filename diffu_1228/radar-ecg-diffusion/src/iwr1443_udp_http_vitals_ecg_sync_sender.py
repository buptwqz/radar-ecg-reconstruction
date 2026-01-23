"""IWR1443 生命体征(UDP) + ECG 重建 + 128点整段同步 HTTP 发送

该脚本是“整段同步发送”的版本：
- 后台不断接收 UDP 雷达帧（CLI 串口仅用于下发 cfg/启动雷达）。
- 累积到 128 点后触发一次扩散模型推理得到 ECG 段。
- 对 breath/heart/ecg 三段做去均值+标准化，然后以 20Hz 速率逐点发送到后端。

相比 iwr1443_udp_http_vitals_ecg_gui.py：本脚本会把 ECG 段与对应的 breath/heart 段对齐后再发送。
"""

import argparse
import os
import sys
import signal
import time
import struct
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from queue import Queue, Empty

import numpy as np
import serial
import requests
import socket

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QGridLayout,
)
from PyQt6.QtCore import QObject, QThread, pyqtSlot, pyqtSignal, Qt
import pyqtgraph as pg

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
    IDX_MOTION,
)


# --- 全局常量（雷达协议） ---
# MAGIC_WORD/LENGTH_* 以及 IDX_* 已迁移到 vitals_common.py


# 模型输入长度（与你的 128 点模型一致）
MODEL_INPUT_LENGTH = 128
PLOT_DISPLAY_LENGTH = MODEL_INPUT_LENGTH


# Ensure local modules are importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# 复用现有推理/模型封装（扩散模型 + 推理线程）
from iwr1443_realtime_ecg_diffusion_gui import ECGReconstructor, InferenceThread  # noqa: E402


@dataclass
class VitalSignsPayload:
    deviceId: str
    breathRate: int
    heartRate: int
    status: str
    ecgValue: float
    breathWavePoint: float
    heartWavePoint: float
    timestamp: str

    def to_json_dict(self) -> dict:
        return asdict(self)


# 统一复用公共发送线程实现（支持 add_batch + 20Hz 发送）
DataSenderThread = HttpSenderThread


class RadarThread(QThread):
    packet_received = pyqtSignal(dict)

    def __init__(self, data_port: str, cli_port: str, config_obj: RadarConfig, cfg_file_path: str):
        super().__init__()
        self.data_port = data_port
        self.cli_port = cli_port
        self.cfg = config_obj
        self.cfg_file_path = cfg_file_path
        self.running = False
        self._udp_sock = None
        self._cli = None

    def run(self) -> None:
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

            self._cli.write(b"sensorStop\n")
            time.sleep(1.0)

            try:
                with open(self.cfg_file_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("%"):
                            continue
                        self._cli.write((line + "\n").encode())
                        time.sleep(0.05)

                time.sleep(1.0)
                self._cli.write(b"sensorStart\n")
                time.sleep(0.5)
                self._cli.write(b"frameStart\n")
                print("--> 发送 frameStart 完成")
            except Exception as e:
                print(f"!!! 读取/发送配置文件失败: {e}")
                return

            self.running = True
            buffer = bytearray()
            print(f"--> 开始监听 UDP 数据 (预期包长: {self.cfg.payload_size} 字节)...")

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

                        frame_data = buffer[idx:required_len]
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

    def parse_frame(self, data: bytes) -> dict:
        """解析单帧数据，返回 stats 与 range_profile。"""
        return parse_frame_bytes(data, self.cfg)

    def stop(self) -> None:
        self.running = False
        self.wait()


class SyncCoordinator(QObject):
    """检测逻辑 + ECG 推理 + 缓冲同步发送。

    检测/显示逻辑来源于 iwr1443_serial_http_vitals_gui.py / iwr1443_udp_http_vitals_gui.py 的实现风格。
    """

    # GUI 更新信号
    wave_updated = pyqtSignal(float, float, float)  # phase, breath, heart
    status_updated = pyqtSignal(int, int, str, float, float)  # br, hr, status, cm_b, cm_h
    ecg_updated = pyqtSignal(np.ndarray)

    def __init__(
        self,
        sender_thread: DataSenderThread,
        inference_thread: InferenceThread,
        device_id: str,
    ):
        super().__init__()
        self.sender_thread = sender_thread
        self.inference_thread = inference_thread
        self.device_id = device_id

        # buffers
        self.phase_buf = deque(maxlen=PLOT_DISPLAY_LENGTH)
        self.breath_buf = deque(maxlen=PLOT_DISPLAY_LENGTH)
        self.heart_buf = deque(maxlen=PLOT_DISPLAY_LENGTH)

        # inference trigger buffers
        self.breath_segment_buf = deque(maxlen=MODEL_INPUT_LENGTH * 4)
        self.heart_segment_buf = deque(maxlen=MODEL_INPUT_LENGTH * 4)

        self.pending_breath_segment: np.ndarray | None = None
        self.pending_heart_segment: np.ndarray | None = None
        self.pending_metadata: dict | None = None
        self.waiting_for_inference = False
        self.samples_since_last_trigger = MODEL_INPUT_LENGTH

        # detection state
        self.heart_cm = 0.0
        self.breath_cm = 0.0
        self.range_bin_val_avg = 0.0
        self.prev_breath_val = 0.0
        self.prev_heart_val = 0.0
        self.dataPlotThresh = 50.0

        self.last_breath_rate = 0
        self.last_heart_rate = 0
        self.last_status = "Initializing"

    @pyqtSlot(dict)
    def handle_packet(self, packet: dict) -> None:
        stats = packet["stats"]
        range_profile = packet.get("range_profile", np.array([]))

        val_phase = float(stats[IDX_PHASE])
        val_breath = float(stats[IDX_BREATH_WAVE])
        val_heart = float(stats[IDX_HEART_WAVE])

        conf_heart_curr = float(stats[IDX_CONF_HEART])
        conf_breath_curr = float(stats[IDX_CONF_BREATH])
        energy_breath = float(stats[IDX_ENERGY_BREATH])
        motion_flag = float(stats[IDX_MOTION])

        # Smooth confidence
        alpha = 0.5
        self.heart_cm = alpha * conf_heart_curr + (1 - alpha) * self.heart_cm
        self.breath_cm = alpha * conf_breath_curr + (1 - alpha) * self.breath_cm

        curr_range_max = float(np.max(range_profile)) if range_profile.size else 0.0
        self.range_bin_val_avg = 0.1 * curr_range_max + 0.9 * self.range_bin_val_avg

        # update plot buffers
        self.phase_buf.append(val_phase)
        self.breath_buf.append(val_breath)
        self.heart_buf.append(val_heart)

        # update segment buffers
        self.breath_segment_buf.append(val_breath)
        self.heart_segment_buf.append(val_heart)
        self.samples_since_last_trigger += 1

        # plotting preprocess（与 iwr1443_serial_http_vitals_gui.py 同风格）
        plot_breath = np.array(self.breath_buf, dtype=np.float32)
        plot_breath = plot_breath - float(np.mean(plot_breath)) if plot_breath.size else plot_breath
        if plot_breath.size:
            if abs(float(plot_breath[-1]) - float(self.prev_breath_val)) > self.dataPlotThresh:
                plot_breath[-1] = float(self.prev_breath_val)
            else:
                self.prev_breath_val = float(plot_breath[-1])

        plot_heart = np.array(self.heart_buf, dtype=np.float32)
        if plot_heart.size:
            if abs(float(plot_heart[-1]) - float(self.prev_heart_val)) > self.dataPlotThresh:
                plot_heart[-1] = float(self.prev_heart_val)
            else:
                self.prev_heart_val = float(plot_heart[-1])

        # decision logic（与 iwr1443_serial_http_vitals_gui.py 同逻辑）
        hr_fft = float(stats[IDX_HEART_RATE_FFT])
        hr_peak = float(stats[IDX_HEART_RATE_PEAK])
        diff_est = abs(hr_fft - hr_peak)
        if (self.heart_cm > 0.25) or (diff_est < 10):
            heart_rate_est = hr_fft
        else:
            heart_rate_est = hr_peak
        breath_rate_est = float(stats[IDX_BREATH_RATE_FFT])

        status_text = "Stationary" if motion_flag <= 0 else "Motion"

        valid_signal = (
            self.range_bin_val_avg >= 50.0
            and energy_breath >= 0.1
            and self.heart_cm >= 0.1
        )

        if valid_signal:
            self.last_heart_rate = int(round(heart_rate_est))
            self.last_breath_rate = int(round(breath_rate_est))
        else:
            self.last_heart_rate = 0
            self.last_breath_rate = 0

        self.last_status = status_text

        self.wave_updated.emit(val_phase, val_breath, val_heart)
        self.status_updated.emit(
            self.last_breath_rate,
            self.last_heart_rate,
            self.last_status,
            float(self.breath_cm),
            float(self.heart_cm),
        )

        if (
            len(self.breath_segment_buf) >= MODEL_INPUT_LENGTH
            and not self.waiting_for_inference
            and self.samples_since_last_trigger >= MODEL_INPUT_LENGTH
        ):
            self.trigger_inference()

    def trigger_inference(self) -> None:
        radar_segment = np.array(self.breath_segment_buf, dtype=np.float32)[-MODEL_INPUT_LENGTH:]
        self.pending_breath_segment = radar_segment.copy()
        self.pending_heart_segment = np.array(self.heart_segment_buf, dtype=np.float32)[-MODEL_INPUT_LENGTH:].copy()

        self.pending_metadata = {
            "breath_rate": self.last_breath_rate,
            "heart_rate": self.last_heart_rate,
            "status": self.last_status,
        }
        self.waiting_for_inference = True
        self.samples_since_last_trigger = 0

        # 推理线程里会走 ECGReconstructor.preprocess（标准化+长度处理）
        self.inference_thread.add_data(radar_segment, MODEL_INPUT_LENGTH)

    @pyqtSlot(object, float, int)
    def on_inference_done(self, ecg_data, inference_time: float, _slide_step: int) -> None:
        if not self.pending_metadata or self.pending_breath_segment is None or self.pending_heart_segment is None:
            self.waiting_for_inference = False
            return

        ecg_array = np.asarray(ecg_data, dtype=np.float32).reshape(-1)
        self.ecg_updated.emit(ecg_array)

        # 与 iwr1443_udp_http_vitals_ecg_gui.py 一致：整段去均值+标准化后逐点发送
        breath_processed = self.pending_breath_segment - np.mean(self.pending_breath_segment)
        std_breath = float(np.std(breath_processed))
        if std_breath > 1e-6:
            breath_processed = breath_processed / std_breath

        heart_processed = self.pending_heart_segment - np.mean(self.pending_heart_segment)
        std_heart = float(np.std(heart_processed))
        if std_heart > 1e-6:
            heart_processed = heart_processed / std_heart

        ecg_processed = ecg_array - float(np.mean(ecg_array))
        std_ecg = float(np.std(ecg_processed))
        if std_ecg > 1e-6:
            ecg_processed = ecg_processed / std_ecg

        br = int(self.pending_metadata["breath_rate"])
        hr = int(self.pending_metadata["heart_rate"])
        status = str(self.pending_metadata["status"])

        base_time = datetime.now()
        payloads: list[VitalSignsPayload] = []
        num_points = int(min(len(ecg_processed), len(breath_processed), len(heart_processed)))
        for i in range(num_points):
            payloads.append(
                VitalSignsPayload(
                    deviceId=self.device_id,
                    breathRate=br,
                    heartRate=hr,
                    status=status,
                    ecgValue=round(float(ecg_processed[i]), 4),
                    breathWavePoint=round(float(breath_processed[i]), 4),
                    heartWavePoint=round(float(heart_processed[i]), 4),
                    timestamp=base_time.isoformat(),
                )
            )

        self.sender_thread.add_batch(payloads)
        print(
            f"[SYNC] Queued {num_points} points | BR:{br} HR:{hr} "
            f"({inference_time*1000:.0f}ms inference)"
        )

        self.pending_breath_segment = None
        self.pending_heart_segment = None
        self.pending_metadata = None
        self.waiting_for_inference = False


class VitalSignsGUI(QMainWindow):
    def __init__(self, coordinator: SyncCoordinator, range_axis: np.ndarray):
        super().__init__()
        self.setWindowTitle("IWR1443 Vital Signs + ECG Sync Sender")
        self.resize(1400, 900)

        self.coordinator = coordinator
        self.range_axis = range_axis

        # display buffers
        self.breath_buf = np.zeros(PLOT_DISPLAY_LENGTH, dtype=np.float32)
        self.heart_buf = np.zeros(PLOT_DISPLAY_LENGTH, dtype=np.float32)
        self.phase_buf = np.zeros(PLOT_DISPLAY_LENGTH, dtype=np.float32)
        self.ecg_buf = np.zeros(PLOT_DISPLAY_LENGTH, dtype=np.float32)

        self.inference_count = 0

        self.init_ui()

        coordinator.wave_updated.connect(self.on_wave)
        coordinator.status_updated.connect(self.on_status)
        coordinator.ecg_updated.connect(self.on_ecg)

    def init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        stats_box = QGroupBox("生命体征监测")
        stats_layout = QHBoxLayout()
        stats_box.setLayout(stats_layout)

        self.style_normal = "background-color: white; color: black; font-size: 20px; border: 1px solid gray; padding: 8px;"
        self.style_alert = "background-color: red; color: white; font-size: 20px; border: 1px solid gray; padding: 8px;"
        self.style_good = "background-color: #4CAF50; color: white; font-size: 20px; border: 1px solid gray; padding: 8px;"

        self.lbl_breath = QLabel("BR: --")
        self.lbl_heart = QLabel("HR: --")
        self.lbl_status = QLabel("状态: 初始化")
        self.lbl_debug = QLabel("CM(B/H): 0.00/0.00")
        self.lbl_infer = QLabel("推理: 0次")
        self.lbl_queue = QLabel("待发送: 0")

        for lbl in [self.lbl_breath, self.lbl_heart, self.lbl_status]:
            lbl.setStyleSheet(self.style_normal)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumWidth(180)
            stats_layout.addWidget(lbl)

        stats_layout.addWidget(self.lbl_debug)
        stats_layout.addWidget(self.lbl_infer)
        stats_layout.addWidget(self.lbl_queue)
        stats_layout.addStretch()

        layout.addWidget(stats_box)

        pg.setConfigOptions(antialias=True)
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground("w")
        layout.addWidget(self.win)

        p1 = self.win.addPlot(title="Breathing Waveform", row=0, col=0)
        p1.showGrid(x=True, y=True, alpha=0.3)
        self.curve_breath = p1.plot(pen=pg.mkPen(color="#2196F3", width=2))
        p1.setXRange(0, PLOT_DISPLAY_LENGTH)

        p2 = self.win.addPlot(title="Heart Waveform", row=1, col=0)
        p2.showGrid(x=True, y=True, alpha=0.3)
        self.curve_heart = p2.plot(pen=pg.mkPen(color="#4CAF50", width=2))
        p2.setXRange(0, PLOT_DISPLAY_LENGTH)

        p3 = self.win.addPlot(title="Phase (Displacement)", row=0, col=1)
        p3.showGrid(x=True, y=True, alpha=0.3)
        self.curve_phase = p3.plot(pen=pg.mkPen(color="#000000", width=2))
        p3.setXRange(0, PLOT_DISPLAY_LENGTH)

        p4 = self.win.addPlot(title="Range Profile", row=1, col=1)
        p4.showGrid(x=True, y=True, alpha=0.3)
        self.curve_range = p4.plot(pen=pg.mkPen(color="#9C27B0", width=2))

        p5 = self.win.addPlot(title="Reconstructed ECG", row=2, col=0, colspan=2)
        p5.showGrid(x=True, y=True, alpha=0.3)
        self.curve_ecg = p5.plot(pen=pg.mkPen(color="#F44336", width=2.5))
        p5.setXRange(0, PLOT_DISPLAY_LENGTH)

    @pyqtSlot(float, float, float)
    def on_wave(self, phase_val: float, breath_val: float, heart_val: float) -> None:
        self.phase_buf = np.roll(self.phase_buf, -1)
        self.phase_buf[-1] = phase_val

        self.breath_buf = np.roll(self.breath_buf, -1)
        self.breath_buf[-1] = breath_val

        self.heart_buf = np.roll(self.heart_buf, -1)
        self.heart_buf[-1] = heart_val

        plot_breath = self.breath_buf - float(np.mean(self.breath_buf))
        std_breath = float(np.std(plot_breath))
        if std_breath > 1e-6:
            plot_breath = plot_breath / std_breath

        plot_heart = self.heart_buf - float(np.mean(self.heart_buf))
        std_heart = float(np.std(plot_heart))
        if std_heart > 1e-6:
            plot_heart = plot_heart / std_heart

        plot_phase = self.phase_buf - float(np.mean(self.phase_buf))
        std_phase = float(np.std(plot_phase))
        if std_phase > 1e-6:
            plot_phase = plot_phase / std_phase

        self.curve_breath.setData(plot_breath)
        self.curve_heart.setData(plot_heart)
        self.curve_phase.setData(plot_phase)

    @pyqtSlot(int, int, str, float, float)
    def on_status(self, br: int, hr: int, status: str, cm_b: float, cm_h: float) -> None:
        if br > 0:
            self.lbl_breath.setText(f"BR: {br}")
            self.lbl_breath.setStyleSheet(self.style_good)
        else:
            self.lbl_breath.setText("BR: --")
            self.lbl_breath.setStyleSheet(self.style_alert)

        if hr > 0:
            self.lbl_heart.setText(f"HR: {hr}")
            self.lbl_heart.setStyleSheet(self.style_good)
        else:
            self.lbl_heart.setText("HR: --")
            self.lbl_heart.setStyleSheet(self.style_alert)

        self.lbl_status.setText(f"状态: {status}")
        if status == "Stationary":
            self.lbl_status.setStyleSheet(self.style_good)
        elif status == "Motion":
            self.lbl_status.setStyleSheet(self.style_alert)
        else:
            self.lbl_status.setStyleSheet(self.style_normal)

        self.lbl_debug.setText(f"CM(B/H): {cm_b:.2f}/{cm_h:.2f}")

    @pyqtSlot(np.ndarray)
    def on_ecg(self, ecg: np.ndarray) -> None:
        self.inference_count += 1
        if len(ecg) >= PLOT_DISPLAY_LENGTH:
            self.ecg_buf = ecg[-PLOT_DISPLAY_LENGTH:]
        else:
            shift = len(ecg)
            self.ecg_buf = np.roll(self.ecg_buf, -shift)
            self.ecg_buf[-shift:] = ecg

        plot_ecg = self.ecg_buf - float(np.mean(self.ecg_buf))
        std_ecg = float(np.std(plot_ecg))
        if std_ecg > 1e-6:
            plot_ecg = plot_ecg / std_ecg

        self.curve_ecg.setData(plot_ecg)
        self.lbl_infer.setText(f"推理: {self.inference_count}次")

    @pyqtSlot(int, int)
    def on_send_progress(self, sent_in_batch: int, queue_remaining: int) -> None:
        self.lbl_queue.setText(f"发送: {sent_in_batch}/{MODEL_INPUT_LENGTH} | 队列: {queue_remaining}")


def main() -> int:
    parser = argparse.ArgumentParser(description="IWR1443 Vital Signs + ECG inference + buffered sync sender")
    parser.add_argument("--data-port", default="9000", help="Radar UDP data port")
    parser.add_argument("--cli-port", default="COM13", help="Radar CLI serial port")
    parser.add_argument(
        "--cfg",
        default=r"D:\\mmWave\\ecg-reconstruction\\profile_2d_VitalSigns_20fps.cfg",
        help="Radar configuration file",
    )
    parser.add_argument(
        "--model",
        default=r"D:\\mmWave\\ecg-reconstruction\\diffu_1228\\radar-ecg-diffusion\\src\\checkpoints\\diffusion_model_final_128_0109.pth",
        help="ECG diffusion model path",
    )
    parser.add_argument(
        "--backend-url",
        default="http://10.29.211.136:8080/api/ti6843/vital/data/data",
        help="HTTP endpoint",
    )
    parser.add_argument("--device-id", default="TI1443_01", help="Device identifier")
    parser.add_argument("--ddim-steps", type=int, default=20, help="DDIM sampling steps")
    args = parser.parse_args()

    cfg_path = args.cfg
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(CURRENT_DIR, cfg_path)

    print("=" * 60)
    print("IWR1443 Vital Signs + ECG Sync Sender")
    print("=" * 60)
    print(f"数据串口: {args.data_port}")
    print(f"配置串口: {args.cli_port}")
    print(f"配置文件: {cfg_path}")
    print(f"模型路径: {args.model}")
    print(f"后端地址: {args.backend_url}")
    print(f"设备ID: {args.device_id}")
    print(f"模型输入长度: {MODEL_INPUT_LENGTH}")
    print("=" * 60)

    cfg = RadarConfig(cfg_path)
    if not cfg.valid:
        print("[ERROR] Radar config invalid")
        return 1

    reconstructor = ECGReconstructor(args.model, ddim_steps=args.ddim_steps)
    if not reconstructor.ready:
        print("[ERROR] Failed to load ECG model")
        return 1

    sender_thread = DataSenderThread(
        args.backend_url,
        queue_maxsize=4096,
        rate_hz=20.0,
        request_timeout=1.0,
    )
    sender_thread.start()

    inference_thread = InferenceThread(reconstructor)
    inference_thread.start()

    coordinator = SyncCoordinator(sender_thread, inference_thread, args.device_id)

    radar_thread = RadarThread(args.data_port, args.cli_port, cfg, cfg_path)
    radar_thread.packet_received.connect(coordinator.handle_packet)
    inference_thread.inference_done.connect(coordinator.on_inference_done)

    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    window = VitalSignsGUI(coordinator, cfg.rangeAxis)
    sender_thread.send_progress.connect(window.on_send_progress)
    window.show()

    radar_thread.start()

    try:
        exit_code = app.exec()
    finally:
        radar_thread.stop()
        inference_thread.stop()
        sender_thread.stop()

    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
