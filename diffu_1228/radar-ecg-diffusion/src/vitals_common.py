"""Shared utilities for IWR1443 vital-signs scripts.

This module centralizes duplicated logic across multiple GUI/sender scripts:
- Frame protocol constants and stats indices
- RadarConfig parsing (cfg -> payload size, range axis)
- Frame parsing (bytes -> stats + range_profile)
- Serial reader thread (CLI+Data serial)
- HTTP sender thread (optionally rate limited)

Keep this file dependency-light so scripts can import it directly.
"""

from __future__ import annotations

import time
import struct
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty
from typing import Any, Optional

import numpy as np
import requests
import serial

from PyQt6.QtCore import QThread, pyqtSignal


# --- Frame protocol constants ---
MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"
LENGTH_MAGIC_WORD = 8
LENGTH_HEADER = 40
LENGTH_TLV_HEADER = 8
LENGTH_DEBUG_DATA = 128
MMWDEMO_SEGMENT_LEN = 32


# --- Stats indices (DebugData: 32 * float32) ---
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
    def __init__(self, cfg_path: str):
        self.valid = False
        self.numRangeBinProcessed = 0
        self.rangeStartMeters = 0.2
        self.rangeEndMeters = 1.0
        self.rangeAxis = np.array([])
        self.payload_size = 0
        self.parse(cfg_path)

    def parse(self, cfg_path: str) -> None:
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            adc_samples = 256
            dig_out_rate = 2500
            freq_slope = 60
            start_freq = 77

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == "profileCfg":
                    start_freq = float(parts[2])
                    freq_slope = float(parts[8])
                    adc_samples = int(parts[10])
                    dig_out_rate = int(parts[11])
                elif parts[0] == "vitalSignsCfg":
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

            total = (
                LENGTH_HEADER
                + LENGTH_TLV_HEADER
                + LENGTH_DEBUG_DATA
                + LENGTH_TLV_HEADER
                + (4 * self.numRangeBinProcessed)
            )
            remainder = total % MMWDEMO_SEGMENT_LEN
            if remainder != 0:
                total += MMWDEMO_SEGMENT_LEN - remainder

            self.payload_size = int(total)
            self.valid = True
            print(
                f"Config: RangeBins={self.numRangeBinProcessed}, PayloadSize={self.payload_size}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Config Error: {exc}")
            self.valid = False


def parse_frame_bytes(data: bytes, cfg: RadarConfig) -> dict[str, Any]:
    """Parse one full frame (already aligned to MAGIC_WORD and length)."""

    ptr = LENGTH_HEADER + LENGTH_TLV_HEADER  # 48

    stats_data = data[ptr : ptr + LENGTH_DEBUG_DATA]
    stats = struct.unpack(f"<{LENGTH_DEBUG_DATA // 4}f", stats_data)

    ptr += LENGTH_DEBUG_DATA
    ptr += LENGTH_TLV_HEADER

    rp_len = 4 * cfg.numRangeBinProcessed
    rp_data = data[ptr : ptr + rp_len]

    rp_raw = np.frombuffer(rp_data, dtype=np.int16)
    rp_real = rp_raw[0::2].astype(np.float64)
    rp_imag = rp_raw[1::2].astype(np.float64)
    rp_abs = np.sqrt(rp_real**2 + rp_imag**2)

    return {"stats": stats, "range_profile": rp_abs}


class SerialRadarThread(QThread):
    """Read mmWave VitalSigns frames from serial data port.

    - CLI serial is used to push cfg and start the sensor.
    - Data serial is used to read binary frames.
    """

    packet_received = pyqtSignal(dict)

    def __init__(
        self,
        data_port: str,
        cli_port: str,
        config_obj: RadarConfig,
        cfg_file_path: str,
        cli_baud: int = 115200,
        data_baud: int = 921600,
    ):
        super().__init__()
        self.data_port = data_port
        self.cli_port = cli_port
        self.cfg = config_obj
        self.cfg_file_path = cfg_file_path
        self.cli_baud = cli_baud
        self.data_baud = data_baud
        self.running = False

    def run(self) -> None:
        print(f"--> 尝试打开串口: CLI={self.cli_port}, Data={self.data_port}")
        cli: Optional[serial.Serial] = None
        data_ser: Optional[serial.Serial] = None

        try:
            cli = serial.Serial(self.cli_port, self.cli_baud, timeout=1)
            data_ser = serial.Serial(self.data_port, self.data_baud, timeout=1)
            print("--> 串口打开成功！")

            cli.write(b"sensorStop\n")
            time.sleep(1.0)
            data_ser.reset_input_buffer()

            with open(self.cfg_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("%"):
                        continue
                    print(f"    发送指令: {line}")
                    cli.write((line + "\n").encode())
                    time.sleep(0.05)

            time.sleep(1.0)
            cli.write(b"sensorStart\n")
            time.sleep(0.5)
            cli.write(b"frameStart\n")
            print("--> 发送 frameStart 完成")

            self.running = True
            buffer = bytearray()
            print(f"--> 开始监听数据 (预期包长: {self.cfg.payload_size} 字节)...")

            while self.running:
                waiting = data_ser.in_waiting
                if waiting <= 0:
                    time.sleep(0.005)
                    continue

                chunk = data_ser.read(waiting)
                buffer.extend(chunk)

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

                    try:
                        parsed = parse_frame_bytes(frame_data, self.cfg)
                    except Exception as exc:  # noqa: BLE001
                        print(f"Parse Error: {exc}")
                        parsed = None

                    if parsed:
                        self.packet_received.emit(parsed)

        except Exception as exc:  # noqa: BLE001
            print(f"Serial Error: {exc}")
        finally:
            try:
                if data_ser is not None:
                    data_ser.close()
            except Exception:
                pass
            try:
                if cli is not None:
                    cli.close()
            except Exception:
                pass

    def stop(self) -> None:
        self.running = False
        self.wait()


class HttpSenderThread(QThread):
    """HTTP sender thread.

    Supports:
    - add_data(payload): enqueue a single payload
    - add_batch(payloads): enqueue a list (resets per-batch counter)

    Payload object must provide a `to_json_dict()` method.
    """

    send_progress = pyqtSignal(int, int)  # sent_in_batch, queue_remaining

    def __init__(
        self,
        url: str,
        queue_maxsize: int = 10,
        rate_hz: Optional[float] = None,
        request_timeout: float = 0.5,
    ):
        super().__init__()
        self.url = url
        self.data_queue: Queue[Any] = Queue(maxsize=int(queue_maxsize))
        self.running = True

        self.request_timeout = float(request_timeout)
        self.send_interval = (1.0 / float(rate_hz)) if rate_hz else None

        self.batch_sent_count = 0
        self.total_sent = 0
        self._last_send_time = 0.0

    def add_data(self, payload: Any) -> None:
        if not self.data_queue.full():
            self.data_queue.put(payload)

    def add_batch(self, payloads: list[Any]) -> None:
        self.batch_sent_count = 0
        for p in payloads:
            if not self.data_queue.full():
                self.data_queue.put(p)

    def get_log_time(self) -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def run(self) -> None:
        print(f"[{self.get_log_time()}] [SYSTEM] HTTP 发送线程启动 -> 目标: {self.url}")

        while self.running:
            try:
                payload = self.data_queue.get(timeout=0.1)
            except Empty:
                continue

            if self.send_interval is not None:
                now = time.time()
                elapsed = now - self._last_send_time
                if elapsed < self.send_interval:
                    time.sleep(self.send_interval - elapsed)

            headers = {"Content-Type": "application/json"}
            try:
                start_ts = time.time()
                response = requests.post(
                    self.url,
                    json=payload.to_json_dict(),
                    headers=headers,
                    timeout=self.request_timeout,
                )
                self._last_send_time = time.time()
                cost_ms = (self._last_send_time - start_ts) * 1000.0

                self.total_sent += 1
                self.batch_sent_count += 1
                queue_size = self.data_queue.qsize()
                self.send_progress.emit(self.batch_sent_count, queue_size)

                if response.status_code != 200:
                    print(
                        f"[{self.get_log_time()}] [HTTP FAIL] {response.status_code} | {response.text[:64]}"
                    )
                else:
                    # Keep log minimal to avoid slowing down UI
                    print(f"[{self.get_log_time()}] [HTTP OK] {cost_ms:.0f}ms | #{self.total_sent}")

            except requests.exceptions.Timeout:
                print(f"[{self.get_log_time()}] [HTTP ERROR] timeout")
            except requests.exceptions.ConnectionError:
                print(f"[{self.get_log_time()}] [HTTP ERROR] connection failed")
            except Exception as exc:  # noqa: BLE001
                print(f"[{self.get_log_time()}] [HTTP ERROR] {exc}")

    def stop(self) -> None:
        self.running = False
        self.wait()
