"""
ECG推理0122
实时雷达数据读取与ECG重建推理
- 从IWR1443雷达读取呼吸波形数据
- 累积指定长度后触发扩散模型推理（默认128点）
- 流式展示雷达信号和重建的ECG

说明
- 本文件同时提供可复用的推理封装：ECGReconstructor / InferenceThread。
- 其他脚本（如 iwr1443_udp_http_vitals_ecg_gui.py）会 import 这些类来复用推理逻辑。
"""

import sys
import os
import time
import struct
import numpy as np
import serial
import serial.tools.list_ports
import torch
from collections import deque
from threading import Thread, Lock
from queue import Queue

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QGroupBox, QGridLayout,
                             QPushButton, QSpinBox, QComboBox)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
import pyqtgraph as pg

# 解析器版本标识：用于确认运行的是最新代码
PARSER_BUILD = "2026-01-22a"

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.unet import ConditionalUNet
from models.scheduler import DiffusionScheduler
from inference.sampler import ddim_sample

# --- 全局常量 ---
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
LENGTH_MAGIC_WORD = 8
LENGTH_HEADER = 40
LENGTH_TLV_HEADER = 8
LENGTH_DEBUG_DATA = 128
MMWDEMO_SEGMENT_LEN = 32

# 模型输入长度（与训练数据一致）
MODEL_INPUT_LENGTH = 128
# 滑动窗口步长（每收集多少新点进行一次推理）
SLIDE_STEP = 64
# 显示长度
PLOT_DISPLAY_LENGTH = 128

"""注意：vitalSigns 固件输出的 stats 是 C 结构体 VitalSignsDemo_OutputStats（见固件 common/mmw_output.h）。
它不是简单的 32 个 float 数组；其中前面包含 uint16/uint32 字段。
因此 Python 端必须按结构体布局解析并按字段名取值。
"""


class RadarConfig:
    """雷达配置解析"""
    def __init__(self, cfg_path):
        self.valid = False
        self.numRangeBinProcessed = 0
        self.rangeStartMeters = 0.2
        self.rangeEndMeters = 1.0
        self.rangeAxis = np.array([])
        self.payload_size = 0
        self.frame_periodicity_ms = None
        self.frame_rate_hz = None
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
                if not parts:
                    continue
                if parts[0] == 'profileCfg':
                    start_freq = float(parts[2])
                    freq_slope = float(parts[8])
                    adc_samples = int(parts[10])
                    dig_out_rate = int(parts[11])
                elif parts[0] == 'vitalSignsCfg':
                    self.rangeStartMeters = float(parts[1])
                    self.rangeEndMeters = float(parts[2])
                elif parts[0] == 'frameCfg':
                    # frameCfg <chirpStart> <chirpEnd> <numLoops> <numFrames> <framePeriodicity(ms)> <triggerSel> <frameTriggerDelay>
                    # 这里只需要第5个参数
                    try:
                        self.frame_periodicity_ms = float(parts[5])
                        if self.frame_periodicity_ms > 0:
                            self.frame_rate_hz = 1000.0 / self.frame_periodicity_ms
                    except Exception:
                        pass

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
            fps_info = f", FPS≈{self.frame_rate_hz:.2f}" if self.frame_rate_hz else ""
            print(f"Config: RangeBins={self.numRangeBinProcessed}, PayloadSize={self.payload_size}{fps_info}")

        except Exception as e:
            print(f"Config Error: {e}")


class ECGReconstructor:
    """ECG重建器 - 使用扩散模型"""
    
    def __init__(self, model_path, device='cuda', ddim_steps=20):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.ddim_steps = ddim_steps
        self.model = None
        self.scheduler = None
        self.ready = False
        
        print(f"[ECG重建器] 使用设备: {self.device}")
        print(f"[ECG重建器] DDIM步数: {ddim_steps}")
        
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """加载模型"""
        try:
            print(f"[ECG重建器] 加载模型: {model_path}")
            self.model = ConditionalUNet(base_ch=64, time_dim=256).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            self.scheduler = DiffusionScheduler(steps=1000, device=self.device)
            self.ready = True
            print("[ECG重建器] 模型加载完成!")
        except Exception as e:
            print(f"[ECG重建器] 模型加载失败: {e}")
            self.ready = False
    
    def preprocess(self, radar_data):
        """预处理雷达数据"""
        radar = np.array(radar_data, dtype=np.float32)
        
        # 确保长度为 MODEL_INPUT_LENGTH
        if len(radar) < MODEL_INPUT_LENGTH:
            radar = np.pad(radar, (0, MODEL_INPUT_LENGTH - len(radar)), mode='edge')
        elif len(radar) > MODEL_INPUT_LENGTH:
            radar = radar[-MODEL_INPUT_LENGTH:]
        
        # Z-score标准化
        mean = np.mean(radar)
        std = np.std(radar) + 1e-8
        radar = (radar - mean) / std
        
        # 转换为tensor: (1, MODEL_INPUT_LENGTH, 1)
        radar_tensor = torch.tensor(radar, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        return radar_tensor.to(self.device)
    
    @torch.no_grad()
    def reconstruct(self, radar_data):
        """从雷达数据重建ECG"""
        if not self.ready:
            return None
        
        radar_tensor = self.preprocess(radar_data)
        
        ecg_reconstructed = ddim_sample(
            self.model,
            self.scheduler,
            radar_tensor,
            ddim_steps=self.ddim_steps,
            eta=0.0
        )
        
        ecg = ecg_reconstructed.cpu().numpy().squeeze()
        return ecg


class InferenceThread(QThread):
    """推理线程 - 异步执行ECG重建"""
    inference_done = pyqtSignal(np.ndarray, float, int)  # ecg数据, 推理时间, 滑动步长
    
    def __init__(self, reconstructor):
        super().__init__()
        self.reconstructor = reconstructor
        self.data_queue = Queue()
        self.running = False
    
    def add_data(self, radar_data, slide_step):
        """添加数据到推理队列"""
        self.data_queue.put((radar_data.copy(), slide_step))
    
    def run(self):
        self.running = True
        print("[推理线程] 启动")
        
        while self.running:
            try:
                # 非阻塞获取数据
                if not self.data_queue.empty():
                    radar_data, slide_step = self.data_queue.get(timeout=0.1)
                    
                    start_time = time.time()
                    ecg = self.reconstructor.reconstruct(radar_data)
                    inference_time = time.time() - start_time
                    
                    if ecg is not None:
                        self.inference_done.emit(ecg, inference_time, slide_step)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[推理线程] 错误: {e}")
                time.sleep(0.1)
        
        print("[推理线程] 停止")
    
    def stop(self):
        self.running = False
        self.wait()


class RadarThread(QThread):
    """雷达数据读取线程"""
    packet_received = pyqtSignal(dict)

    def __init__(self, data_port, cli_port, config_obj, cfg_file_path):
        super().__init__()
        self.data_port = data_port
        self.cli_port = cli_port
        self.cfg = config_obj
        self.cfg_file_path = cfg_file_path
        self.running = False

        # 统计与调试
        self._bytes_in = 0
        self._magic_hits = 0
        self._frames_ok = 0
        self._frames_bad = 0
        self._last_stats_print = 0.0

        # bad 帧调试：避免刷屏，只在每 N 个 bad 帧打印一次 TLV 概览
        self._bad_debug_every = 50

    def _read_cli_lines(self, cli, max_wait_s=1.0):
        """尽力读取 CLI 回显，便于判断命令是否被接受。"""
        end_time = time.time() + float(max_wait_s)
        lines = []
        buf = bytearray()

        # 先快速收一波已有数据
        try:
            n0 = getattr(cli, 'in_waiting', 0)
            if n0:
                buf.extend(cli.read(n0))
        except Exception:
            pass

        while time.time() < end_time:
            try:
                n = getattr(cli, 'in_waiting', 0)
                if n:
                    buf.extend(cli.read(n))
                else:
                    # 兼容有些固件按行返回
                    raw = cli.readline()
                    if raw:
                        buf.extend(raw)
                    else:
                        time.sleep(0.01)
            except Exception:
                break

            if b'\n' in buf or b'\r' in buf:
                try:
                    text = buf.decode(errors='ignore')
                except Exception:
                    text = ''
                for s in [x.strip() for x in text.replace('\r', '\n').split('\n') if x.strip()]:
                    lines.append(s)
                    if ('Done' in s) or ('Error' in s) or ('Ignored' in s) or ('not recognized' in s):
                        return lines

        if buf and not lines:
            # 有回显但没有换行，给个原始可视化（截断）
            try:
                raw_txt = buf.decode(errors='ignore')
            except Exception:
                raw_txt = str(bytes(buf))
            raw_txt = raw_txt.strip()
            if raw_txt:
                lines.append(raw_txt[:120])
        return lines

    def _drain_cli(self, cli, max_wait_s=0.8):
        """抓取一段时间内的 CLI 输出（用于 sensorStart 后看是否有隐藏 Error）。"""
        lines = self._read_cli_lines(cli, max_wait_s=max_wait_s)
        for s in lines:
            print(f"[CLI-OUT] {s}")
        return lines

    def _send_cli_cmd(self, cli, cmd, wait_s=0.6):
        """发送一条 CLI 命令并打印回显。"""
        # mmWave CLI 在 Windows 上常见需要 CRLF
        cmd_line = cmd.strip() + '\r\n'
        cmd_to_send = cmd_line.encode('ascii', errors='ignore')
        cli.write(cmd_to_send)
        try:
            cli.flush()
        except Exception:
            pass
        time.sleep(0.02)
        lines = self._read_cli_lines(cli, max_wait_s=wait_s)
        if not lines:
            print(f"[CLI] {cmd} -> (no response)")
            return []

        # 打印最后一行作为摘要；若包含 Error/recognized 等则把所有行都打印出来
        summary = lines[-1]
        print(f"[CLI] {cmd} -> {summary}")
        lower = summary.lower()
        if ("error" in lower) or ("not recognized" in lower) or ("ignored" in lower) or ("fail" in lower):
            for s in lines:
                print(f"[CLI-OUT] {s}")
        return lines

    def _sniff_magic(self, port_name: str, baud: int, timeout_s: float = 2.0) -> int:
        """在给定串口/波特率上嗅探 magic word 命中次数。返回命中次数。"""
        hit = 0
        ser = None
        try:
            ser = serial.Serial(port_name, baud, timeout=0.05)
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            end_t = time.time() + float(timeout_s)
            buf = bytearray()
            while time.time() < end_t:
                chunk = ser.read(4096)
                if chunk:
                    buf.extend(chunk)
                    # 统计 magic 命中
                    # 为避免 O(n^2)，只对新增数据附近做 find：这里简单做全量 find，数据量小(几秒)可接受
                    i = buf.find(MAGIC_WORD)
                    while i != -1:
                        hit += 1
                        # 继续往后找
                        i = buf.find(MAGIC_WORD, i + 1)
                    # 控制 buffer 大小
                    if len(buf) > 16384:
                        buf = buf[-8192:]
                else:
                    time.sleep(0.002)
        except Exception as e:
            print(f"[SNIFF] {port_name}@{baud}: open/read failed: {e}")
        finally:
            try:
                if ser is not None:
                    ser.close()
            except Exception:
                pass
        return hit

    def _auto_select_data_port_and_baud(self, prefer_port: str, alt_port: str):
        """当 Data 口无数据时，自动尝试端口/波特率组合，找到能出现 magic word 的组合。"""
        candidates_ports = [prefer_port, alt_port]
        candidates_bauds = [921600, 460800, 115200]

        best = (None, None, 0)
        print("[SNIFF] Data=0，开始自动嗅探端口/波特率...", flush=True)
        for p in candidates_ports:
            for b in candidates_bauds:
                print(f"[SNIFF] probing {p} @ {b} ...", flush=True)
                hits = self._sniff_magic(p, b, timeout_s=2.0)
                print(f"[SNIFF] result {p}@{b}: magic_hits={hits}", flush=True)
                if hits > best[2]:
                    best = (p, b, hits)

        if best[2] > 0:
            print(f"[SNIFF] 选择 {best[0]} @ {best[1]} (magic_hits={best[2]})", flush=True)
            return best[0], best[1]

        print("[SNIFF] 未在任意组合中发现 magic word。更可能是雷达未输出数据（未真正启动/未开启UART输出）。", flush=True)
        return None, None

    def _describe_port(self, port_name: str) -> str:
        for p in serial.tools.list_ports.comports():
            if p.device == port_name:
                desc = p.description or ''
                hwid = p.hwid or ''
                return f"{port_name} | {desc} | {hwid}"
        return f"{port_name} | (not found in list_ports)"

    def _probe_cli(self, cli) -> bool:
        """探测串口是否真的是 CLI：发送空行/帮助手势并看是否有任何回显。"""
        try:
            cli.reset_input_buffer()
        except Exception:
            pass
        # 先发空行，很多固件会回提示符
        self._send_cli_cmd(cli, '', wait_s=0.4)
        # 再发一个常见命令（不保证所有固件支持 help，但通常会至少回 Error）
        self._send_cli_cmd(cli, 'help', wait_s=0.8)
        # 只要有任意回显就认为 ok
        lines = self._read_cli_lines(cli, max_wait_s=0.2)
        return len(lines) > 0

    def _try_extract_one_frame(self, buffer: bytearray):
        """从 buffer 中尝试抽取 1 帧 (基于 totalPacketLen)，成功则返回(frame_bytes, new_buffer)。"""
        idx = buffer.find(MAGIC_WORD)
        if idx == -1:
            # 丢弃过长垃圾，只保留可能的 magic 前缀
            if len(buffer) > 4096:
                buffer = buffer[-LENGTH_MAGIC_WORD:]
            return None, buffer

        # 等到至少有完整 header
        if len(buffer) < idx + LENGTH_HEADER:
            return None, buffer

        # header 布局：magic(8) + 8个uint32
        try:
            (version,
             total_packet_len,
             platform,
             frame_number,
             time_cpu_cycles,
             num_detected_obj,
             num_tlvs,
             subframe_number) = struct.unpack_from('<IIIIIIII', buffer, idx + 8)
        except Exception:
            # header 不可解析，丢弃一个字节继续找
            return None, buffer[idx + 1:]

        # 健壮性检查（过滤 payload 内偶然出现的 magic，避免“伪帧”导致 ok 永远为 0）
        if total_packet_len < LENGTH_HEADER or total_packet_len > 8192:
            return None, buffer[idx + 1:]
        if (total_packet_len % MMWDEMO_SEGMENT_LEN) != 0:
            # 本固件输出会按 32 字节对齐，总长度应为 32 的倍数
            return None, buffer[idx + 1:]
        if num_tlvs <= 0 or num_tlvs > 8:
            return None, buffer[idx + 1:]

        if len(buffer) < idx + total_packet_len:
            return None, buffer

        # 进一步验证 TLV 结构：本固件固定 TLV0=STATS(len=128), TLV1=RANGE_PROFILE
        # 这样几乎可以彻底排除 payload 内的伪 magic。
        try:
            tlv0_type, tlv0_len = struct.unpack_from('<II', buffer, idx + LENGTH_HEADER)
            if tlv0_type != 0 or tlv0_len != 128:
                return None, buffer[idx + 1:]
            off1 = idx + LENGTH_HEADER + LENGTH_TLV_HEADER + int(tlv0_len)
            if off1 + LENGTH_TLV_HEADER > idx + total_packet_len:
                return None, buffer[idx + 1:]
            tlv1_type, tlv1_len = struct.unpack_from('<II', buffer, off1)
            if tlv1_type != 1:
                return None, buffer[idx + 1:]
            off_end = off1 + LENGTH_TLV_HEADER + int(tlv1_len)
            if off_end > idx + total_packet_len:
                return None, buffer[idx + 1:]
        except Exception:
            return None, buffer[idx + 1:]

        self._magic_hits += 1

        frame = bytes(buffer[idx: idx + total_packet_len])
        buffer = buffer[idx + total_packet_len:]
        return frame, buffer

    def run(self):
        print(f"[PARSER] build={PARSER_BUILD} | script={os.path.abspath(__file__)}", flush=True)
        print(f"--> 尝试打开串口: CLI={self.cli_port}, Data={self.data_port}", flush=True)
        print(f"--> 端口信息: {self._describe_port(self.cli_port)}", flush=True)
        print(f"--> 端口信息: {self._describe_port(self.data_port)}", flush=True)
        cli = None
        data_ser = None
        try:
            cli = serial.Serial(self.cli_port, 115200, timeout=0.2)
            data_baud = 921600
            data_ser = serial.Serial(self.data_port, data_baud, timeout=0.05)
            print("--> 串口打开成功！", flush=True)

            # 串口清空
            try:
                cli.reset_input_buffer()
                cli.reset_output_buffer()
            except Exception:
                pass
            try:
                data_ser.reset_input_buffer()
            except Exception:
                pass

            print("--> 发送 sensorStop...", flush=True)
            # 先探测 CLI 是否真的有回显；没有的话非常可能端口选错（COM对调）
            cli_ok = self._probe_cli(cli)
            if not cli_ok:
                print("[WARN] CLI 串口无任何回显，可能 CLI/Data 端口填反了。尝试自动交换...", flush=True)
                try:
                    cli.close()
                except Exception:
                    pass
                try:
                    data_ser.close()
                except Exception:
                    pass

                # 交换重试：用 data_port 当 CLI(115200)，用 cli_port 当 Data(921600)
                cli = serial.Serial(self.data_port, 115200, timeout=0.2)
                data_ser = serial.Serial(self.cli_port, 921600, timeout=0.05)
                print(f"[WARN] 已交换端口：CLI={self.data_port}, Data={self.cli_port}", flush=True)
                print(f"--> 端口信息: {self._describe_port(self.data_port)}", flush=True)
                print(f"--> 端口信息: {self._describe_port(self.cli_port)}", flush=True)

                cli_ok = self._probe_cli(cli)
                if not cli_ok:
                    print("[ERROR] 交换后 CLI 仍无回显。请检查：1) CLI端口是否被占用 2) 波特率是否为115200 3) 板子是否已上电/已刷入固件。", flush=True)
                    return

            self._send_cli_cmd(cli, 'sensorStop', wait_s=1.0)
            time.sleep(0.3)
            try:
                data_ser.reset_input_buffer()
            except Exception:
                pass

            print(f"--> 正在发送配置文件: {self.cfg_file_path}", flush=True)
            try:
                with open(self.cfg_file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('%'):
                            continue

                        # 由代码统一控制启停，避免 cfg 里重复/冲突
                        if line.startswith('sensorStart') or line.startswith('sensorStop') or line.startswith('frameStart'):
                            continue

                        self._send_cli_cmd(cli, line, wait_s=0.3)
                        time.sleep(0.02)

                print("--> 配置发送完毕！", flush=True)
                time.sleep(0.3)
                # sensorStart 有时会先回 Done，再输出额外信息；这里等久一点并 drain
                self._send_cli_cmd(cli, 'sensorStart', wait_s=2.0)
                self._drain_cli(cli, max_wait_s=1.0)
                print("--> 雷达启动命令已发送 (sensorStart)", flush=True)

            except Exception as e:
                print(f"!!! 配置文件发送失败: {e}")
                return

            # 若 data 口没有任何字节流，尝试自动嗅探正确的数据口/波特率
            # 期望：Aux Data Port @ 921600 能很快出现 magic。
            # 如果完全 0 字节，可能是 data_baud 不匹配或 data_port 选错。
            bytes_probe = 0
            probe_end = time.time() + 2.0
            while time.time() < probe_end:
                chunk0 = data_ser.read(4096)
                if chunk0:
                    bytes_probe += len(chunk0)
                    break
                time.sleep(0.01)

            if bytes_probe == 0:
                # 关闭当前 data_ser，用 sniff 结果重新打开
                try:
                    data_ser.close()
                except Exception:
                    pass

                selected_port, selected_baud = self._auto_select_data_port_and_baud(self.data_port, self.cli_port)
                if selected_port is None:
                    print("[ERROR] Data 口仍无输出。建议：确认你确实刷的是 VitalSigns demo；把 mmWave Demo Visualizer 连上看是否有数据；或在 CCS 控制台看是否有错误打印。", flush=True)
                    return

                # 若嗅探结果把 data_port 指向 CLI 口，需要相应更新（但 CLI 正在使用中，通常不会发生）
                if selected_port == self.cli_port:
                    print("[WARN] 嗅探认为 CLI 口在出数据。这种情况很少见；建议检查硬件/固件 UART 映射。", flush=True)

                self.data_port = selected_port
                data_baud = int(selected_baud)
                data_ser = serial.Serial(self.data_port, data_baud, timeout=0.05)
                try:
                    data_ser.reset_input_buffer()
                except Exception:
                    pass
                print(f"[SNIFF] 已重新打开 Data={self.data_port} @ {data_baud}", flush=True)

            self.running = True
            buffer = bytearray()

            self._last_stats_print = time.time()
            t0 = self._last_stats_print

            while self.running:
                try:
                    chunk = data_ser.read(4096)
                    if chunk:
                        buffer.extend(chunk)
                        self._bytes_in += len(chunk)
                    else:
                        time.sleep(0.002)

                    # 每秒打印一次统计，快速区分“没出数” vs “解析失败”
                    now = time.time()
                    if now - self._last_stats_print >= 1.0:
                        dt = now - t0
                        kbps = (self._bytes_in / 1024.0) / dt if dt > 0 else 0.0
                        print(
                            f"[DATA] {kbps:.1f} KB/s | buf={len(buffer)} | magic={self._magic_hits} | ok={self._frames_ok} | bad={self._frames_bad}",
                            flush=True,
                        )
                        self._last_stats_print = now
                        self._bytes_in = 0
                        t0 = now

                    # 尽可能多解析
                    while True:
                        frame_data, buffer = self._try_extract_one_frame(buffer)
                        if frame_data is None:
                            break

                        parsed = self.parse_frame(frame_data)
                        if parsed is not None:
                            self._frames_ok += 1
                            self.packet_received.emit(parsed)
                        else:
                            self._frames_bad += 1

                except Exception as e:
                    print(f"Loop Error: {e}")
                    time.sleep(0.1)

        except Exception as e:
            print(f"Serial Error: {e}")
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

    def parse_frame(self, data):
        """解析数据帧（基于 header.totalPacketLen + TLV 遍历）。

        本工程固件的 TLV 定义（见 common/mmw_output.h）：
        - TLV header: type(uint32) + length(uint32)
        - length 表示 payload 字节数（不包含 TLV header 自身）
        - stats payload: sizeof(VitalSignsDemo_OutputStats) = 128 bytes
        - range profile payload: numRangeBinProcessed * sizeof(uint32)
        """
        if data is None or len(data) < LENGTH_HEADER:
            return None

        # 解析 header
        try:
            (version,
             total_packet_len,
             platform,
             frame_number,
             time_cpu_cycles,
             num_detected_obj,
             num_tlvs,
             subframe_number) = struct.unpack_from('<IIIIIIII', data, 8)
        except Exception:
            return None

        if total_packet_len != len(data):
            # 有些实现里 totalPacketLen 可能大于当前截取长度；这里不直接失败
            total_packet_len = len(data)

        ptr = LENGTH_HEADER
        stats_obj = None
        rp_abs = None

        def _maybe_debug_bad_frame(reason: str, tlvs_seen):
            try:
                if self._bad_debug_every <= 0:
                    return
                if (self._frames_bad % int(self._bad_debug_every)) != 0:
                    return
                tlv_str = ', '.join([f"(t={t},len={l})" for (t, l) in tlvs_seen])
                print(
                    f"[PARSE-BAD] reason={reason} frame={frame_number} total={len(data)} numTLVs={num_tlvs} platform=0x{platform:X} version=0x{version:X} tlvs=[{tlv_str}]",
                    flush=True,
                )
            except Exception:
                pass

        tlvs_seen = []

        def _parse_vitalsigns_output_stats(payload: bytes):
            # C 结构体布局（little-endian）：
            # uint16, uint16, float, uint32, uint16, uint16, (28 * float)  => 128 bytes
            if payload is None or len(payload) < 128:
                return None
            try:
                rangeBinIndexMax, rangeBinIndexPhase, maxVal, processingCyclesOut, rangeBinStartIndex, rangeBinEndIndex = \
                    struct.unpack_from('<HHfIHH', payload, 0)
                floats = struct.unpack_from('<' + 'f' * 28, payload, 16)
            except Exception:
                return None

            (unwrapPhasePeak_mm,
             outputFilterBreathOut,
             outputFilterHeartOut,
             heartRateEst_FFT,
             heartRateEst_FFT_4Hz,
             heartRateEst_xCorr,
             heartRateEst_peakCount_filtered,
             breathingRateEst_FFT,
             breathingRateEst_xCorr,
             breathingRateEst_peakCount,
             confidenceMetricBreathOut,
             confidenceMetricBreathOut_xCorr,
             confidenceMetricHeartOut,
             confidenceMetricHeartOut_4Hz,
             confidenceMetricHeartOut_xCorr,
             sumEnergyBreathWfm,
             sumEnergyHeartWfm,
             motionDetectedFlag,
             breathingRateEst_harmonicEnergy,
             heartRateEst_harmonicEnergy,
             reserved7,
             reserved8,
             reserved9,
             reserved10,
             reserved11,
             reserved12,
             reserved13,
             reserved14) = floats

            return {
                'rangeBinIndexMax': int(rangeBinIndexMax),
                'rangeBinIndexPhase': int(rangeBinIndexPhase),
                'maxVal': float(maxVal),
                'processingCyclesOut': int(processingCyclesOut),
                'rangeBinStartIndex': int(rangeBinStartIndex),
                'rangeBinEndIndex': int(rangeBinEndIndex),
                'unwrapPhasePeak_mm': float(unwrapPhasePeak_mm),
                'outputFilterBreathOut': float(outputFilterBreathOut),
                'outputFilterHeartOut': float(outputFilterHeartOut),
                'heartRateEst_FFT': float(heartRateEst_FFT),
                'heartRateEst_FFT_4Hz': float(heartRateEst_FFT_4Hz),
                'heartRateEst_xCorr': float(heartRateEst_xCorr),
                'heartRateEst_peakCount_filtered': float(heartRateEst_peakCount_filtered),
                'breathingRateEst_FFT': float(breathingRateEst_FFT),
                'breathingRateEst_xCorr': float(breathingRateEst_xCorr),
                'breathingRateEst_peakCount': float(breathingRateEst_peakCount),
                'confidenceMetricBreathOut': float(confidenceMetricBreathOut),
                'confidenceMetricBreathOut_xCorr': float(confidenceMetricBreathOut_xCorr),
                'confidenceMetricHeartOut': float(confidenceMetricHeartOut),
                'confidenceMetricHeartOut_4Hz': float(confidenceMetricHeartOut_4Hz),
                'confidenceMetricHeartOut_xCorr': float(confidenceMetricHeartOut_xCorr),
                'sumEnergyBreathWfm': float(sumEnergyBreathWfm),
                'sumEnergyHeartWfm': float(sumEnergyHeartWfm),
                'motionDetectedFlag': float(motionDetectedFlag),
                'breathingRateEst_harmonicEnergy': float(breathingRateEst_harmonicEnergy),
                'heartRateEst_harmonicEnergy': float(heartRateEst_harmonicEnergy),
                'reserved': [float(reserved7), float(reserved8), float(reserved9), float(reserved10),
                             float(reserved11), float(reserved12), float(reserved13), float(reserved14)],
            }

        for _ in range(int(num_tlvs)):
            if ptr + LENGTH_TLV_HEADER > len(data):
                _maybe_debug_bad_frame('tlv_header_truncated', tlvs_seen)
                break
            try:
                tlv_type, tlv_len = struct.unpack_from('<II', data, ptr)
            except Exception:
                _maybe_debug_bad_frame('tlv_header_unpack_fail', tlvs_seen)
                break
            ptr += LENGTH_TLV_HEADER
            tlvs_seen.append((int(tlv_type), int(tlv_len)))

            # 本工程固件中 tlv_len 是 payload 字节数（不包含 TLV header）。
            # 为防止遇到不同固件/错误对齐，这里做一次自适应：若长度不合理，尝试 tlv_len-8。
            payload_len = int(tlv_len)
            if payload_len < 0:
                _maybe_debug_bad_frame('payload_len_negative', tlvs_seen)
                break
            if ptr + payload_len > len(data):
                # 尝试另一种常见定义：tlv_len = header(8)+payload
                if payload_len >= LENGTH_TLV_HEADER and (ptr + payload_len - LENGTH_TLV_HEADER) <= len(data):
                    payload_len = payload_len - LENGTH_TLV_HEADER
                else:
                    _maybe_debug_bad_frame('payload_len_oob', tlvs_seen)
                    break

            payload = data[ptr: ptr + payload_len]
            ptr += payload_len

            # stats payload: 128 bytes (VitalSignsDemo_OutputStats)
            if payload_len == 128 and stats_obj is None:
                stats_obj = _parse_vitalsigns_output_stats(payload)
                continue

            # range profile: payload is uint32 array (complex packed as 32-bit), we compute abs
            expected_rp_len = 4 * int(self.cfg.numRangeBinProcessed) if self.cfg is not None else 0
            if expected_rp_len > 0 and payload_len == expected_rp_len and rp_abs is None:
                try:
                    # 固件发的是 obj->pRangeProfileCplx（int16 real/imag交织）按 uint32 发送。
                    rp_raw_u32 = np.frombuffer(payload, dtype=np.uint32)
                    rp_raw_i16 = rp_raw_u32.view(np.int16)
                    rp_real = rp_raw_i16[0::2].astype(np.float64)
                    rp_imag = rp_raw_i16[1::2].astype(np.float64)
                    rp_abs = np.sqrt(rp_real ** 2 + rp_imag ** 2)
                except Exception:
                    rp_abs = None
                continue

        if stats_obj is None:
            _maybe_debug_bad_frame('stats_not_found', tlvs_seen)
            return None

        if rp_abs is None:
            rp_abs = np.array([], dtype=np.float64)

        return {'stats': stats_obj, 'range_profile': rp_abs}

    def stop(self):
        self.running = False
        self.wait()


class RealtimeECGGUI(QMainWindow):
    """实时ECG重建GUI"""
    
    def __init__(self, data_port, cli_port, cfg_path, model_path, ddim_steps=20):
        super().__init__()
        self.setWindowTitle("实时雷达ECG重建系统")
        self.resize(1400, 900)

        self.cfg = RadarConfig(cfg_path)
        if not self.cfg.valid:
            print("Invalid Config")
            return

        # 初始化ECG重建器
        self.reconstructor = ECGReconstructor(model_path, ddim_steps=ddim_steps)
        
        # 数据缓冲区
        # 仍以固件输出的 breath waveform 作为模型输入（但不再显示呼吸/心跳曲线）
        self.breath_buffer = deque(maxlen=MODEL_INPUT_LENGTH * 2)  # 模型输入缓冲
        self.new_sample_count = 0  # 新采样点计数
        self.slide_step = SLIDE_STEP  # 滑动步长
        
        # === ECG 实时显示缓冲区（推理输出，存在延迟）===
        self.realtime_ecg = np.zeros(PLOT_DISPLAY_LENGTH)     # ECG（推理后立即显示）
        self.realtime_ecg_queue = deque()  # ECG实时输出队列

        # 用于心率统计的ECG历史（更长窗口 -> BPM更稳）
        fps0 = float(self.cfg.frame_rate_hz) if (hasattr(self, 'cfg') and self.cfg.frame_rate_hz) else 20.0
        self.ecg_history = deque(maxlen=int(max(PLOT_DISPLAY_LENGTH, fps0 * 10.0)))

        # 心率估计（基于重建 ECG 的尖峰检测）
        self.bpm_est = None
        self._last_bpm_update_t = 0.0
        
        # 推理状态
        self.inference_count = 0
        self.last_inference_time = 0.0
        self.is_inferencing = False
        
        # 滤波器状态
        self.prev_breath_val = 0.0
        self.dataPlotThresh = 50.0

        self.init_ui()
        
        # 启动推理线程
        self.inference_thread = InferenceThread(self.reconstructor)
        self.inference_thread.inference_done.connect(self.on_inference_done)
        self.inference_thread.start()
        
        # 启动雷达线程
        self.radar_thread = RadarThread(data_port, cli_port, self.cfg, cfg_path)
        self.radar_thread.packet_received.connect(self.update_data)
        self.radar_thread.start()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 控制面板
        control_box = QGroupBox("控制面板")
        control_layout = QHBoxLayout()
        control_box.setLayout(control_layout)
        
        # 滑动步长设置
        control_layout.addWidget(QLabel("滑动步长:"))
        self.spin_slide = QSpinBox()
        self.spin_slide.setRange(16, 256)
        self.spin_slide.setValue(SLIDE_STEP)
        self.spin_slide.valueChanged.connect(self.on_slide_step_changed)
        control_layout.addWidget(self.spin_slide)
        
        control_layout.addSpacing(20)
        
        # 清除队列按钮
        self.btn_clear = QPushButton("清除队列")
        self.btn_clear.clicked.connect(self.clear_ecg_queue)
        control_layout.addWidget(self.btn_clear)
        
        control_layout.addSpacing(20)
        
        # 状态显示
        self.lbl_status = QLabel("状态: 等待数据...")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")
        control_layout.addWidget(self.lbl_status)
        
        self.lbl_inference = QLabel("推理: 0次 | 耗时: 0ms")
        self.lbl_inference.setStyleSheet("font-size: 14px; color: green;")
        control_layout.addWidget(self.lbl_inference)
        
        self.lbl_buffer = QLabel("采集: 0/128 | ECG队列: 0")
        self.lbl_buffer.setStyleSheet("font-size: 14px;")
        control_layout.addWidget(self.lbl_buffer)

        # 心率显示（由重建 ECG 的尖峰检测得到）
        self.lbl_bpm = QLabel("心率: -- bpm")
        self.lbl_bpm.setStyleSheet("font-size: 14px; color: #E91E63;")
        control_layout.addWidget(self.lbl_bpm)
        
        # 延迟显示（ECG队列长度 / 雷达帧率）
        self.lbl_delay = QLabel("ECG延迟: ~0s")
        self.lbl_delay.setStyleSheet("font-size: 14px; color: #666;")
        control_layout.addWidget(self.lbl_delay)
        
        control_layout.addStretch()
        layout.addWidget(control_box)

        # 图表区域
        pg.setConfigOptions(antialias=True)
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground('w')
        layout.addWidget(self.win)

        # 样式设置
        title_style = {'color': '#333', 'size': '11pt', 'bold': True}
        label_style = {'color': '#666', 'font-size': '9pt'}

        # ========== 仅保留：重建 ECG（推理后输出）==========
        p3 = self.win.addPlot(title="", row=0, col=0)
        p3.showGrid(x=True, y=True, alpha=0.3)
        p3.setTitle("ECG重建 (推理输出)", **title_style)
        self.curve_realtime_ecg = p3.plot(pen=pg.mkPen(color='#F44336', width=2.5))
        p3.setXRange(0, PLOT_DISPLAY_LENGTH, padding=0)
        # ECG幅值范围固定到 ±3
        p3.setYRange(-3, 3)
        p3.setLabel('left', '幅值', **label_style)
        p3.setLabel('bottom', '采样点', **label_style)
        p3.getViewBox().disableAutoRange(axis=pg.ViewBox.XAxis)
        p3.getViewBox().disableAutoRange(axis=pg.ViewBox.YAxis)
        p3.setMouseEnabled(x=False, y=False)

    def clear_ecg_queue(self):
        """清除所有输出队列"""
        self.realtime_ecg_queue.clear()
        self.realtime_ecg = np.zeros(PLOT_DISPLAY_LENGTH)
        self.bpm_est = None
        try:
            self.lbl_bpm.setText("心率: -- bpm")
        except Exception:
            pass
        print("所有输出队列已清除")

    def _estimate_bpm_from_ecg(self, ecg: np.ndarray, fs: float, thr: float = 1.5):
        """按用户要求：统计标准化ECG中 > thr 的尖峰频率作为心跳(BPM)。

        规则：
        - 先对 ecg 做去均值/标准化
        - 只找局部极大且 x[i] > thr
        - 300ms 不应期去重（避免一个QRS被算多次）
        - BPM = 峰个数 / 窗口秒数 * 60
        """
        if ecg is None:
            return None
        if fs is None or fs <= 0:
            return None
        if len(ecg) < max(20, int(fs * 2.0)):
            return None

        x = np.asarray(ecg, dtype=np.float64)
        x = x - np.mean(x)
        s = float(np.std(x) + 1e-8)
        x = x / s

        # 局部极大 + 固定阈值
        peaks = []
        min_dist = max(1, int(0.30 * fs))
        last_p = -10**9
        for i in range(1, len(x) - 1):
            if x[i] <= thr:
                continue
            if not (x[i - 1] < x[i] and x[i] > x[i + 1]):
                continue
            if (i - last_p) < min_dist:
                # 若离上一个峰太近，只保留更高的那个
                if peaks and x[i] > x[peaks[-1]]:
                    peaks[-1] = i
                    last_p = i
                continue
            peaks.append(i)
            last_p = i

        if len(peaks) < 2:
            return None

        duration_s = len(x) / float(fs)
        if duration_s <= 0:
            return None
        bpm = (len(peaks) / duration_s) * 60.0
        if not (30.0 <= bpm <= 220.0):
            return None
        return float(bpm)

    def on_slide_step_changed(self, value):
        self.slide_step = value
        print(f"滑动步长更新为: {value}")

    def update_data(self, data):
        """接收雷达数据并更新"""
        stats = data['stats']
        # 结构体字段名（见固件 VitalSignsDemo_OutputStats）
        val_breath = float(stats.get('outputFilterBreathOut', 0.0))
        
        # Glitch滤波
        if abs(val_breath - self.prev_breath_val) > self.dataPlotThresh:
            val_breath = self.prev_breath_val
        else:
            self.prev_breath_val = val_breath
        
        # 添加到缓冲区（用于后续推理）
        self.breath_buffer.append(val_breath)
        self.new_sample_count += 1
        
        # ECG从队列取（推理完成后立即输出）
        if len(self.realtime_ecg_queue) > 0:
            ecg_val = self.realtime_ecg_queue.popleft()
            self.realtime_ecg = np.roll(self.realtime_ecg, -1)
            self.realtime_ecg[-1] = ecg_val
            self.ecg_history.append(float(ecg_val))

        # ========== 绘制 ECG（只做时间平移，不做任何随窗口变化的归一化/去均值）==========
        # 用户诉求：曲线应仅随时间向左平移，幅值不应因显示窗口变化而“晃动”。
        # 因此这里直接绘制模型输出的原始幅值（Y 轴范围已固定为 ±3）。
        self.curve_realtime_ecg.setData(self.realtime_ecg)
        
        # 更新状态显示
        buffer_len = len(self.breath_buffer)
        rt_queue_len = len(self.realtime_ecg_queue)
        self.lbl_buffer.setText(f"采集: {buffer_len}/{MODEL_INPUT_LENGTH} | ECG队列: {rt_queue_len}")
        
        # 计算延迟（对齐模式的延迟）
        fps = float(self.cfg.frame_rate_hz) if (hasattr(self, 'cfg') and self.cfg.frame_rate_hz) else 20.0
        delay_s = rt_queue_len / fps
        self.lbl_delay.setText(f"ECG延迟: ~{delay_s:.1f}s")

        # 低频更新 BPM（避免每帧都算）
        now = time.time()
        if now - self._last_bpm_update_t >= 0.5:
            self._last_bpm_update_t = now
            hist = np.array(self.ecg_history, dtype=np.float64)
            bpm = self._estimate_bpm_from_ecg(hist, fps, thr=1.5)
            if bpm is not None:
                self.bpm_est = bpm
                self.lbl_bpm.setText(f"心率: {bpm:.0f} bpm")
            else:
                self.lbl_bpm.setText("心率: -- bpm")
        
        # 检查是否触发推理
        if buffer_len >= MODEL_INPUT_LENGTH and self.new_sample_count >= self.slide_step:
            if not self.is_inferencing and self.reconstructor.ready:
                self.trigger_inference()
    
    def trigger_inference(self):
        """触发推理"""
        self.is_inferencing = True
        self.new_sample_count = 0
        
        # 取最近128点数据
        radar_data = np.array(list(self.breath_buffer))[-MODEL_INPUT_LENGTH:]

        # 保存对应的原始数据（保留接口兼容；当前仅用于推理输入）
        self.pending_breath_data = radar_data.copy()
        
        # 发送到推理线程，同时传递当前滑动步长
        self.inference_thread.add_data(radar_data, self.slide_step)
        self.lbl_status.setText("状态: 推理中...")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: orange;")
    
    def on_inference_done(self, ecg_data, inference_time, slide_step):
        """推理完成回调 - 同时更新实时和对齐显示"""
        self.is_inferencing = False
        self.inference_count += 1
        self.last_inference_time = inference_time
        
        # ========== 实时ECG队列：推理完成后立即加入 ==========
        # 只加入新增的slide_step个点
        if self.inference_count == 1:
            # 第一次推理，加入全部
            for val in ecg_data:
                self.realtime_ecg_queue.append(val)
        else:
            # 后续只加入新增部分
            for val in ecg_data[-slide_step:]:
                self.realtime_ecg_queue.append(val)
        
        # 更新状态
        self.lbl_status.setText("状态: 运行中")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: green;")
        self.lbl_inference.setText(f"推理: {self.inference_count}次 | 耗时: {inference_time*1000:.0f}ms")

    def closeEvent(self, event):
        if hasattr(self, 'radar_thread'):
            self.radar_thread.stop()
        if hasattr(self, 'inference_thread'):
            self.inference_thread.stop()
        event.accept()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='实时雷达ECG重建')
    parser.add_argument('--data-port', default='COM14', help='数据串口')
    parser.add_argument('--cli-port', default='COM13', help='配置串口')
    parser.add_argument('--cfg', default=r'D:\Code\radar-ecg-reconstruction\diffu_1228\radar-ecg-diffusion\src\profile_2d_VitalSigns.cfg',
                        help='雷达配置文件')
    parser.add_argument('--model', default=r'D:\Code\radar-ecg-reconstruction\diffu_1228\radar-ecg-diffusion\src\checkpoints\diffusion_model_final_128_0122.pth', help='模型路径')
    parser.add_argument('--ddim-steps', type=int, default=20, help='DDIM采样步数(越少越快)')
    
    args = parser.parse_args()

    print(f"[PARSER] build={PARSER_BUILD} | script={os.path.abspath(__file__)}")
    
    # 处理模型路径
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    
    print("=" * 60)
    print("实时雷达ECG重建系统")
    print("=" * 60)
    print(f"数据串口: {args.data_port}")
    print(f"配置串口: {args.cli_port}")
    print(f"配置文件: {args.cfg}")
    print(f"模型路径: {model_path}")
    print(f"DDIM步数: {args.ddim_steps}")
    print("=" * 60)
    
    app = QApplication(sys.argv)
    window = RealtimeECGGUI(
        data_port=args.data_port,
        cli_port=args.cli_port,
        cfg_path=args.cfg,
        model_path=model_path,
        ddim_steps=args.ddim_steps
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
