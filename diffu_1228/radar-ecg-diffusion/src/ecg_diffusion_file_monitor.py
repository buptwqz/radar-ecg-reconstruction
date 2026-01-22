"""扩散模型 ECG 重建：文件监控/批处理工具。

用途
- 监控某个目录中新生成的 CSV（雷达序列），触发扩散模型推理并输出 *_ecg.csv。
- 或对已有目录批量推理。

适用场景
- 你已经有雷达序列文件（例如离线采集/其他程序落盘），不需要串口/UDP 读雷达。
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.unet import ConditionalUNet
from models.scheduler import DiffusionScheduler
from inference.sampler import ddim_sample


class ECGReconstructor:
    """心电信号重建器"""
    
    def __init__(self, model_path, device='cuda', ddim_steps=50, input_length=128):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.ddim_steps = ddim_steps
        self.input_length = int(input_length)
        
        print(f"使用设备: {self.device}")
        print(f"加载模型: {model_path}")
        
        # 加载模型
        self.model = ConditionalUNet(base_ch=64, time_dim=256).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # 初始化调度器
        self.scheduler = DiffusionScheduler(steps=1000, device=self.device)
        
        print("模型加载完成!")
    
    def preprocess(self, radar_data):
        """预处理雷达数据 - 标准化"""
        radar = np.array(radar_data, dtype=np.float32)
        
        # 确保数据长度与模型一致
        L = self.input_length
        if len(radar) < L:
            radar = np.pad(radar, (0, L - len(radar)), mode='edge')
        elif len(radar) > L:
            radar = radar[:L]
        
        # 标准化
        mean = np.mean(radar)
        std = np.std(radar) + 1e-8
        radar = (radar - mean) / std
        
        # 转换为tensor: (1, L, 1)
        radar_tensor = torch.tensor(radar, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        return radar_tensor.to(self.device)
    
    @torch.no_grad()
    def reconstruct(self, radar_data):
        """从雷达数据重建ECG信号"""
        # 预处理
        radar_tensor = self.preprocess(radar_data)
        
        # DDIM采样重建
        ecg_reconstructed = ddim_sample(
            self.model, 
            self.scheduler, 
            radar_tensor, 
            ddim_steps=self.ddim_steps,
            eta=0.0
        )
        
        # 转换为numpy
        ecg = ecg_reconstructed.cpu().numpy().squeeze()
        return ecg
    
    def reconstruct_from_file(self, file_path):
        """从CSV文件读取雷达数据并重建ECG"""
        # 读取雷达数据
        radar_data = np.loadtxt(file_path)
        
        # 重建ECG
        start_time = time.time()
        ecg = self.reconstruct(radar_data)
        inference_time = time.time() - start_time
        
        return radar_data, ecg, inference_time


class RealtimeFileHandler(FileSystemEventHandler):
    """实时文件监控处理器"""
    
    def __init__(self, reconstructor, output_dir, visualize=True):
        self.reconstructor = reconstructor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualize = visualize
        self.processed_files = set()
        
        if visualize:
            plt.ion()  # 开启交互模式
            self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 6))
            plt.tight_layout()
    
    def on_created(self, event):
        """当新文件创建时触发"""
        if event.is_directory:
            return
        
        if event.src_path.endswith('.csv'):
            # 等待文件写入完成
            time.sleep(0.1)
            self.process_file(event.src_path)
    
    def process_file(self, file_path):
        """处理单个文件"""
        if file_path in self.processed_files:
            return
        
        self.processed_files.add(file_path)
        
        try:
            print(f"\n处理文件: {file_path}")
            radar, ecg, inference_time = self.reconstructor.reconstruct_from_file(file_path)
            print(f"推理时间: {inference_time*1000:.2f} ms")
            
            # 保存结果
            filename = Path(file_path).stem
            output_path = self.output_dir / f"{filename}_ecg.csv"
            np.savetxt(output_path, ecg, fmt='%.6f')
            print(f"ECG已保存: {output_path}")
            
            # 可视化
            if self.visualize:
                self.update_plot(radar, ecg, filename)
                
        except Exception as e:
            print(f"处理失败: {e}")
    
    def update_plot(self, radar, ecg, title):
        """更新实时图表"""
        self.axes[0].clear()
        self.axes[0].plot(radar, 'b-', linewidth=0.8)
        self.axes[0].set_title(f'雷达信号 - {title}')
        self.axes[0].set_xlabel('Sample')
        self.axes[0].set_ylabel('Amplitude')
        self.axes[0].grid(True, alpha=0.3)
        
        self.axes[1].clear()
        self.axes[1].plot(ecg, 'r-', linewidth=0.8)
        self.axes[1].set_title('重建的ECG信号')
        self.axes[1].set_xlabel('Sample')
        self.axes[1].set_ylabel('Amplitude')
        self.axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)


def run_realtime_monitoring(reconstructor, watch_dir, output_dir, visualize=True):
    """运行实时文件监控"""
    print(f"\n开始监控目录: {watch_dir}")
    print("等待新的雷达数据文件...")
    print("按 Ctrl+C 停止\n")
    
    handler = RealtimeFileHandler(reconstructor, output_dir, visualize)
    observer = Observer()
    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n停止监控...")
        observer.stop()
    
    observer.join()
    if visualize:
        plt.ioff()
        plt.close()


def run_batch_inference(reconstructor, input_dir, output_dir, visualize=False):
    """批量处理已有文件"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(input_path.glob('*.csv'))
    print(f"找到 {len(csv_files)} 个文件待处理\n")
    
    results = []
    
    for i, file_path in enumerate(csv_files):
        print(f"[{i+1}/{len(csv_files)}] 处理: {file_path.name}")
        
        radar, ecg, inference_time = reconstructor.reconstruct_from_file(str(file_path))
        
        # 保存结果
        output_file = output_path / f"{file_path.stem}_ecg.csv"
        np.savetxt(output_file, ecg, fmt='%.6f')
        
        results.append({
            'file': file_path.name,
            'inference_time': inference_time
        })
        
        print(f"    推理时间: {inference_time*1000:.2f} ms")
        
        # 可视化前几个
        if visualize and i < 5:
            plt.figure(figsize=(12, 5))
            plt.subplot(2, 1, 1)
            plt.plot(radar, 'b-', linewidth=0.8)
            plt.title(f'雷达信号 - {file_path.name}')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(ecg, 'r-', linewidth=0.8)
            plt.title('重建的ECG信号')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / f"{file_path.stem}_plot.png", dpi=150)
            plt.close()
    
    # 统计
    avg_time = np.mean([r['inference_time'] for r in results])
    print(f"\n处理完成!")
    print(f"平均推理时间: {avg_time*1000:.2f} ms")
    print(f"结果保存到: {output_path}")


def run_single_inference(reconstructor, input_file, output_file=None, visualize=True):
    """单文件推理"""
    print(f"处理文件: {input_file}")
    
    radar, ecg, inference_time = reconstructor.reconstruct_from_file(input_file)
    
    print(f"推理时间: {inference_time*1000:.2f} ms")
    
    # 保存结果
    if output_file:
        np.savetxt(output_file, ecg, fmt='%.6f')
        print(f"ECG已保存: {output_file}")
    
    # 可视化
    if visualize:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(radar, 'b-', linewidth=0.8)
        plt.title('输入: 雷达信号')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(ecg, 'r-', linewidth=0.8)
        plt.title('输出: 重建的ECG信号')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return ecg


def main():
    parser = argparse.ArgumentParser(description='雷达到ECG实时推理')
    parser.add_argument('--model', default='checkpoints/diffusion_model_final_128_0109.pth',
                        help='模型路径')
    parser.add_argument('--mode', choices=['realtime', 'batch', 'single'], default='single',
                        help='运行模式: realtime=实时监控, batch=批量处理, single=单文件')
    parser.add_argument('--input', default='d:/mmWave/ecg-reconstruction/radar-data',
                        help='输入目录或文件路径')
    parser.add_argument('--output', default='inference_output',
                        help='输出目录')
    parser.add_argument('--ddim-steps', type=int, default=50,
                        help='DDIM采样步数（越大越慢但质量可能更好）')
    parser.add_argument('--device', default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--input-length', type=int, default=128,
                        help='模型输入长度（默认 128）')
    parser.add_argument('--no-visualize', action='store_true',
                        help='禁用可视化')
    
    args = parser.parse_args()
    
    # 初始化重建器
    reconstructor = ECGReconstructor(
        model_path=args.model,
        device=args.device,
        ddim_steps=args.ddim_steps,
        input_length=args.input_length,
    )
    
    visualize = not args.no_visualize
    
    if args.mode == 'realtime':
        # 实时监控模式
        run_realtime_monitoring(reconstructor, args.input, args.output, visualize)
        
    elif args.mode == 'batch':
        # 批量处理模式
        run_batch_inference(reconstructor, args.input, args.output, visualize)
        
    else:
        # 单文件模式
        if os.path.isdir(args.input):
            # 如果是目录，取第一个文件
            csv_files = sorted(Path(args.input).glob('*.csv'))
            if csv_files:
                input_file = str(csv_files[0])
            else:
                print("目录中没有CSV文件!")
                return
        else:
            input_file = args.input
        
        output_file = os.path.join(args.output, 'reconstructed_ecg.csv') if args.output else None
        os.makedirs(args.output, exist_ok=True)
        run_single_inference(reconstructor, input_file, output_file, visualize)


if __name__ == '__main__':
    main()
