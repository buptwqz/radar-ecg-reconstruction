"""
准备 512 点数据集
将 radar-data-0113 和 ecg-data-0113 文件夹中的数据整合为 npz 格式
"""
import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_folder_data(folder_path: Path, pattern: str = "*", expected_length: int = 512):
    """从文件夹加载所有数据文件"""
    files = sorted(folder_path.glob(pattern))
    data_list = []
    
    for f in tqdm(files, desc=f"加载 {folder_path.name}"):
        try:
            data = np.loadtxt(f)
            if data.ndim != 1:
                data = data.reshape(-1)
            
            if len(data) != expected_length:
                print(f"警告: {f.name} 长度为 {len(data)}，期望 {expected_length}，跳过")
                continue
                
            data_list.append(data)
        except Exception as e:
            print(f"读取 {f} 失败: {e}")
            continue
    
    return np.array(data_list, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="准备 512 点训练数据集")
    parser.add_argument("--radar-dir", default="../../radar-data-0113", 
                        help="雷达数据文件夹路径")
    parser.add_argument("--ecg-dir", default="../../ecg-data-0113", 
                        help="心电数据文件夹路径")
    parser.add_argument("--output", "-o", default="dataset_512_0113.npz", 
                        help="输出文件名")
    parser.add_argument("--radar-pattern", default="*.csv", 
                        help="雷达文件匹配模式")
    parser.add_argument("--ecg-pattern", default="*.txt", 
                        help="心电文件匹配模式")
    parser.add_argument("--length", type=int, default=512, 
                        help="每个样本的点数")
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    
    radar_dir = Path(args.radar_dir)
    ecg_dir = Path(args.ecg_dir)
    
    # 如果是相对路径，则相对于脚本目录
    if not radar_dir.is_absolute():
        radar_dir = script_dir / radar_dir
    if not ecg_dir.is_absolute():
        ecg_dir = script_dir / ecg_dir
    
    print(f"雷达数据目录: {radar_dir.absolute()}")
    print(f"心电数据目录: {ecg_dir.absolute()}")
    print(f"样本长度: {args.length}")
    
    if not radar_dir.exists():
        raise FileNotFoundError(f"雷达数据目录不存在: {radar_dir}")
    if not ecg_dir.exists():
        raise FileNotFoundError(f"心电数据目录不存在: {ecg_dir}")
    
    # 加载数据
    print("\n正在加载雷达数据...")
    radar_data = load_folder_data(radar_dir, args.radar_pattern, args.length)
    
    print("\n正在加载心电数据...")
    ecg_data = load_folder_data(ecg_dir, args.ecg_pattern, args.length)
    
    print(f"\n雷达数据形状: {radar_data.shape}")
    print(f"心电数据形状: {ecg_data.shape}")
    
    # 检查数量是否匹配
    if len(radar_data) != len(ecg_data):
        print(f"\n警告: 雷达数据 ({len(radar_data)}) 和心电数据 ({len(ecg_data)}) 数量不匹配!")
        min_len = min(len(radar_data), len(ecg_data))
        print(f"将只使用前 {min_len} 个样本")
        radar_data = radar_data[:min_len]
        ecg_data = ecg_data[:min_len]
    
    # 保存数据集
    output_path = script_dir / args.output
    print(f"\n正在保存到: {output_path}")
    np.savez(output_path, radar=radar_data, ecg=ecg_data)
    
    # 验证
    print("\n验证保存的数据...")
    with np.load(output_path) as f:
        print(f"  radar 形状: {f['radar'].shape}")
        print(f"  ecg 形状: {f['ecg'].shape}")
    
    print(f"\n完成! 数据集已保存到: {output_path.absolute()}")
    print(f"样本数: {len(radar_data)}")
    print(f"每个样本点数: {args.length}")


if __name__ == "__main__":
    main()
