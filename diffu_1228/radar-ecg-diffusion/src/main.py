import os
import argparse
import torch
import numpy as np
from data.dataset import RadarECGDataset
from data.preprocessing import load_data_source, standardize
from models.unet import ConditionalUNet
from models.scheduler import DiffusionScheduler
from training.trainer import Trainer
from evaluation.metrics import evaluate_model
from torch.utils.data import DataLoader, random_split


def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    radar_raw, ecg_raw = load_data_source(args.data, expected_len=args.length)
    print(f"原始数据: Radar {radar_raw.shape}, ECG {ecg_raw.shape}")
    if radar_raw.ndim != 2 or ecg_raw.ndim != 2:
        raise ValueError(f"数据维度应为 (N, L)。当前: radar {radar_raw.shape}, ecg {ecg_raw.shape}")
    if radar_raw.shape != ecg_raw.shape:
        raise ValueError(f"radar/ecg 形状不一致: radar {radar_raw.shape}, ecg {ecg_raw.shape}")
    if radar_raw.shape[1] != args.length:
        raise ValueError(f"数据长度不匹配: 期望 L={args.length}，实际 L={radar_raw.shape[1]}。请使用 128 点数据集训练/推理。")
    
    # 标准化
    radar_std, ecg_std = standardize(radar_raw), standardize(ecg_raw)
    dataset = RadarECGDataset(radar_std, ecg_std)
    
    # 数据集划分（留更多用于训练）
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size], 
                                      generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              drop_last=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1)
    print(f"训练集: {len(train_ds)} 样本, 测试集: {len(test_ds)} 样本")

    # 初始化模型和调度器
    model = ConditionalUNet(base_ch=64, time_dim=256).to(device)
    scheduler = DiffusionScheduler(steps=1000, device=device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建训练器并训练
    trainer = Trainer(model, scheduler, device, lr=args.lr)
    trainer.train(train_loader, args.epochs)

    # 保存最终模型（避免覆盖旧模型）
    os.makedirs('checkpoints', exist_ok=True)
    tag = args.tag or '128_0109'
    final_path = f'checkpoints/diffusion_model_final_{tag}.pth'
    torch.save(model.state_dict(), final_path)
    print(f"最终模型已保存到 {final_path}")

    # 评估模型
    results = evaluate_model(model, test_loader, scheduler, device)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='雷达到心电信号重建 - 扩散模型')
    parser.add_argument('--data', default='dataset_128_0109.npz',
                        help='数据集路径')
    parser.add_argument('--length', type=int, default=128,
                        help='输入序列长度（训练与推理需一致；默认128）')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮次（默认200）')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率（默认1e-4）')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批大小（小数据集建议用较小的batch）')
    parser.add_argument('--tag', default=None,
                        help='输出模型文件名标签（默认 128_0109）')
    
    args = parser.parse_args()
    print("\n" + "="*50)
    print("雷达到心电信号重建 - 条件扩散模型")
    print("="*50)
    print(f"数据集: {args.data}")
    print(f"序列长度: {args.length}")
    print(f"训练轮次: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"批大小: {args.batch_size}")
    print("="*50 + "\n")
    
    main(args)