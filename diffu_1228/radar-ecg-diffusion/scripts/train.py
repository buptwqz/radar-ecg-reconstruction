import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, random_split

# 让 scripts/ 可直接运行：把 ../src 放到 sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.dataset import RadarECGDataset  # noqa: E402
from data.preprocessing import load_data_source, standardize  # noqa: E402
from models.unet import ConditionalUNet  # noqa: E402
from models.scheduler import DiffusionScheduler  # noqa: E402
from training.trainer import Trainer  # noqa: E402

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('checkpoints', exist_ok=True)

    # Load data
    radar_raw, ecg_raw = load_data_source(args.data, expected_len=args.length)
    if radar_raw.ndim != 2 or ecg_raw.ndim != 2:
        raise ValueError(f"数据维度应为 (N, L)。当前: radar {radar_raw.shape}, ecg {ecg_raw.shape}")
    if radar_raw.shape != ecg_raw.shape:
        raise ValueError(f"radar/ecg 形状不一致: radar {radar_raw.shape}, ecg {ecg_raw.shape}")
    if radar_raw.shape[1] != args.length:
        raise ValueError(f"数据长度不匹配: 期望 L={args.length}，实际 L={radar_raw.shape[1]}。请使用 128 点数据集训练/推理。")
    radar_std, ecg_std = standardize(radar_raw), standardize(ecg_raw)
    dataset = RadarECGDataset(radar_std, ecg_std)
    train_size = int(0.9 * len(dataset))
    train_ds, _ = random_split(dataset, [train_size, len(dataset) - train_size],
                               generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Initialize model and scheduler
    model = ConditionalUNet(base_ch=64, time_dim=256).to(device)
    scheduler = DiffusionScheduler(steps=1000, device=device)

    # Train
    trainer = Trainer(model, scheduler, device, lr=args.lr)
    trainer.train(train_loader, args.epochs)

    tag = args.tag or f'{args.length}_train'
    out_path = f'checkpoints/diffusion_model_final_{tag}.pth'
    torch.save(model.state_dict(), out_path)
    print(f"最终模型已保存到 {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset_128_0109.npz')
    parser.add_argument('--length', type=int, default=128,
                        help='输入序列长度（训练与推理需一致；默认128）')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--tag', default=None, help='输出模型文件名标签（默认: <length>_train）')
    main(parser.parse_args())