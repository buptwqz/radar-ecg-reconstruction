import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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
from inference.sampler import ddim_sample  # noqa: E402
from evaluation.metrics import calculate_correlation  # noqa: E402

def evaluate_model(data_path, model_path, ddim_steps=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    radar_raw, ecg_raw = load_data_source(data_path)
    radar_data, ecg_data = standardize(radar_raw), standardize(ecg_raw)
    dataset = RadarECGDataset(radar_data, ecg_data)
    test_loader = DataLoader(dataset, batch_size=1)

    # Initialize the model and load weights
    model = ConditionalUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize the diffusion scheduler
    scheduler = DiffusionScheduler(steps=1000, device=device)

    all_corrs = []
    for radar, ecg in test_loader:
        radar = radar.to(device)
        reconstruction = ddim_sample(model, scheduler, radar, ddim_steps=ddim_steps)

        # Calculate correlation
        corr = calculate_correlation(reconstruction.cpu().numpy(), ecg.cpu().numpy())
        all_corrs.append(corr)

        # Plot the results
        plt.figure(figsize=(12, 5))
        plt.plot(ecg.cpu().numpy().flatten(), label='True ECG', color='blue', alpha=0.6)
        plt.plot(reconstruction.cpu().numpy().flatten(), label='Reconstructed ECG', color='red', linestyle='--')
        plt.title(f"Radar to ECG Reconstruction (Corr: {corr:.4f})")
        plt.legend()
        plt.show()

    avg_corr = np.mean(all_corrs)
    print(f"Average correlation across test samples: {avg_corr:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to the dataset file (.npz)')
    parser.add_argument('--model', required=True, help='Path to the trained model weights')
    parser.add_argument('--ddim-steps', type=int, default=20, help='Number of DDIM sampling steps')
    args = parser.parse_args()

    evaluate_model(args.data, args.model, args.ddim_steps)