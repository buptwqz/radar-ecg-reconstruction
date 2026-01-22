import matplotlib.pyplot as plt
import numpy as np

def plot_reconstruction(true_ecg, reconstructed_ecg, correlation):
    plt.figure(figsize=(12, 5))
    plt.plot(true_ecg, label='True ECG', color='blue', alpha=0.6)
    plt.plot(reconstructed_ecg, label='Reconstructed ECG', color='red', linestyle='--')
    plt.title(f"ECG Reconstruction (Correlation: {correlation:.4f})")
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

def save_plot(true_ecg, reconstructed_ecg, correlation, save_path):
    plt.figure(figsize=(12, 5))
    plt.plot(true_ecg, label='True ECG', color='blue', alpha=0.6)
    plt.plot(reconstructed_ecg, label='Reconstructed ECG', color='red', linestyle='--')
    plt.title(f"ECG Reconstruction (Correlation: {correlation:.4f})")
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()