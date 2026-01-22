import os
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


class RadarECGDataset(Dataset):
    def __init__(self, radar_data, ecg_data):
        self.radar_data = radar_data.astype(np.float32)
        self.ecg_data = ecg_data.astype(np.float32)

    def __len__(self):
        return len(self.radar_data)

    def __getitem__(self, idx):
        radar = np.expand_dims(self.radar_data[idx], axis=-1)
        ecg = np.expand_dims(self.ecg_data[idx], axis=-1)
        return torch.from_numpy(radar), torch.from_numpy(ecg)

def load_data_source(data_path, expected_len: int | None = None):
    if data_path.endswith('.npz'):
        with np.load(data_path) as f:
            return f['radar'], f['ecg']
    else:
        radar_list, ecg_list = [], []
        for f in os.listdir(data_path):
            if f.endswith('.mat'):
                mat = scipy.io.loadmat(os.path.join(data_path, f))
                r, e = mat['radar'].reshape(1, -1), mat['ecg'].reshape(1, -1)
                if expected_len is None or r.shape[1] == expected_len:
                    radar_list.append(r)
                    ecg_list.append(e)
        return np.concatenate(radar_list, axis=0), np.concatenate(ecg_list, axis=0)