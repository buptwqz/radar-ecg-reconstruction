import os
import numpy as np
import scipy.io

def load_data_source(data_path, expected_len: int | None = None):
    # 如果是相对路径且文件不存在，尝试在项目根目录查找
    if not os.path.isabs(data_path) and not os.path.exists(data_path):
        # 获取项目根目录 (radar-ecg-diffusion)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        alt_path = os.path.join(project_root, data_path)
        if os.path.exists(alt_path):
            data_path = alt_path
    
    if data_path.endswith('.npz'):
        with np.load(data_path) as f:
            radar = f['radar']
            ecg = f['ecg']
        if expected_len is not None:
            if radar.ndim != 2 or ecg.ndim != 2:
                raise ValueError(
                    f"npz 内 radar/ecg 维度应为 (N, L)。当前: radar {radar.shape}, ecg {ecg.shape}"
                )
            if radar.shape[1] != expected_len or ecg.shape[1] != expected_len:
                raise ValueError(
                    f"npz 数据长度不匹配: 期望 L={expected_len}，实际 radar L={radar.shape[1]} ecg L={ecg.shape[1]}"
                )
        return radar, ecg
    else:
        radar_list, ecg_list = [], []
        for f in os.listdir(data_path):
            if f.endswith('.mat'):
                mat = scipy.io.loadmat(os.path.join(data_path, f))
                r, e = mat['radar'].reshape(1, -1), mat['ecg'].reshape(1, -1)
                if expected_len is None or r.shape[1] == expected_len:
                    radar_list.append(r)
                    ecg_list.append(e)
        if not radar_list:
            raise ValueError(
                f"未找到符合长度 expected_len={expected_len} 的样本（或目录中无 .mat）。路径: {data_path}"
            )
        return np.concatenate(radar_list, axis=0), np.concatenate(ecg_list, axis=0)

def standardize(data, eps=1e-8):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / (std + eps)