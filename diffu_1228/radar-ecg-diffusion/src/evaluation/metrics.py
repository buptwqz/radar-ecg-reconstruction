import torch
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import correlate
from inference.sampler import ddim_sample, multi_sample_average
from evaluation.visualization import save_plot
import os


def calculate_pearson_correlation(predictions, targets):
    """计算预测值与目标值之间的皮尔逊相关系数"""
    p = predictions.flatten()
    t = targets.flatten()
    if np.std(p) < 1e-6 or np.std(t) < 1e-6:
        return 0.0
    correlation, _ = pearsonr(p, t)
    return correlation


def calculate_rmse(predictions, targets):
    """计算均方根误差"""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def calculate_normalized_correlation(pred, target):
    """计算归一化互相关（对时间偏移更鲁棒）"""
    pred = (pred - np.mean(pred)) / (np.std(pred) + 1e-8)
    target = (target - np.mean(target)) / (np.std(target) + 1e-8)
    
    corr = correlate(pred.flatten(), target.flatten(), mode='full')
    max_corr = np.max(corr) / len(pred)
    return max_corr


def evaluate_model(model, test_loader, scheduler, device, save_dir='diffusion_results'):
    """评估模型性能，使用多种指标"""
    print("\n" + "="*50)
    print("开始评估模型（使用DDIM采样）...")
    print("="*50)
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    all_corrs = []
    all_norm_corrs = []
    all_rmses = []
    
    # 使用更多采样步数
    ddim_steps = 100
    
    for i, (radar, ecg) in enumerate(test_loader):
        radar = radar.to(device)
        
        # 方法1: 单次DDIM采样
        reconstruction = ddim_sample(model, scheduler, radar, ddim_steps=ddim_steps, eta=0.0)
        
        # 方法2: 多次采样取平均（更稳定但更慢）
        # reconstruction = multi_sample_average(model, scheduler, radar, num_samples=3, ddim_steps=ddim_steps)
        
        # 计算指标
        pred = reconstruction.cpu().numpy()
        target = ecg.numpy()
        
        corr = calculate_pearson_correlation(pred, target)
        norm_corr = calculate_normalized_correlation(pred, target)
        rmse = calculate_rmse(pred.flatten(), target.flatten())
        
        all_corrs.append(corr)
        all_norm_corrs.append(norm_corr)
        all_rmses.append(rmse)
        
        # 保存所有样本的对比图
        save_path = os.path.join(save_dir, f'comparison_{i}.png')
        save_plot(target.flatten(), pred.flatten(), corr, save_path)
        
        if i < 5:  # 只打印前5个
            print(f"样本 {i}: Pearson={corr:.4f}, NormCorr={norm_corr:.4f}, RMSE={rmse:.4f}")
    
    avg_corr = np.mean(all_corrs)
    avg_norm_corr = np.mean(all_norm_corrs)
    avg_rmse = np.mean(all_rmses)
    
    print("\n" + "="*50)
    print("评估结果汇总")
    print("="*50)
    print(f"测试样本数: {len(all_corrs)}")
    print(f"平均 Pearson 相关系数: {avg_corr:.4f} (±{np.std(all_corrs):.4f})")
    print(f"平均归一化互相关:     {avg_norm_corr:.4f} (±{np.std(all_norm_corrs):.4f})")
    print(f"平均 RMSE:            {avg_rmse:.4f} (±{np.std(all_rmses):.4f})")
    print(f"最佳 Pearson:         {np.max(all_corrs):.4f}")
    print(f"最差 Pearson:         {np.min(all_corrs):.4f}")
    print(f"\n结果图像已保存到: {save_dir}/")
    print("="*50)
    
    return {
        'pearson': avg_corr,
        'norm_corr': avg_norm_corr,
        'rmse': avg_rmse,
        'all_corrs': all_corrs
    }