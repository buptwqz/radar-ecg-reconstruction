import torch
import torch.nn as nn
import numpy as np
class DiffusionScheduler:
    """管理前向加噪和 DDIM/DDPM 采样逻辑"""
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.steps = steps
        self.device = device
        self.beta = torch.linspace(beta_start, beta_end, steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.ones(1).to(device), self.alpha_cumprod[:-1]])
        
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    def q_sample(self, x_0, t, noise):
        """加噪过程: x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise"""
        target_device = x_0.device
        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod[t].to(target_device).view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[t].to(target_device).view(-1, 1, 1)
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise