import torch
import torch.nn.functional as F


@torch.no_grad()
def ddim_sample(model, scheduler, radar_cond, ddim_steps=100, eta=0.0, guidance_scale=1.0):
    """
    改进的DDIM采样
    - 更多采样步数
    - 可选的guidance scale
    """
    model.eval()
    device = next(model.parameters()).device
    b = radar_cond.shape[0]
    seq_len = radar_cond.shape[1]
    
    # 从纯噪声开始
    x = torch.randn((b, seq_len, 1), device=device)
    
    # 定义采样的时间序列（从大到小）
    times = torch.linspace(scheduler.steps - 1, 0, ddim_steps + 1).long().to(device)
    
    for i in range(ddim_steps):
        t_cur = times[i]
        t_next = times[i + 1]
        
        t = torch.full((b,), t_cur.item(), device=device, dtype=torch.long)
        
        # 预测噪声
        pred_noise = model(x, t, radar_cond)
        
        # 如果使用guidance
        if guidance_scale > 1.0:
            # 无条件预测（用零条件）
            uncond_noise = model(x, t, torch.zeros_like(radar_cond))
            # Classifier-free guidance
            pred_noise = uncond_noise + guidance_scale * (pred_noise - uncond_noise)
        
        # 获取alpha参数
        alpha_bar = scheduler.alpha_cumprod[t_cur].view(-1, 1, 1)
        
        # 计算预测的x0
        pred_x0 = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
        
        # 限制范围
        pred_x0 = torch.clamp(pred_x0, -5, 5)
        
        if t_next > 0:
            alpha_bar_next = scheduler.alpha_cumprod[t_next].view(-1, 1, 1)
            
            # DDIM更新
            sigma = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar)) * \
                    torch.sqrt(1 - alpha_bar / alpha_bar_next)
            
            # 计算方向
            direction = torch.sqrt(1 - alpha_bar_next - sigma**2) * pred_noise
            
            # 随机噪声
            noise = torch.randn_like(x) if eta > 0 else 0
            
            x = torch.sqrt(alpha_bar_next) * pred_x0 + direction + sigma * noise
        else:
            x = pred_x0
    
    return x


@torch.no_grad()
def ddpm_sample(model, scheduler, radar_cond, steps=None):
    """
    标准DDPM采样（更慢但可能质量更高）
    """
    model.eval()
    device = next(model.parameters()).device
    b = radar_cond.shape[0]
    seq_len = radar_cond.shape[1]
    steps = steps or scheduler.steps
    
    x = torch.randn((b, seq_len, 1), device=device)
    
    for i in reversed(range(steps)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        
        pred_noise = model(x, t, radar_cond)
        
        alpha = scheduler.alpha[i]
        alpha_bar = scheduler.alpha_cumprod[i]
        beta = scheduler.beta[i]
        
        # DDPM更新公式
        if i > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta)
        else:
            noise = 0
            sigma = 0
        
        # 计算均值
        pred_x0 = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
        pred_x0 = torch.clamp(pred_x0, -5, 5)
        
        mean = (torch.sqrt(alpha) * (1 - scheduler.alpha_cumprod_prev[i]) * x + 
                torch.sqrt(scheduler.alpha_cumprod_prev[i]) * beta * pred_x0) / (1 - alpha_bar)
        
        x = mean + sigma * noise
    
    return x


@torch.no_grad()
def multi_sample_average(model, scheduler, radar_cond, num_samples=5, ddim_steps=100):
    """
    多次采样取平均 - 提升稳定性
    """
    samples = []
    for _ in range(num_samples):
        sample = ddim_sample(model, scheduler, radar_cond, ddim_steps=ddim_steps, eta=0.2)
        samples.append(sample)
    
    return torch.stack(samples).mean(dim=0)