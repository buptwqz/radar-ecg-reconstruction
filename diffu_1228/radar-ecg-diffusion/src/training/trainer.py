import os
import torch
import torch.optim as optim
import copy
from tqdm import tqdm


class EMA:
    """指数移动平均 - 提升模型稳定性"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()
    
    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class DataAugmentation:
    """数据增强 - 扩充小数据集"""
    @staticmethod
    def augment(radar, ecg):
        batch_size = radar.shape[0]
        augmented_radar = radar.clone()
        augmented_ecg = ecg.clone()
        
        for i in range(batch_size):
            # 随机选择增强方式
            aug_type = torch.randint(0, 4, (1,)).item()
            
            if aug_type == 0:
                # 添加小量噪声
                noise_level = 0.05
                augmented_radar[i] += torch.randn_like(radar[i]) * noise_level
                
            elif aug_type == 1:
                # 随机缩放幅度
                scale = 0.8 + torch.rand(1).item() * 0.4  # 0.8-1.2
                augmented_radar[i] *= scale
                augmented_ecg[i] *= scale
                
            elif aug_type == 2:
                # 时间平移（循环移位）
                shift = torch.randint(-20, 21, (1,)).item()
                augmented_radar[i] = torch.roll(radar[i], shifts=shift, dims=0)
                augmented_ecg[i] = torch.roll(ecg[i], shifts=shift, dims=0)
                
            elif aug_type == 3:
                # 随机翻转
                if torch.rand(1).item() > 0.5:
                    augmented_radar[i] = torch.flip(radar[i], dims=[0])
                    augmented_ecg[i] = torch.flip(ecg[i], dims=[0])
        
        return augmented_radar, augmented_ecg


class Trainer:
    def __init__(self, model, scheduler, device, lr=2e-4):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.ema = EMA(model, decay=0.995)
        self.augmentation = DataAugmentation()
        
        # 学习率调度器
        self.lr_scheduler = None
    
    def _setup_lr_scheduler(self, epochs, steps_per_epoch):
        """设置余弦退火学习率调度"""
        total_steps = epochs * steps_per_epoch
        warmup_steps = min(500, total_steps // 10)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _compute_loss(self, radar, ecg, use_augmentation=True):
        """计算损失 - 支持数据增强"""
        if use_augmentation and torch.rand(1).item() > 0.3:
            radar, ecg = self.augmentation.augment(radar, ecg)
        
        # 随机采样时间步（偏向更难的中间步骤）
        if torch.rand(1).item() > 0.5:
            t = torch.randint(0, self.scheduler.steps, (radar.shape[0],), device=self.device)
        else:
            # 更多采样中间时间步
            t = torch.randint(100, 900, (radar.shape[0],), device=self.device)
        
        noise = torch.randn_like(ecg)
        
        # 前向加噪
        x_t = self.scheduler.q_sample(ecg, t, noise)
        
        # 预测噪声
        pred_noise = self.model(x_t, t, radar)
        
        # MSE损失
        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        
        # 添加L1正则化（提升稀疏性）
        l1_loss = torch.nn.functional.l1_loss(pred_noise, noise)
        
        return loss + 0.1 * l1_loss
    
    def train(self, train_loader, epochs):
        print("开始扩散模型训练...")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        self._setup_lr_scheduler(epochs, len(train_loader))
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for radar, ecg in pbar:
                radar, ecg = radar.to(self.device), ecg.to(self.device)
                
                loss = self._compute_loss(radar, ecg, use_augmentation=True)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.ema.update()
                
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                
                epoch_loss += loss.item()
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{current_lr:.2e}'})
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs('checkpoints', exist_ok=True)
                # 使用EMA权重保存
                self.ema.apply_shadow()
                torch.save(self.model.state_dict(), 'checkpoints/best_model.pth')
                self.ema.restore()
                print(f"最佳模型已保存 (loss: {best_loss:.6f})")
            
            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0:
                os.makedirs('checkpoints', exist_ok=True)
                self.ema.apply_shadow()
                torch.save(self.model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')
                self.ema.restore()
                print(f"检查点已保存: checkpoints/model_epoch_{epoch+1}.pth")
        
        # 训练结束后应用EMA权重
        self.ema.apply_shadow()
        print("训练完成，已应用EMA权重")