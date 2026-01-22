import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SelfAttention1D(nn.Module):
    """1D自注意力机制"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # 计算注意力
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhcl,bhck->bhlk', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhlk,bhck->bhcl', attn, v)
        out = out.reshape(B, C, L)
        return x + self.proj(out)


class CrossAttention1D(nn.Module):
    """交叉注意力 - 用于条件融合"""
    def __init__(self, channels, cond_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.norm_cond = nn.GroupNorm(8, cond_channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.kv = nn.Conv1d(cond_channels, channels * 2, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x, cond):
        B, C, L = x.shape
        h = self.norm(x)
        cond = self.norm_cond(cond)
        
        q = self.q(h).reshape(B, self.num_heads, C // self.num_heads, L)
        kv = self.kv(cond).reshape(B, 2, self.num_heads, C // self.num_heads, -1)
        k, v = kv[:, 0], kv[:, 1]
        
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhcl,bhck->bhlk', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhlk,bhck->bhcl', attn, v)
        out = out.reshape(B, C, L)
        return x + self.proj(out)


class ResBlock(nn.Module):
    """残差块 + 时间嵌入"""
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        
        # 残差连接
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(t).unsqueeze(-1)
        h = h + time_emb
        
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.skip(x)


class RadarEncoder(nn.Module):
    """专门的雷达特征编码器"""
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, 7, padding=3),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv1d(base_ch, base_ch * 2, 5, padding=2),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
            nn.Conv1d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
        )
        # 多尺度特征
        self.pool2 = nn.AvgPool1d(2)
        self.pool4 = nn.AvgPool1d(4)
        
    def forward(self, x):
        feat = self.encoder(x)
        return {
            'full': feat,
            'half': self.pool2(feat),
            'quarter': self.pool4(feat)
        }


class ConditionalUNet(nn.Module):
    """增强版条件UNet - 带注意力和多尺度条件融合"""
    def __init__(self, base_ch=64, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # 雷达条件编码器
        self.radar_encoder = RadarEncoder(in_ch=1, base_ch=base_ch)
        
        # 编码器 (x_t: 1ch + radar_feat: 128ch = 129ch)
        self.inc = nn.Conv1d(1 + base_ch * 2, base_ch, 3, padding=1)
        
        self.down1 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.attn1 = SelfAttention1D(base_ch * 2)
        self.cross1 = CrossAttention1D(base_ch * 2, base_ch * 2)
        self.pool1 = nn.Conv1d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1)
        
        self.down2 = ResBlock(base_ch * 2, base_ch * 4, time_dim)
        self.attn2 = SelfAttention1D(base_ch * 4)
        self.cross2 = CrossAttention1D(base_ch * 4, base_ch * 2)
        self.pool2 = nn.Conv1d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1)
        
        # 瓶颈层
        self.bottleneck = ResBlock(base_ch * 4, base_ch * 8, time_dim)
        self.bottleneck_attn = SelfAttention1D(base_ch * 8)
        
        # 解码器
        self.up2 = nn.ConvTranspose1d(base_ch * 8, base_ch * 4, 2, 2)
        self.dec2 = ResBlock(base_ch * 8, base_ch * 4, time_dim)
        self.dec_attn2 = SelfAttention1D(base_ch * 4)
        
        self.up1 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, 2, 2)
        self.dec1 = ResBlock(base_ch * 4, base_ch * 2, time_dim)
        self.dec_attn1 = SelfAttention1D(base_ch * 2)
        
        # 输出层
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
            nn.Conv1d(base_ch * 2, 1, 3, padding=1)
        )

    def forward(self, x_t, t, radar_cond):
        # x_t: (B, L, 1) -> (B, 1, L)
        # radar_cond: (B, L, 1) -> (B, 1, L)
        x_t = x_t.transpose(1, 2)
        radar_cond = radar_cond.transpose(1, 2)
        
        # 时间嵌入
        t_emb = self.time_mlp(t)
        
        # 雷达特征编码（多尺度）
        radar_feats = self.radar_encoder(radar_cond)
        
        # 拼接输入
        x = torch.cat([x_t, radar_feats['full']], dim=1)
        x = self.inc(x)
        
        # 编码器
        x1 = self.down1(x, t_emb)
        x1 = self.attn1(x1)
        x1 = self.cross1(x1, radar_feats['full'])
        x1_pool = self.pool1(x1)
        
        x2 = self.down2(x1_pool, t_emb)
        x2 = self.attn2(x2)
        x2 = self.cross2(x2, radar_feats['half'])
        x2_pool = self.pool2(x2)
        
        # 瓶颈
        b = self.bottleneck(x2_pool, t_emb)
        b = self.bottleneck_attn(b)
        
        # 解码器 + skip connections
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, x2], dim=1), t_emb)
        d2 = self.dec_attn2(d2)
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1), t_emb)
        d1 = self.dec_attn1(d1)
        
        out = self.final(d1)
        return out.transpose(1, 2)  # (B, L, 1)