"""最小 LiDAR 条件点扩散模型(类 4D-RaDiff 思路的极简重实现, P1 基线).

- LidarEncoder: 逐点 MLP -> 下采样 token + 全局 max-pool 特征
- Denoiser: 雷达噪声点 (N,5) + t-emb, L 层 [self-attn + cross-attn(LiDAR tokens) + FFN], 预测 ε
规模刻意压小(~2M 参数), 目标是"可跑 + loss 可降 + 可采样", 非 SOTA.
"""
import math

import torch
import torch.nn as nn


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class LidarEncoder(nn.Module):
    def __init__(self, dim=128, n_tokens=256, in_ch=4):
        super().__init__()
        self.n_tokens = n_tokens
        self.mlp = nn.Sequential(nn.Linear(in_ch, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.global_proj = nn.Linear(dim, dim)

    def forward(self, lidar):                       # (B,M,4)
        f = self.mlp(lidar)                         # (B,M,D)
        g = self.global_proj(f.max(1).values)       # (B,D)
        # 均匀抽 token(训练/推理同构, 免 FPS 依赖)
        idx = torch.linspace(0, lidar.shape[1] - 1, self.n_tokens,
                             device=lidar.device).long()
        tokens = f[:, idx]                          # (B,T,D)
        return tokens, g


class Block(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.sa = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.ca = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, x, ctx):
        h = self.n1(x); x = x + self.sa(h, h, h, need_weights=False)[0]
        h = self.n2(x); x = x + self.ca(h, ctx, ctx, need_weights=False)[0]
        x = x + self.ff(self.n3(x))
        return x


class RadarPointDenoiser(nn.Module):
    def __init__(self, dim=128, depth=4, heads=4, pt_ch=5, lidar_ch=4, n_tokens=256):
        super().__init__()
        self.enc = LidarEncoder(dim, n_tokens, lidar_ch)
        self.embed = nn.Linear(pt_ch, dim)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, pt_ch))
        self.dim = dim

    def forward(self, x_t, t, lidar):               # x_t (B,N,5), t (B,), lidar (B,M,4)
        tokens, g = self.enc(lidar)
        te = self.t_mlp(timestep_embedding(t, self.dim))    # (B,D)
        h = self.embed(x_t) + (te + g)[:, None, :]
        for blk in self.blocks:
            h = blk(h, tokens)
        return self.head(h)
