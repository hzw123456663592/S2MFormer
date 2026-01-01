import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_map = self.conv(x)  # (B, 1, H, W)
        attn_map = self.sigmoid(attn_map)  # 归一化
        return x * attn_map  # 应用注意力权重

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]  # Self-Attention
        x = x + self.mlp(self.norm2(x))  # MLP Block
        return x

class SpaMambaTransformer(nn.Module):
    def __init__(self, channels, use_residual=True, num_heads=4):
        super(SpaMambaTransformer, self).__init__()
        self.use_residual = use_residual
        self.spatial_attention = SpatialAttention(channels)  # 添加空间注意力
        self.transformer = TransformerEncoder(dim=channels, num_heads=num_heads)  # Transformer Encoder

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 计算空间注意力
        x = self.spatial_attention(x)

        # 2. 调整形状为 Transformer 输入格式 (B, L, C)
        x = rearrange(x, "b c h w -> b (h w) c")

        # 3. Transformer 处理
        x = self.transformer(x)

        # 4. 恢复形状 (B, C, H, W)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x
