import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba

# 分组+双向扫描
class SpeMamba(nn.Module):
    def __init__(self, channels, token_num=8, use_residual=True):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba(
            d_model=self.group_channel_num,  # 分组
            # d_model=channels, # 双向
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.proj = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W), device=x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def bidirectional_scan(self, x):
        """
        正向 + 反向 扫描
        """
        # 正向扫描输入
        forward_x = x.clone()
        # 反向扫描输入（沿 token 维度翻转）
        backward_x = torch.flip(x, dims=[1])

        # 经过 Mamba 计算
        forward_out = self.mamba(forward_x)
        backward_out = torch.flip(self.mamba(backward_x), dims=[1])

        # 结果融合（这里简单相加，也可以尝试拼接等方式）
        return forward_out + backward_out

    # 分组+双向
    def forward(self, x):
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape

        # 进行双向扫描
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_flat = self.bidirectional_scan(x_flat)

        x_recon = x_flat.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)

        if self.use_residual:
            return x + x_proj
        else:
            return x_proj

    # 双向
    # def forward(self, x):
    #     """
    #     x: (B, C, H, W)
    #     """
    #     B, C, H, W = x.shape
    #     L = H * W  # 展平后的序列长度
    #
    #     # (B, C, H, W) -> (B, L, C)
    #     x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)
    #
    #     # 双向扫描
    #     x = self.bidirectional_scan(x)
    #
    #     # 恢复形状 -> (B, C, H, W)
    #     x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    #
    #     # 投影 + 归一化
    #     x_proj = self.proj(x)
    #
    #     if self.use_residual:
    #         return x + x_proj
    #     else:
    #         return x_proj
