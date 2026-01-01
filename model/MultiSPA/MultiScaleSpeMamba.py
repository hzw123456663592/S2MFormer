import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba

class MultiScaleSpeMamba(nn.Module):
    def __init__(self, channels, token_nums=(4, 8, 16), use_residual=True, use_proj=True):
        super(MultiScaleSpeMamba, self).__init__()
        self.use_residual = use_residual
        self.token_nums = token_nums
        self.num_scales = len(token_nums)
        self.use_proj = use_proj

        self.mambas = nn.ModuleList([
            SpeMamba(channels, token_num=t, use_residual=False) for t in token_nums
        ])

        if self.use_proj:
            self.fusion_weights = nn.Sequential(
                nn.Conv2d(channels * self.num_scales, self.num_scales, kernel_size=1),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        outputs = [mamba(x) for mamba in self.mambas]  # 每个输出: (B, C, H, W)

        if self.use_proj:
            stacked = torch.stack(outputs, dim=1)  # (B, S, C, H, W)
            concat_feats = torch.cat(outputs, dim=1)  # (B, S*C, H, W)
            weights = self.fusion_weights(concat_feats)  # (B, S, H, W)
            weights = weights.unsqueeze(2)  # (B, S, 1, H, W)
            fused = (stacked * weights).sum(dim=1)  # (B, C, H, W)
        else:
            fused = sum(outputs) / len(outputs)  # 简单平均融合（可选）

        if self.use_residual:
            return x + fused
        else:
            return fused


# 原始的 SpeMamba 不变
class SpeMamba(nn.Module):
    def __init__(self, channels, token_num=8, use_residual=True):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba(
            d_model=self.group_channel_num,
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
            return torch.cat([x, pad_features], dim=1)
        return x

    def bidirectional_scan(self, x):
        forward_x = x.clone()
        backward_x = torch.flip(x, dims=[1])
        forward_out = self.mamba(forward_x)
        backward_out = torch.flip(self.mamba(backward_x), dims=[1])
        return forward_out + backward_out

    def forward(self, x):
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape

        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        # x_flat = self.bidirectional_scan(x_flat)

        x_recon = x_flat.view(B, H, W, C_pad).permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)

        if self.use_residual:
            return x + x_proj
        else:
            return x_proj
