# import torch
# import torch.nn as nn
# import math
#
# # from mamba_ssm import Mamba
# from model.mamba_simple import Mamba
# import torch.nn.functional as F
#
# class SpeMamba(nn.Module):
#     def __init__(self, channels, token_num=8, use_residual=True):
#         super(SpeMamba, self).__init__()
#         self.token_num = token_num
#         self.use_residual = use_residual
#         self.group_channel_num = math.ceil(channels / token_num)
#         self.channel_num = self.token_num * self.group_channel_num
#         self.mamba = Mamba(
#             d_model=self.group_channel_num,
#             # d_state=16,
#             # d_conv=4,
#             # expand=2,
#             d_state=8,
#             d_conv=2,
#             expand=1,
#         )
#
#         self.proj = nn.Sequential(
#             nn.BatchNorm2d(channels),
#             nn.SiLU()
#         )
#
#     def padding_feature(self, x):
#         B, C, H, W = x.shape
#         if C < self.channel_num:
#             pad_c = self.channel_num - C
#             pad_features = torch.zeros((B, pad_c, H, W), device=x.device)
#             cat_features = torch.cat([x, pad_features], dim=1)
#             return cat_features
#         else:
#             return x
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_pad = self.padding_feature(x)
#         # 对整个 patch 的每个像素进行建模
#         x_pad = x_pad.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
#         x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
#         x_flat = self.mamba(x_flat)
#         x_recon = x_flat.view(B, H, W, self.channel_num).permute(0, 3, 1, 2).contiguous()
#         x_proj = self.proj(x_recon)
#         if self.use_residual:
#             return x + x_proj
#         else:
#             return x_proj
#


import torch
import torch.nn as nn
import math
from model.mamba_simple import Mamba  # 或 from mamba_ssm import Mamba
import torch.nn.functional as F

class SpeMamba(nn.Module):
    def __init__(self, channels, token_num=8, use_residual=True):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        # 计算每个分组的通道数
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num  # 可能会超过 channels

        # 为每个分组创建一个 Mamba 实例
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=self.group_channel_num,
                d_state=8,
                d_conv=2,
                expand=1,
            ) for _ in range(token_num)
        ])

        # 后处理模块（注意最终只保留原始 channel 数）
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

    def forward(self, x):
        B, C, H, W = x.shape
        x_pad = self.padding_feature(x)  # [B, C_pad, H, W]
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C_pad]

        # 分组
        x_groups = torch.chunk(x_pad, self.token_num, dim=-1)  # 每个是 [B, H, W, group_channel_num]
        outputs = []

        for i in range(self.token_num):
            x_group = x_groups[i]  # [B, H, W, group_channel_num]
            x_flat = x_group.view(B * H * W, 1, self.group_channel_num)  # 单个 token 序列
            x_out = self.mamba_layers[i](x_flat)  # [B*H*W, 1, group_channel_num]
            x_out = x_out.view(B, H, W, self.group_channel_num)
            outputs.append(x_out)

        # 拼接所有输出： [B, H, W, group_channel_num * token_num]
        x_cat = torch.cat(outputs, dim=-1)

        # 转回 [B, C_pad, H, W]
        x_cat = x_cat.permute(0, 3, 1, 2).contiguous()

        # 如果有多余的通道，截断回原始 C
        x_cat = x_cat[:, :C, :, :]

        x_proj = self.proj(x_cat)

        if self.use_residual:
            return x + x_proj
        else:
            return x_proj
