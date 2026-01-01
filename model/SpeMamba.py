# import torch
# import torch.nn as nn
# import math
# from model.mamba_simple import Mamba  # 或 from mamba_ssm import Mamba
# import torch.nn.functional as F
# class SpeMamba(nn.Module):
#     def __init__(self, channels, token_num=8, use_residual=True):
#         super(SpeMamba, self).__init__()
#         self.token_num = token_num
#         self.use_residual = use_residual
#
#         # 计算每个分组的通道数
#         self.group_channel_num = math.ceil(channels / token_num)
#         self.channel_num = self.token_num * self.group_channel_num  # 可能会超过 channels
#
#         # 为每个分组创建一个 Mamba 实例
#         self.mamba_layers = nn.ModuleList([
#             Mamba(
#                 d_model=self.group_channel_num,
#                 d_state=8,
#                 d_conv=2,
#                 expand=1,
#             ) for _ in range(token_num)
#         ])
#
#         # 后处理模块（注意最终只保留原始 channel 数）
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
#         x_pad = self.padding_feature(x)  # [B, C_pad, H, W]
#         x_pad = x_pad.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C_pad]
#
#         # 分组
#         x_groups = torch.chunk(x_pad, self.token_num, dim=-1)  # 每个是 [B, H, W, group_channel_num]
#         outputs = []
#
#         for i in range(self.token_num):
#             x_group = x_groups[i]  # [B, H, W, group_channel_num]
#             x_flat = x_group.view(B * H * W, 1, self.group_channel_num)  # 单个 token 序列
#             x_out = self.mamba_layers[i](x_flat)  # [B*H*W, 1, group_channel_num]
#             x_out = x_out.view(B, H, W, self.group_channel_num)
#             outputs.append(x_out)
#
#         # 拼接所有输出： [B, H, W, group_channel_num * token_num]
#         x_cat = torch.cat(outputs, dim=-1)
#
#         # 转回 [B, C_pad, H, W]
#         x_cat = x_cat.permute(0, 3, 1, 2).contiguous()
#
#         # 如果有多余的通道，截断回原始 C
#         x_cat = x_cat[:, :C, :, :]
#
#         x_proj = self.proj(x_cat)
#
#         if self.use_residual:
#             return x + x_proj
#         else:
#             return x_proj


# 间隔分组
import torch
import torch.nn as nn
import math
from model.mamba_simple import Mamba  # 或 from mamba_ssm import Mamba
import torch.nn.functional as F

class SpeMamba(nn.Module):
    def __init__(self, channels, token_num=4, use_residual=True):
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
            pad_features = torch.zeros((B, pad_c, H, W), device=x.device, dtype=x.dtype)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self, x):
        B, C, H, W = x.shape
        x_pad = self.padding_feature(x)  # [B, C_pad, H, W]，其中 C_pad = token_num * group_channel_num
        C_pad = x_pad.shape[1]

        # 转成 [B, H, W, C_pad]，方便按通道做跳跃分组
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C_pad]

        outputs = []

        # 跳跃分组：
        # 第 i 组取通道：i, i + token_num, i + 2*token_num, ...
        # 由于 C_pad = token_num * group_channel_num，
        # 每一组得到的最后一维长度恰好是 group_channel_num
        for i in range(self.token_num):
            # [B, H, W, group_channel_num]
            x_group = x_pad[..., i::self.token_num]

            # 保险起见（理论上长度一定是 group_channel_num）
            if x_group.shape[-1] != self.group_channel_num:
                # 如果因为某些原因不等，做一下 padding
                pad_c = self.group_channel_num - x_group.shape[-1]
                x_group = F.pad(x_group, (0, pad_c))

            # 展平成序列喂给 Mamba： [B*H*W, 1, group_channel_num]
            x_flat = x_group.view(B * H * W, 1, self.group_channel_num)
            x_out = self.mamba_layers[i](x_flat)  # [B*H*W, 1, group_channel_num]
            x_out = x_out.view(B, H, W, self.group_channel_num)
            outputs.append(x_out)

        # 拼接所有输出： [B, H, W, group_channel_num * token_num]
        x_cat = torch.cat(outputs, dim=-1)

        # 转回 [B, C_pad, H, W]
        x_cat = x_cat.permute(0, 3, 1, 2).contiguous()

        # 截断回原始通道 C
        x_cat = x_cat[:, :C, :, :]

        x_proj = self.proj(x_cat)

        if self.use_residual:
            return x + x_proj
        else:
            return x_proj

