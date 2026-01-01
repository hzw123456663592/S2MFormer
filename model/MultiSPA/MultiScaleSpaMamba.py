import torch
from torch import nn
import torch.nn.functional as F
from model.SpaMamba import SpaMamba

class MultiScaleSpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, use_proj=True):
        super(MultiScaleSpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj

        self.mamba_origin = SpaMamba(channels, use_residual=False, use_proj=False)
        self.mamba_2 = SpaMamba(channels, use_residual=False, use_proj=False)
        self.mamba_4 = SpaMamba(channels, use_residual=False, use_proj=False)

        # 可学习的融合权重
        self.scale_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

        if self.use_proj:
            self.proj = nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.SiLU()
            )

    def forward(self, x):
        B, C, H, W = x.shape

        def center_crop(x, target_h, target_w):
            start_h = (H - target_h) // 2
            start_w = (W - target_w) // 2
            return x[:, :, start_h:start_h + target_h, start_w:start_w + target_w]

        # 多尺度中心裁剪
        patch_2 = center_crop(x, H - 2, W - 2)
        patch_4 = center_crop(x, H - 4, W - 4)

        # 多尺度 Mamba 编码
        out_origin = self.mamba_origin(x)
        out_2 = self.mamba_2(patch_2)
        out_4 = self.mamba_4(patch_4)

        # 上采样到原始尺寸
        out_2 = F.interpolate(out_2, size=(H, W), mode='bilinear', align_corners=False)
        out_4 = F.interpolate(out_4, size=(H, W), mode='bilinear', align_corners=False)

        # 归一化融合权重
        norm_weights = torch.softmax(self.scale_weights, dim=0)

        # 融合三个尺度输出
        fused = (
            norm_weights[0] * out_origin +
            norm_weights[1] * out_2 +
            norm_weights[2] * out_4
        )

        if self.use_proj:
            fused = self.proj(fused)

        if self.use_residual:
            return x + fused
        else:
            return fused
