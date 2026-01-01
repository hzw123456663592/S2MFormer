import torch
from torch import nn
from mamba_ssm import Mamba

class SpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
                           d_model=channels,  # Model dimension d_model
                           d_state=16,  # SSM state expansion factor
                           d_conv=4,  # Local convolution width
                           expand=2,  # Block expansion factor
                           )
        # 可学习的权重，用于加权合并四种扫描结果
        self.scan_weights = nn.Parameter(torch.ones(4) / 4)  # 初始化为均等权重
        self.softmax = nn.Softmax(dim=0)  # 对权重进行归一化

        if self.use_proj:
            self.proj = nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.SiLU()
            )

        # 使用平均池化进行特征图降维
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 上采样时根据输入动态调整尺寸
        self.upsample = lambda x, size: nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 从上到下扫描
        x_top_to_bottom = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        x_top_to_bottom = x_top_to_bottom.view(B, H * W, C)
        x_top_to_bottom = self.mamba(x_top_to_bottom)
        x_top_to_bottom = x_top_to_bottom.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # 2. 从下到上扫描（使用池化进行特征图降维）
        x_bottom_to_top = torch.flip(x, dims=[2])
        x_bottom_to_top = self.pool(x_bottom_to_top)  # 特征图降采样
        B, C, H_half, W_half = x_bottom_to_top.shape
        x_bottom_to_top = x_bottom_to_top.permute(0, 2, 3, 1).contiguous().view(B, H_half * W_half, C)
        x_bottom_to_top = self.mamba(x_bottom_to_top)
        x_bottom_to_top = x_bottom_to_top.view(B, H_half, W_half, C).permute(0, 3, 1, 2).contiguous()
        x_bottom_to_top = self.upsample(x_bottom_to_top, size=(H, W))  # 上采样回原尺寸
        x_bottom_to_top = torch.flip(x_bottom_to_top, dims=[2])  # 恢复原始顺序

        # 3. 从左到右扫描
        x_left_to_right = x.permute(0, 3, 2, 1).contiguous()  # B, W, H, C
        x_left_to_right = x_left_to_right.view(B ,W * H, C)
        x_left_to_right = self.mamba(x_left_to_right)
        x_left_to_right = x_left_to_right.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()

        # 4. 从右到左扫描（使用池化进行特征图降维）
        x_right_to_left = torch.flip(x, dims=[3])
        x_right_to_left = self.pool(x_right_to_left)  # 特征图降采样
        B, C, H_half, W_half = x_right_to_left.shape
        x_right_to_left = x_right_to_left.permute(0, 2, 3, 1).contiguous().view(B, H_half * W_half, C)
        x_right_to_left = self.mamba(x_right_to_left)
        x_right_to_left = x_right_to_left.view(B, H_half, W_half, C).permute(0, 3, 1, 2).contiguous()
        x_right_to_left = self.upsample(x_right_to_left, size=(H, W))  # 上采样回原尺寸
        x_right_to_left = torch.flip(x_right_to_left, dims=[3])  # 恢复原始顺序

        # 对权重进行归一化
        weights = self.softmax(self.scan_weights)  # 使用 Softmax 确保权重和为 1
        weights = weights.view(1, 1, 1, 4)  # 调整形状以便广播

        # 加权合并四种扫描结果
        x_recon = (
                weights[..., 0] * x_top_to_bottom +
                weights[..., 1] * x_bottom_to_top +
                weights[..., 2] * x_left_to_right +
                weights[..., 3] * x_right_to_left
        )

        if self.use_proj:
            x_recon = self.proj(x_top_to_bottom)

        if self.use_residual:
            return x_recon + x
        else:
            return x_recon
