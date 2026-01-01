import torch.nn as nn
# from model.BothMamba import BothMamba
# from model.MultiSPA.SSAAMamba import SSAAMamba
from model.MultiSPA.DynamicFusionMamba import DynamicFusionMamba as SSAAMamba
from model.MultiSPA.MultiScaleSpaMamba import MultiScaleSpaMamba
from model.MultiSPA.MultiScaleSpeMamba import MultiScaleSpeMamba
import torch.nn.functional as F

class MambaHSI(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, use_residual=True, mamba_type='spe', token_num=8,use_att=True):
        super(MambaHSI, self).__init__()
        self.mamba_type = mamba_type

        # Patch Embedding Layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.silu1 = nn.SiLU()

        # Mamba Layers
        if mamba_type == 'spa':
            self.mamba1 = MultiScaleSpaMamba(hidden_dim,True,True)
            self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1, ceil_mode=True)
            self.mamba2 = MultiScaleSpaMamba(hidden_dim,True,True)
            self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1, ceil_mode=True)

        elif mamba_type == 'spe':
            self.mamba1 = MultiScaleSpeMamba(hidden_dim, use_residual=True, use_proj=True)
            self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1, ceil_mode=True)
            self.mamba2 = MultiScaleSpeMamba(hidden_dim, use_residual=True, use_proj=True)
            self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1, ceil_mode=True)
        elif mamba_type == 'both':
            self.mamba1 = SSAAMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual)
            # self.mamba1 = BothMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual, use_att=use_att)
            self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1,ceil_mode=True)
            self.mamba2 = SSAAMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual)
            # self.mamba2 = BothMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual, use_att=use_att)
            self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1,ceil_mode=True)

        # Classification Head Layers
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.silu2 = nn.SiLU()
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Permute and reshape input
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        B, C, H, W = x.shape

        # Patch Embedding
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu1(x)

        # Mamba Blocks
        if self.mamba_type == 'spa' or self.mamba_type == 'spe' or self.mamba_type == 'both':
            x = self.mamba1(x)
            x = self.avgpool1(x)
            x = self.mamba2(x)
            x = self.avgpool2(x)

        # Classification Head
        x = self.conv2(x)
        x = self.bn2(x)  # 直接使用 BatchNorm
        x = self.silu2(x)
        logits = self.conv3(x)

        # Interpolate to original size
        logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)  # B, C, H, W
        return logits
