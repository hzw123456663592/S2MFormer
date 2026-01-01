import torch.nn as nn
import torch
from model.Conv_fusion import SimpleFusion
# from model.fusionBlock import FusionBlock
from model.BothMamba import BothMamba
from model.SpaMamba import SpaMamba
from model.SpeMamba import SpeMamba
import torch.nn.functional as F

class MambaHSI(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, use_residual=False, mamba_type='spe', token_num=4, use_att=True,use_weighted_fusion=True):
        super(MambaHSI, self).__init__()
        self.mamba_type = mamba_type
        self.use_weighted_fusion = use_weighted_fusion
        # Patch Embedding Layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.silu1 = nn.SiLU()

        self.fusion = SimpleFusion(hidden_dim)
        # self.fusion = FusionBlock(hidden_dim)

        # Mamba Layers
        if mamba_type == 'spa':
            self.mamba1 = SpaMamba(hidden_dim, use_residual=use_residual)
            self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.mamba2 = SpaMamba(hidden_dim, use_residual=use_residual)
            self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.mamba3 = SpaMamba(hidden_dim, use_residual=use_residual)
        elif mamba_type == 'spe':
            self.mamba1 = SpeMamba(hidden_dim, 8)
            self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.mamba2 = SpeMamba(hidden_dim, 8)
            self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.mamba3 = SpeMamba(hidden_dim, 8)
        elif mamba_type == 'both':
            self.mamba1 = BothMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual, use_att=use_att)
            self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.mamba2 = BothMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual, use_att=use_att)
            self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.mamba3 = BothMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual, use_att=use_att)

        self.weights = nn.Parameter(torch.ones(3) / 3)  # 三个分支
        self.softmax = nn.Softmax(dim=0)

        # Classification Head Layers
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(128)  # 替换 GroupNorm 为 BatchNorm
        self.silu2 = nn.SiLU()
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

    # 模型端到端 B,H,W,C -> B,C,H,W
    def forward(self, x):
        # Permute and reshape input
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        B, C, H, W = x.shape

        # Patch Embedding
        x = self.conv1(x)
        x = self.bn1(x)  # 直接使用 BatchNorm
        x = self.silu1(x)

        # Mamba Blocks
        if self.mamba_type == 'spa' or self.mamba_type == 'spe' or self.mamba_type == 'both':
            x1 = self.mamba1(x)  # (9,9)
            x2 = self.avgpool1(x1)  # (4,4)
            x2 = self.mamba2(x2)
            x3 = self.avgpool2(x2) # (2,2)
            x4 = self.mamba3(x3)
        if self.use_weighted_fusion and self.mamba_type == 'both':
            B1,C1,H1,W1 = x1.shape
            B2,C2,H2,W2 = x2.shape
            B3,C3,H3,W3 = x3.shape
            # 下采样 (9,9) -> (4,4)
            x1 = F.interpolate(x1, (H2, W2), mode='bilinear', align_corners=True)  # B, C, H, W
            x1_x2 = self.fusion(x1, x2) # (4,4)
            # 下采样 (4,4) -> (2,2)
            x2 = F.interpolate(x2, (H3, W3), mode='bilinear', align_corners=True)  # B, C, H, W
            x2_x3 = self.fusion(x2,x3)   # (2,2)
            # 下采样 (4,4) -> (2,2)
            x1_x2 = F.interpolate(x1_x2, (H3, W3), mode='bilinear', align_corners=True)  # B, C, H, W
            x1_x2_x3 = self.fusion(x1_x2,x2_x3)
            B4,C4,H4,W4 = x1_x2_x3.shape

            # 自适应加权融合
            w = self.softmax(self.weights)  # [3]
            x = w[0] * x1_x2 + w[1] * x2_x3 + w[2] * x1_x2_x3  # shape: [B, C, H, W]

            x = x + x4
        else:
            x4 = F.interpolate(x4,(H,W), mode='bilinear',align_corners=True)
            x = x4
        # Classification Head
        x = self.conv2(x)
        x = self.bn2(x)  # 直接使用 BatchNorm
        x = self.silu2(x)
        logits = self.conv3(x)

        # Interpolate to original size
        logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)  # B, C, H, W
        return logits


# if __name__ == "__main__":
#     x = torch.randn(512, 9, 9, 128).cuda()  # 加上 .cuda()
#     B, H, W, C = x.shape
#     net = MambaHSI(in_channels=C, num_classes=9, hidden_dim=128, mamba_type="both").cuda()  # 模型也要放到 CUDA
#
#     out = net(x)
#     print(f"Output shape: {out.shape}")
