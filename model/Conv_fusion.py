import torch.nn as nn
import torch
class SimpleFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.fuse(x)
