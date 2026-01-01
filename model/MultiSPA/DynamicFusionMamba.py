import torch
from torch import nn

from model.MultiSPA.MultiScaleSpaMamba import MultiScaleSpaMamba
from model.MultiSPA.MultiScaleSpeMamba import MultiScaleSpeMamba


class DynamicFusionMamba(nn.Module):
    def __init__(self, channels, token_num, use_residual, use_att=True):
        super(DynamicFusionMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual

        self.spa_mamba = MultiScaleSpaMamba(channels, True, True)
        self.spe_mamba = MultiScaleSpeMamba(channels, use_residual=True, use_proj=True)

        if self.use_att:
            self.pool = nn.AdaptiveAvgPool2d(1)  # 输出 [B, C, 1, 1]
            self.att_fc = nn.Sequential(
                nn.Linear(channels, channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(channels // 4, 2)  # 输出两个融合权重
            )
            self.softmax = nn.Softmax(dim=1)  # 对 [B, 2] 做 softmax

    def forward(self, x):
        spa_x = self.spa_mamba(x)  # 假设输出 [B, C, H, W]
        spe_x = self.spe_mamba(x)  # 假设输出 [B, C, H, W]

        if self.use_att:
            # 全局平均池化 + 权重计算
            global_feat = spa_x + spe_x              # [B, C, H, W]
            pooled = self.pool(global_feat).squeeze(-1).squeeze(-1)  # [B, C]
            weights = self.softmax(self.att_fc(pooled))  # [B, 2]
            w_spa = weights[:, 0].view(-1, 1, 1, 1)
            w_spe = weights[:, 1].view(-1, 1, 1, 1)
            fusion_x = spa_x * w_spa + spe_x * w_spe  # [B, C, H, W]
        else:
            fusion_x = spa_x + spe_x

        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x
