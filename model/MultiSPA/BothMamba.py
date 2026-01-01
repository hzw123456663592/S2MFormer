import torch
from torch import nn

from model.MultiSPA.MultiScaleSpaMamba import MultiScaleSpaMamba
from model.MultiSPA.MultiScaleSpeMamba import MultiScaleSpeMamba


class BothMamba(nn.Module):
    def __init__(self,channels,token_num,use_residual,use_att=True):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)


        self.spa_mamba = MultiScaleSpaMamba(channels, True, True)
        self.spe_mamba = MultiScaleSpeMamba(channels, use_residual=True, use_proj=True)

    def forward(self,x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x
        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x