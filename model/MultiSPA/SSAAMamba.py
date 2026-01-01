import torch
from torch import nn
from model.MultiSPA.MultiScaleSpaMamba import MultiScaleSpaMamba
from model.MultiSPA.MultiScaleSpeMamba import MultiScaleSpeMamba


class SSAAMamba(nn.Module):
    def __init__(self, channels, token_num=None, use_residual=True):
        super(SSAAMamba, self).__init__()
        self.use_residual = use_residual
        self.channels = channels

        self.spa_mamba = MultiScaleSpaMamba(channels, use_residual=True, use_proj=True)
        self.spe_mamba = MultiScaleSpeMamba(channels, use_residual=True, use_proj=True)

        # Linear projection to align dimensions if needed
        # self.spa_proj = nn.Linear(channels, channels)
        # self.spe_proj = nn.Linear(channels, channels)

        # Learnable attention kernel q (1 × C vector)
        self.query = nn.Parameter(torch.randn(channels))

        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # x shape: (B, C, H, W) or (B, N, C), depending on your data
        spa_feat = self.spa_mamba(x)    # (B, C, H, W)
        spe_feat = self.spe_mamba(x)    # (B, C, H, W)

        # Global pooling to 1xC vector (assuming (B, C, H, W) input)
        spa_vec = torch.mean(spa_feat, dim=(2, 3))  # -> (B, C)
        spe_vec = torch.mean(spe_feat, dim=(2, 3))  # -> (B, C)

        # Linear projection
        # spa_proj = self.spa_proj(spa_vec)  # (B, C)
        # spe_proj = self.spe_proj(spe_vec)  # (B, C)

        # Attention score: dot with query
        spa_score = torch.sum(spa_vec * self.query, dim=1)  # (B,)
        spe_score = torch.sum(spe_vec * self.query, dim=1)  # (B,)

        # Stack and normalize
        scores = torch.stack([spa_score, spe_score], dim=0)  # (2, B)
        weights = self.softmax(scores)  # (2, B)

        # Expand weights for broadcasting
        w_spa = weights[0].view(-1, 1, 1, 1)
        w_spe = weights[1].view(-1, 1, 1, 1)

        # Weighted fusion
        fusion = w_spa * spa_feat + w_spe * spe_feat

        if self.use_residual:
            fusion = fusion + x

        return fusion