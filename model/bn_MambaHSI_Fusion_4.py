import torch.nn as nn
import torch
from model.MultiScaleAttention import Attention,Residual,LayerNormalize,MLP_Block
from model.SpaMamba import SpaMamba
from model.SpeMamba import SpeMamba
import torch.nn.functional as F
from einops import rearrange
from model.SSFusion import AdaptiveGlobalLocalFusion
class MambaHSI(nn.Module):
    def __init__(self, in_channels=128,
                 hidden_dim=64,
                 num_classes=10,
                 use_residual=False,
                 mamba_type='spe',
                 token_num=4,
                 use_att=True,
                 depth=1,
                 trans=True,
                 heads=8,
                 value = 9):
        super(MambaHSI, self).__init__()
        self.mamba_type = mamba_type
        self.depth = depth
        self.trans = trans
        self.heads = heads
        self.value = value
        self.weight = nn.Parameter(torch.ones(2))
        # Patch Embedding Layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.silu1 = nn.SiLU()
        self.linear1 = nn.Conv1d(in_channels=112, out_channels=hidden_dim, kernel_size=1) # dim = 64
        # self.linear1 = nn.Conv1d(in_channels=140, out_channels=hidden_dim, kernel_size=1) # dim = 80
        # self.linear1 = nn.Conv1d(in_channels=224, out_channels=hidden_dim, kernel_size=1) # dim = 128
        # self.linear1 = nn.Conv1d(in_channels=56, out_channels=hidden_dim, kernel_size=1) # dim = 32
        # self.linear1 = nn.Conv1d(in_channels=28, out_channels=hidden_dim, kernel_size=1) # dim = 16
        # self.linear1 = nn.Conv1d(in_channels=84, out_channels=hidden_dim, kernel_size=1) # dim = 48,heads = 4

        # Mamba Layers
        if mamba_type == 'spa':
            self.mamba1 = SpaMamba(hidden_dim, use_residual=use_residual)
            self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.mamba2 = SpaMamba(hidden_dim, use_residual=use_residual)
            self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.mamba3 = SpaMamba(hidden_dim, use_residual=use_residual)
        elif mamba_type == 'spe':
            self.mamba1 = SpeMamba(channels= hidden_dim, token_num = 8)
            self.channel_down1 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1)
            self.mamba2 = SpeMamba(channels= hidden_dim // 2, token_num = 8)
            self.channel_down2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=1)
            self.mamba3 = SpeMamba(channels = hidden_dim // 4,token_num = 8)
        elif mamba_type == 'both':
            self.spa_mamba1 = SpaMamba(hidden_dim, use_residual=use_residual)
            self.spa_avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.spe_mamba1 = SpeMamba(hidden_dim, 8)
            self.channel_down1 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1)
            self.spa_mamba2 = SpaMamba(hidden_dim, use_residual=use_residual)
            self.spa_avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.spe_mamba2 = SpeMamba(hidden_dim//2, 8)
            self.channel_down2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=1)
            self.spa_mamba3 = SpaMamba(hidden_dim, use_residual=use_residual)
            self.spe_mamba3 = SpeMamba(hidden_dim//4, 8)

        self.transformer_spa1 = nn.ModuleList(
            [Residual(LayerNormalize(hidden_dim, Attention(hidden_dim, heads=heads, dropout=0.1)))
             for i in range(depth)])
        self.transformer_spa2 = nn.ModuleList(
            [Residual(LayerNormalize(hidden_dim, Attention(hidden_dim, heads=heads, dropout=0.1)))
             for i in range(depth)])
        self.transformer_spa3 = nn.ModuleList(
            [Residual(LayerNormalize(hidden_dim, Attention(hidden_dim, heads=heads, dropout=0.1)))
             for i in range(depth)])

        self.FFN_spa = nn.ModuleList([Residual(
            LayerNormalize(
                hidden_dim, MLP_Block(hidden_dim, hidden_dim * 2, dropout=0.1)))
            for i in range(depth)])


        self.transformer_spe1 = nn.ModuleList(
            [Residual(LayerNormalize(hidden_dim, Attention(hidden_dim, heads=heads, dropout=0.1)))
             for i in range(depth)])
        self.transformer_spe2 = nn.ModuleList(
            [Residual(LayerNormalize(hidden_dim//2, Attention(hidden_dim//2, heads=heads, dropout=0.1)))
             for i in range(depth)])
        self.transformer_spe3 = nn.ModuleList(
            [Residual(LayerNormalize(hidden_dim//4, Attention(hidden_dim//4, heads=heads, dropout=0.1)))
             for i in range(depth)])

        self.FFN_spe = nn.ModuleList([Residual(
            LayerNormalize(
                hidden_dim+hidden_dim//2+hidden_dim//4, MLP_Block(hidden_dim+hidden_dim//2+hidden_dim//4, hidden_dim*2, dropout=0.1)))
            for i in range(depth)])

        if self.mamba_type == 'spe' and not self.trans:
            self.conv2 = nn.Conv1d(in_channels=hidden_dim//4, out_channels=128, kernel_size=1, stride=1)  # 1D卷积
        else:
            self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1)  # 1D卷积

        # Classification Head Layers
        self.bn2 = nn.BatchNorm1d(128)  # 使用 BatchNorm1d
        self.silu2 = nn.SiLU()
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1)  # 1D卷积

    # 模型 B,H,W,C -> B,C
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        B, C, H, W = x.shape

        # Patch Embedding
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu1(x)

        # Mamba Blocks
        if self.mamba_type == 'spe':
            # x1 = self.mamba1(x)  # (9,9)
            x1 = x # 消融添加
            x2 = self.channel_down1(x1)
            # x2 = self.mamba2(x2)
            x3 = self.channel_down2(x2)
            # x3 = self.mamba3(x3)
        if self.mamba_type == 'spa':
            # x1 = self.mamba1(x)  # (9,9)
            x1 = x # 消融添加
            x2 = self.avgpool1(x1)
            # x2 = self.mamba2(x2)
            x3 = self.avgpool2(x2)
            # x3 = self.mamba3(x3)
        if self.mamba_type == 'both' and self.trans:
            # 空间分支
            spa_x1 = self.spa_mamba1(x)  # (9,9)
            spa_x2 = self.spa_avgpool1(spa_x1)  # (4,4)
            spa_x2 = self.spa_mamba2(spa_x2)
            spa_x3 = self.spa_avgpool2(spa_x2)  # (2,2)
            spa_x3 = self.spa_mamba3(spa_x3)

            spe_x1 = self.spe_mamba1(x)  # (9,9,64)
            spe_x2 = self.channel_down1(spe_x1)
            spe_x2 = self.spe_mamba2(spe_x2) # (9,9,32)
            spe_x3 = self.channel_down2(spe_x2)
            spe_x3 = self.spe_mamba3(spe_x3) # (9,9,16)

        # 空间消融
        if self.mamba_type == 'spa' and self.trans:
            spa_x1 = x1
            spa_x2 = x2
            spa_x3 = x3
            # 空间模块
            spa_x1 = rearrange(spa_x1,'b c h w -> b h w c')
            spa_x1 = rearrange(spa_x1,'b h w c-> b (h w) c')
            spa_x2 = rearrange(spa_x2, 'b c h w -> b h w c')
            spa_x2 = rearrange(spa_x2, 'b h w c-> b (h w) c')
            spa_x3 = rearrange(spa_x3, 'b c h w -> b h w c')
            spa_x3 = rearrange(spa_x3, 'b h w c-> b (h w) c')
            LG_1 = spa_x1
            LG_2 = spa_x2
            LG = torch.cat([spa_x1, spa_x2, spa_x3], dim=1)
            for i in range(self.depth):
                LG_1 = self.transformer_spa1[i](LG[:, 0:LG_1.shape[1], :], mask=None)
                LG_2 = self.transformer_spa2[i](LG[:, LG_1.shape[1]:LG_1.shape[1] + LG_2.shape[1], :], mask=None)
                LG_3 = self.transformer_spa3[i](LG[:, LG_1.shape[1] + LG_2.shape[1]:, :], mask=None)
                LG_T = torch.cat([LG_1, LG_2, LG_3], dim=1)
                LG_T = self.FFN_spa[i](LG_T)
                LG = LG + LG_T
            spa_x4 = LG.mean(dim = 1) # B,N,C -> B,C
            x4 = spa_x4
        elif self.mamba_type == 'spa' and not self.trans:
            x4 = x3.mean(dim=[2,3])

        # 光谱消融
        if self.mamba_type == 'spe' and self.trans:
            spe_x1 = x1
            spe_x2 = x2
            spe_x3 = x3
            # 光谱模块
            spe_x1 = rearrange(spe_x1, 'b c h w -> b h w c')
            spe_x1 = rearrange(spe_x1, 'b h w c-> b (h w) c')
            spe_x2 = rearrange(spe_x2, 'b c h w -> b h w c')
            spe_x2 = rearrange(spe_x2, 'b h w c-> b (h w) c')
            spe_x3 = rearrange(spe_x3, 'b c h w -> b h w c')
            spe_x3 = rearrange(spe_x3, 'b h w c-> b (h w) c')
            LG_1 = spe_x1
            LG_2 = spe_x2
            LG = torch.cat([spe_x1, spe_x2, spe_x3], dim=2)

            for i in range(self.depth):
                LG_1 = self.transformer_spe1[i](LG[:,:,0:LG_1.shape[2]], mask=None)
                LG_2 = self.transformer_spe2[i](LG[:,:,LG_1.shape[2]:LG_1.shape[2] + LG_2.shape[2]], mask=None)
                LG_3 = self.transformer_spe3[i](LG[:,:,LG_1.shape[2] + LG_2.shape[2]:], mask=None)
                LG_T = torch.cat([LG_1, LG_2, LG_3], dim=2)
                LG_T = self.FFN_spe[i](LG_T)
                LG = LG + LG_T # 1 112 64
            spe_x4 = LG.mean(dim=1)  #  1 112
            spe_x4 = spe_x4.unsqueeze(2)  # -> [B, 112, 1]
            spe_x4 = self.linear1(spe_x4)  # -> [B, hidden_dim, 1]
            spe_x4 = spe_x4.squeeze(2)  # -> [B, hidden_dim]
            x4 = spe_x4
        elif self.mamba_type == 'spe' and not self.trans:
            x4 = x3.mean(dim=[2,3])

        # 整体模型
        if self.mamba_type == 'both' and self.trans:
            # 空间模块
            spa_x1 = rearrange(spa_x1,'b c h w -> b h w c')
            spa_x1 = rearrange(spa_x1,'b h w c-> b (h w) c')
            spa_x2 = rearrange(spa_x2, 'b c h w -> b h w c')
            spa_x2 = rearrange(spa_x2, 'b h w c-> b (h w) c')
            spa_x3 = rearrange(spa_x3, 'b c h w -> b h w c')
            spa_x3 = rearrange(spa_x3, 'b h w c-> b (h w) c')
            LG_1 = spa_x1
            LG_2 = spa_x2
            LG = torch.cat([spa_x1, spa_x2, spa_x3], dim=1)
            for i in range(self.depth):
                LG_1 = self.transformer_spa1[i](LG[:, 0:LG_1.shape[1], :], mask=None)
                LG_2 = self.transformer_spa2[i](LG[:, LG_1.shape[1]:LG_1.shape[1] + LG_2.shape[1], :], mask=None)
                LG_3 = self.transformer_spa3[i](LG[:, LG_1.shape[1] + LG_2.shape[1]:, :], mask=None)
                LG_T = torch.cat([LG_1, LG_2, LG_3], dim=1)
                LG_T = self.FFN_spa[i](LG_T)
                LG = LG + LG_T
            spa_x4 = LG.mean(dim = 1) # B,N,C -> B,C

            # 光谱模块
            spe_x1 = rearrange(spe_x1, 'b c h w -> b h w c')
            spe_x1 = rearrange(spe_x1, 'b h w c-> b (h w) c')
            spe_x2 = rearrange(spe_x2, 'b c h w -> b h w c')
            spe_x2 = rearrange(spe_x2, 'b h w c-> b (h w) c')
            spe_x3 = rearrange(spe_x3, 'b c h w -> b h w c')
            spe_x3 = rearrange(spe_x3, 'b h w c-> b (h w) c')
            LG_1 = spe_x1
            LG_2 = spe_x2
            LG = torch.cat([spe_x1, spe_x2, spe_x3], dim=2)
            for i in range(self.depth):
                LG_1 = self.transformer_spe1[i](LG[:, :, 0:LG_1.shape[2]], mask=None)
                LG_2 = self.transformer_spe2[i](LG[:, :, LG_1.shape[2]:LG_1.shape[2] + LG_2.shape[2]], mask=None)
                LG_3 = self.transformer_spe3[i](LG[:, :, LG_1.shape[2] + LG_2.shape[2]:], mask=None)
                LG_T = torch.cat([LG_1, LG_2, LG_3], dim=2)
                LG_T = self.FFN_spe[i](LG_T)
                LG = LG + LG_T
            spe_x4 = LG.mean(dim=1)
            spe_x4 = spe_x4.unsqueeze(2)
            spe_x4 = self.linear1(spe_x4)
            spe_x4 = spe_x4.squeeze(2)

            # 可学习权重
            weights = F.softmax(self.weight,dim = 0)
            x4 = weights[0]*spa_x4 + weights[1]*spe_x4

            # 相加
            # x4 = spa_x4 + spe_x4

        x4 = x4.unsqueeze(-1) # B,C -> B，C,1
        x4 = self.conv2(x4)
        x4 = self.bn2(x4)
        x4 = self.silu2(x4)
        x4 = self.conv3(x4)
        x4 = x4.squeeze(-1)
        return x4

