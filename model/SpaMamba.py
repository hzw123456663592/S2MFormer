import math
import torch
from torch import nn
from mamba_ssm import Mamba
# from model.mamba_simple import Mamba
import torch.nn.functional as F
class SpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(
            d_model=channels,
            d_state=8,
            d_conv=2,
            expand=1,
        )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.SiLU()
            )

    # 蛇形扫描
    def snake_scan(self, x):
        B, H, W, C = x.shape
        out = []
        for i in range(H):
            if i % 2 == 0:
                out.append(x[:, i, :, :])  # left to right
            else:
                out.append(x[:, i, torch.arange(W - 1, -1, -1), :])  # right to left
        out = torch.cat(out, dim=1)  # (B, H*W, C)
        return out

    def snake_unscan(self, x, H, W):
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        out = torch.zeros_like(x)
        for i in range(H):
            if i % 2 == 0:
                out[:, i, :, :] = x[:, i, :, :]
            else:
                out[:, i, :, :] = x[:, i, torch.arange(W - 1, -1, -1), :]
        return out


    # 按行展开
    # def row_scan(self, x):
    #     B, H, W, C = x.shape
    #     out = []
    #     for i in range(H):
    #         out.append(x[:, i, :, :])  # 每一行都从左到右
    #     out = torch.cat(out, dim=1)  # (B, H*W, C)
    #     return out
    #
    # def row_unscan(self, x, H, W):
    #     B, _, C = x.shape
    #     x = x.view(B, H, W, C)  # 直接按行还原
    #     return x


    # # 按列展开
    # def col_scan(self, x):
    #     B, H, W, C = x.shape
    #     out = []
    #     for j in range(W):
    #         out.append(x[:, :, j, :])  # 取每一列：B, H, C
    #     out = torch.cat(out, dim=1)  # 拼接成 B, H*W, C
    #     return out
    # def col_unscan(self, x, H, W):
    #     B, _, C = x.shape
    #     x = x.view(B, H, W, C)  # 按列顺序展开的还原方式和行顺序一样
    #     out = torch.zeros_like(x)
    #     for j in range(W):
    #         out[:, :, j, :] = x[:, :, j, :]
    #     return out

    # 四个方向
    # 左上角 逐行
    def row_scan_lu(self, x):
        B, H, W, C = x.shape
        return x.view(B, H * W, C)

    def row_unscan_lu(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        return x

    # 左上角 逐列
    def col_scan_lu(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 2, 1, 3)  # B, W, H, C
        return x.contiguous().view(B, H * W, C)

    def col_unscan_lu(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, W, H, C).permute(0, 2, 1, 3).contiguous()
        return x

    # 右下角 逐行
    def row_scan_rd(self, x):
        B, H, W, C = x.shape
        x = torch.flip(x, dims=[1, 2])  # 翻转H和W
        return x.view(B, H * W, C)

    def row_unscan_rd(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        return torch.flip(x, dims=[1, 2])

    # 右下角 逐列
    def col_scan_rd(self, x):
        B, H, W, C = x.shape
        x = torch.flip(x, dims=[1, 2])
        x = x.permute(0, 2, 1, 3)  # B, W, H, C
        return x.contiguous().view(B, H * W, C)

    def col_unscan_rd(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, W, H, C).permute(0, 2, 1, 3).contiguous()
        return torch.flip(x, dims=[1, 2])

    def forward(self, x):
        x_re = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        B, H, W, C = x_re.shape
        # 蛇形扫描
        x_snake = self.snake_scan(x_re)  # B, H*W, C
        x_out = self.mamba(x_snake)  # B, H*W, C
        x_recon = self.snake_unscan(x_out, H, W)  # B, H, W, C
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # B, C, H, W

        # 逐行扫描
        # x_row = self.row_scan(x_re)  # B, H*W, C
        # x_out = self.mamba(x_row)  # B, H*W, C
        # x_recon = self.row_unscan(x_out, H, W)  # B, H, W, C
        # x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # B, C, H, W

        # 逐列扫描
        # x_col = self.col_scan(x_re)  # B, H*W, C
        # x_out = self.mamba(x_col)  # B, H*W, C
        # x_recon = self.col_unscan(x_out, H, W)  # B, H, W, C
        # x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # B, C, H, W

        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon

    # 四个方向结合
    # def forward(self, x):
    #     x_re = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
    #     B, H, W, C = x_re.shape
    #
    #     outputs = []
    #
    #     for scan_func, unscan_func in [
    #         (self.row_scan_lu, self.row_unscan_lu),
    #         (self.col_scan_lu, self.col_unscan_lu),
    #         (self.row_scan_rd, self.row_unscan_rd),
    #         (self.col_scan_rd, self.col_unscan_rd),
    #     ]:
    #         x_seq = scan_func(x_re)
    #         x_out = self.mamba(x_seq)
    #         x_back = unscan_func(x_out, H, W)
    #         outputs.append(x_back)
    #
    #     x_recon = sum(outputs) / len(outputs)  # 融合四个方向
    #
    #     x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
    #
    #     if self.use_proj:
    #         x_recon = self.proj(x_recon)
    #     if self.use_residual:
    #         return x + x_recon
    #     else:
    #         return x_recon





