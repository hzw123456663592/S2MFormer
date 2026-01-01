import torch
import torch.nn as nn
import torch.nn.functional as F
#
# class SoftmaxFusion(nn.Module):
#     def __init__(self, channels, hidden_dim=None):
#         super(SoftmaxFusion, self).__init__()
#         if hidden_dim is None:
#             hidden_dim = channels  # 默认隐藏层维度等于输入维度
#
#         # 共享编码器（权重生成器）
#         self.encoder = nn.Sequential(
#             nn.Linear(channels, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, 1)  # 输出1个打分值
#         )
#
#     def forward(self, Y, P):
#         """
#         Y: Tensor of shape (B, C) - 空间或光谱特征
#         P: Tensor of shape (B, C) - 光谱或空间特征
#         Return:
#             Fused tensor of shape (B, C)
#         """
#         # 计算每个特征的打分（非归一化权重）
#         score_Y = self.encoder(Y)  # (B, 1)
#         score_P = self.encoder(P)  # (B, 1)
#
#         # 拼接后做softmax归一化：动态权重 ∈ (0, 1)，且总和为1
#         scores = torch.cat([score_Y, score_P], dim=1)  # (B, 2)
#         weights = F.softmax(scores, dim=1)  # (B, 2)
#
#         # 加权融合
#         fused = weights[:, 0:1] * Y + weights[:, 1:2] * P  # 广播乘法 (B, C)
#
#         return fused


# class AddThenResidualRefine(nn.Module):
#     def __init__(self, channels):
#         super(AddThenResidualRefine, self).__init__()
#         self.refine = nn.Sequential(
#             nn.Linear(channels, channels),
#             nn.GELU()
#         )
#
#     def forward(self, Y, P):
#         fused = Y + P
#         refined = self.refine(fused)
#         return fused + refined  # 残差增强


import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGlobalLocalFusion(nn.Module):
    def __init__(self, channels, reduction=2):
        super(AdaptiveGlobalLocalFusion, self).__init__()
        reduced_channels = channels * reduction

        # MLP: W2 -> BN -> ReLU -> W1 -> Sigmoid
        self.fusion_mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels),  # W2
            # nn.BatchNorm1d(reduced_channels),
            nn.SiLU(),
            nn.Linear(reduced_channels, 1),         # W1
            nn.Sigmoid()                            # 输出 Wg ∈ (B, 1)
        )

    def forward(self, G, L):
        """
        G: global feature, shape (B, C)
        L: local feature, shape (B, C)
        """
        fused = G + L                    # (B, C)
        z = self.fusion_mlp(fused)      # z ∈ (B, 1), Wg
        Wg = z                          # (B, 1)
        Wl = 1 - Wg                     # (B, 1)

        # 权重乘法 + 残差连接
        out = G + L + Wg * G + Wl * L   # (B, C)
        return out
