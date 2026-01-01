import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleSpectralSpatialCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleSpectralSpatialCNN, self).__init__()

        # 3D CNN blocks for spatial feature enhancement
        self.spatial_conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.spatial_conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=(5, 5, 1), padding=(2, 2, 0))
        self.spatial_conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=(7, 7, 1), padding=(3, 3, 0))

        # 3D CNN blocks for spectral feature enhancement
        self.spectral_conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.spectral_conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 5), padding=(0, 0, 2))
        self.spectral_conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 7), padding=(0, 0, 3))

        # HetConv: a hybrid convolution combining 1x1 and 3x3 filters
        self.hetconv1 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)  # Adjusted input channels
        self.hetconv2 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)  # Adjusted input channels

    def forward(self, x):
        x = x.unsqueeze(2)  # Add an extra dimension for depth (1), shape becomes (B, C, 1, H, W)

        # Spatial feature enhancement
        spatial_feat1 = F.relu(self.spatial_conv1(x))  # (B, out_channels, 1, H, W)
        spatial_feat2 = F.relu(self.spatial_conv2(x))
        spatial_feat3 = F.relu(self.spatial_conv3(x))
        spatial_feat = torch.cat([spatial_feat1, spatial_feat2, spatial_feat3], dim=1)  # (B, 3*out_channels, 1, H, W)

        # Spectral feature enhancement
        spectral_feat1 = F.relu(self.spectral_conv1(x))  # (B, out_channels, 1, H, W)
        spectral_feat2 = F.relu(self.spectral_conv2(x))
        spectral_feat3 = F.relu(self.spectral_conv3(x))
        spectral_feat = torch.cat([spectral_feat1, spectral_feat2, spectral_feat3], dim=1)  # (B, 3*out_channels, 1, H, W)

        # HetConv to fuse features
        hetconv_feat1 = self.hetconv1(spatial_feat.squeeze(2))  # (B, out_channels, H, W)
        hetconv_feat2 = self.hetconv2(spectral_feat.squeeze(2))  # (B, out_channels, H, W)

        # Combine the spatial and spectral features
        combined_feat = hetconv_feat1 + hetconv_feat2  # (B, out_channels, H, W)

        return combined_feat


# Example usage
B, H, W, C = 8, 32, 32, 64  # Batch size, Height, Width, Channels
input_tensor = torch.randn(B, C, H, W)

model = MultiScaleSpectralSpatialCNN(in_channels=C, out_channels=32)
output = model(input_tensor)

print("Output shape:", output.shape)  # Check the output shape
