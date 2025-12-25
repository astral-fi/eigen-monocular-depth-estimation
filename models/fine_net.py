import torch
import torch.nn as nn
import torch.nn.functional as F


class FineDepthNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Input channels: RGB (3) + coarse depth (1) = 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, rgb, coarse_depth):
        """
        rgb          : (B, 3, H, W)
        coarse_depth : (B, 1, H, W)
        """

        x = torch.cat([rgb, coarse_depth], dim=1)  # (B, 4, H, W)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x

