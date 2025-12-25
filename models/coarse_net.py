import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseDepthNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ---------- Encoder ----------
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # ---------- Decoder ----------
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.dec4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.enc1(x)  # (B, 32, H/2, W/2)
        x = self.enc2(x)  # (B, 64, H/4, W/4)
        x = self.enc3(x)  # (B, 128, H/8, W/8)
        x = self.enc4(x)  # (B, 256, H/16, W/16)

        # Decoder
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec1(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec2(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec3(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec4(x)

        return x

