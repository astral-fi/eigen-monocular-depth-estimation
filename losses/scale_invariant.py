import torch
import torch.nn as nn


class ScaleInvariantLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, mask):
        """
        pred   : (B, 1, H, W) predicted depth
        target : (B, 1, H, W) ground truth depth (meters)
        mask   : (B, 1, H, W) valid depth mask (bool or 0/1)
        """

        # Avoid log(0)
        pred = torch.clamp(pred, min=self.eps)
        target = torch.clamp(target, min=self.eps)

        # Log depth difference
        diff = torch.log(pred) - torch.log(target)

        # Apply mask
        diff = diff * mask

        # Number of valid pixels per image
        N = torch.sum(mask, dim=(1, 2, 3))

        # Per-image loss
        diff_sq = torch.sum(diff ** 2, dim=(1, 2, 3)) / N
        mean_diff = torch.sum(diff, dim=(1, 2, 3)) / N

        loss = diff_sq - mean_diff ** 2

        
        return loss.mean()


