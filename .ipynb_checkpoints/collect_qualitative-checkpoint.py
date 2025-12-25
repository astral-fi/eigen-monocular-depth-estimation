import os
import torch
from torch.utils.data import DataLoader, Subset
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_ROOT = "results/qualitative"
NUM_SAMPLES = 5   # number of qualitative samples to save
BATCH_SIZE = 5
os.makedirs(SAVE_ROOT, exist_ok=True)

def normalize_to_uint8(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = x.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x = (x * 255.0).astype(np.uint8)
    return x


def save_image(path, img):
    if img.ndim == 2:
        cv2.imwrite(path, img)
    elif img.ndim == 3:
        cv2.imwrite(path, img[:, :, ::-1])  # RGB → BGR


from models.dataset import Dataset
from models.coarse_net import CoarseDepthNet
from models.fine_net import FineDepthNet

coarse = CoarseDepthNet().to(DEVICE)
coarse.load_state_dict(torch.load("./checkpoints/coarse_epoch_10.pth"))
coarse.eval()

fine = FineDepthNet().to(DEVICE)
fine.load_state_dict(torch.load("./checkpoints/fine_epoch_10.pth"))
fine.eval()



dataset = Dataset("./data/")
indices = list(range(len(dataset)))
val_indices = indices[:100]   # take first 100 as validation
dataset = Subset(dataset, val_indices)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

fig, axes = plt.subplots(5, 4, figsize=(16, 18))

with torch.no_grad():

    row = 0

    for batch in tqdm(dataloader):
        if row >= NUM_SAMPLES:
            break

        input_img = batch["image"].to(DEVICE)   # [B, C, H, W]
        gt = batch["depth"].to(DEVICE)              # [B, H, W] or [B, 1, H, W]
        
        coarse_pred = coarse(input_img)
        fine_pred = fine(input_img, coarse_pred)

        inp = input_img[0].permute(1, 2, 0)      # HWC
        gt_ = gt[0].squeeze()
        coarse_pred_ = coarse_pred[0].squeeze()
        fine_pred_ = fine_pred[0].squeeze()


        inp = normalize_to_uint8(inp)
        gt_ = normalize_to_uint8(gt_)
        coarse_pred_ = normalize_to_uint8(coarse_pred_)
        fine_pred_ = normalize_to_uint8(fine_pred_)

        axes[row, 0].imshow(inp)
        axes[row, 0].set_title("RGB")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_, cmap="plasma")
        axes[row, 1].set_title("GT Depth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(coarse_pred_, cmap="plasma")
        axes[row, 2].set_title("Coarse")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(fine_pred_, cmap="plasma")
        axes[row, 3].set_title("Fine")
        axes[row, 3].axis("off")

        row += 1
plt.tight_layout()
plt.savefig("result.png", dpi=200)
plt.show()


