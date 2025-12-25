import os
import torch
from torch.utils.data import DataLoader, Subset
import cv2
import numpy as np
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_ROOT = "results/qualitative"
NUM_SAMPLES = 5   # number of qualitative samples to save
BATCH_SIZE = 1
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
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

with torch.no_grad():
    sample_idx = 0

    for batch in tqdm(dataloader):
        if sample_idx >= NUM_SAMPLES:
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

        sample_dir = os.path.join(SAVE_ROOT, f"sample_{sample_idx:02d}")
        os.makedirs(sample_dir, exist_ok=True)

        save_image(os.path.join(sample_dir, "input.png"), inp)
        save_image(os.path.join(sample_dir, "gt.png"), gt_)
        save_image(os.path.join(sample_dir, "coarse_prediction.png"), coarse_pred_)
        save_image(os.path.join(sample_dir, "fine_prediction.png"), fine_pred_)

        sample_idx += 1

print("Qualitative results saved to:", SAVE_ROOT)

