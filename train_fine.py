import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import os

from models.dataset import Dataset
from models.coarse_net import CoarseDepthNet
from models.fine_net import FineDepthNet
from losses.scale_invariant import ScaleInvariantLoss


def train():
    # -------- Config --------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 1e-4
    IMG_SIZE = (224, 224)

    DATA_ROOT = "data/"
    COARSE_CKPT = "checkpoints/coarse_epoch_10.pth"
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # -------- Dataset --------
    dataset = Dataset(DATA_ROOT, img_size=IMG_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # -------- Models --------
    coarse = CoarseDepthNet().to(DEVICE)
    coarse.load_state_dict(torch.load(COARSE_CKPT, map_location=DEVICE))
    coarse.eval()

    # Freeze coarse network
    for p in coarse.parameters():
        p.requires_grad = False

    fine = FineDepthNet().to(DEVICE)

    # -------- Loss & Optimizer --------
    criterion = ScaleInvariantLoss()
    optimizer = Adam(fine.parameters(), lr=LR)

    # -------- Training Loop --------
    for epoch in range(EPOCHS):
        fine.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Fine Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            img = batch["image"].to(DEVICE)
            gt_depth = batch["depth"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            optimizer.zero_grad()

            with torch.no_grad():
                coarse_depth = coarse(img)

            refined_depth = fine(img, coarse_depth)
            loss = criterion(refined_depth, gt_depth, mask)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        torch.save(
            fine.state_dict(),
            os.path.join(SAVE_DIR, f"fine_epoch_{epoch+1}.pth")
        )

    print("Fine network training complete.")


if __name__ == "__main__":
    train()

