import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import os

from models.dataset import Dataset
from models.coarse_net import CoarseDepthNet
from losses.scale_invariant import ScaleInvariantLoss


def train():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 1e-4
    IMG_SIZE = (224, 224)

    DATA_ROOT = "data/"
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # -------- Dataset & Loader --------
    dataset = Dataset(DATA_ROOT, img_size=IMG_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # -------- Model --------
    model = CoarseDepthNet().to(DEVICE)

    # -------- Loss & Optimizer --------
    criterion = ScaleInvariantLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # -------- Training Loop --------
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            img = batch["image"].to(DEVICE)
            depth = batch["depth"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            optimizer.zero_grad()

            pred = model(img)
            loss = criterion(pred, depth, mask)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        # -------- Save checkpoint --------
        ckpt_path = os.path.join(SAVE_DIR, f"coarse_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)

    print("Training complete.")


if __name__ == "__main__":
    train()

