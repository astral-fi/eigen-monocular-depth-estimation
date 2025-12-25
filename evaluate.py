import torch
import numpy as np
from tqdm import tqdm

from models.dataset import Dataset
from models.coarse_net import CoarseDepthNet
from models.fine_net import FineDepthNet


def compute_metrics(gt, pred):
    """
    gt, pred: numpy arrays (N,)
    """

    thresh = np.maximum(gt / pred, pred / gt)

    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def evaluate():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = Dataset("data/")
    coarse = CoarseDepthNet().to(DEVICE)
    fine = FineDepthNet().to(DEVICE)

    coarse.load_state_dict(torch.load("checkpoints/coarse_epoch_10.pth"))
    fine.load_state_dict(torch.load("checkpoints/fine_epoch_10.pth"))

    coarse.eval()
    fine.eval()

    metrics = []

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]

        img = sample["image"].unsqueeze(0).to(DEVICE)
        gt = sample["depth"].squeeze().numpy()
        mask = sample["mask"].squeeze().numpy().astype(bool)

        with torch.no_grad():
            coarse_pred = coarse(img)
            pred = fine(img, coarse_pred).squeeze().cpu().numpy()

        # Mask valid pixels
        gt = gt[mask]
        pred = pred[mask]

        # Scale alignment (Eigen protocol)
        scale = np.median(gt) / np.median(pred)
        pred *= scale

        metrics.append(compute_metrics(gt, pred))

    metrics = np.array(metrics).mean(axis=0)

    print("\nEvaluation Results:")
    print(f"AbsRel   : {metrics[0]:.4f}")
    print(f"SqRel    : {metrics[1]:.4f}")
    print(f"RMSE     : {metrics[2]:.4f}")
    print(f"RMSE log : {metrics[3]:.4f}")
    print(f"δ < 1.25 : {metrics[4]:.4f}")
    print(f"δ < 1.25²: {metrics[5]:.4f}")
    print(f"δ < 1.25³: {metrics[6]:.4f}")

if __name__ == "__main__":
    evaluate()

