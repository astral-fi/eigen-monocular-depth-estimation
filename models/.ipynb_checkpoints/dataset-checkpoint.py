import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


class Dataset(Dataset):
    def __init__(self, root_dir, img_size=(224, 224)):
        """
        root_dir: path to kitti_paired
        img_size: (H, W)
        """

        self.img_dir = os.path.join(root_dir, "images_paired")
        self.depth_dir = os.path.join(root_dir, "depths_paired")
        self.img_size = img_size

        self.files = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        #Loading RGB
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        
        #Loading Depth Map
        depth_path = os.path.join(self.depth_dir, fname)
        print(depth_path)
        depth = np.array(Image.open(depth_path))

        depth = depth.astype(np.float32) / 256.0
        
        #Excluding the unvalid depths as in paper
        valid_mask = depth > 0
        depth[~valid_mask] = 0.0

        img = cv2.resize(img, self.img_size)
        depth = cv2.resize(
            depth, self.img_size, interpolation=cv2.INTER_NEAREST
        )
        valid_mask = cv2.resize(
            valid_mask.astype(np.uint8),
            self.img_size,
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        img = img.astype(np.float32) / 255.0

        img = torch.from_numpy(img).permute(2, 0, 1)   # (H, W, 3) -> (3, H, W) As Expected by CNNs
        depth = torch.from_numpy(depth).unsqueeze(0)  # (H, W)->(1, H, W)
        valid_mask = torch.from_numpy(valid_mask).unsqueeze(0)

        return {
            "image": img,
            "depth": depth,
            "mask": valid_mask,
            "filename": fname
        }

