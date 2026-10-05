"""
dataset.py
----------
Two things live here:

1. LaneDataset — loads (image, mask) pairs from data/images/ and
   data/masks/ for training on a real dataset (e.g. TuSimple, CULane —
   export their labels as binary PNG masks with the same filenames as
   the images).

2. generate_synthetic_dataset() — draws simple synthetic road scenes with
   known lane lines, so you can train and see the whole pipeline working
   end-to-end in minutes, with zero downloads, before plugging in a real
   dataset.
"""

import argparse
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess import make_model_input, INPUT_SIZE


class LaneDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.filenames = sorted(
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        image = cv2.imread(os.path.join(self.images_dir, name))
        mask = cv2.imread(os.path.join(self.masks_dir, name), cv2.IMREAD_GRAYSCALE)

        x = make_model_input(image)  # (4, H, W) float32 in [0,1]
        mask = cv2.resize(mask, INPUT_SIZE)
        y = (mask.astype(np.float32) / 255.0)[None, :, :]  # (1, H, W)

        return torch.from_numpy(x), torch.from_numpy(y)


# --------------------------------------------------------------------------- #
# Synthetic data — a quick way to get the whole project running immediately
# --------------------------------------------------------------------------- #

def _draw_synthetic_scene(size=(480, 360)):
    """Return (image, mask) — a plausible-looking road with two curved
    lane lines, plus a binary mask marking exactly those lines."""
    w, h = size
    image = np.full((h, w, 3), (60, 60, 60), dtype=np.uint8)  # asphalt-grey
    image[: int(h * 0.4)] = (150, 110, 70)  # sky band

    mask = np.zeros((h, w), dtype=np.uint8)

    curve = random.uniform(-40, 40)
    base_gap = random.randint(int(w * 0.25), int(w * 0.35))
    center_x = w // 2 + random.randint(-30, 30)

    for side in (-1, 1):
        x_bottom = center_x + side * base_gap
        x_top = center_x + side * (base_gap // 3) + int(curve * side)
        pts = np.array([
            [x_bottom, h],
            [x_top, int(h * 0.45)],
        ])
        color = (40, 200, 230) if side < 0 else (40, 230, 150)
        cv2.line(image, tuple(pts[0]), tuple(pts[1]), color, 8)
        cv2.line(mask, tuple(pts[0]), tuple(pts[1]), 255, 14)

    # a bit of noise so it isn't trivially easy / looks more photographic
    noise = np.random.randint(0, 20, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)

    return image, mask


def generate_synthetic_dataset(out_dir, count=200):
    images_dir = os.path.join(out_dir, "images")
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    for i in range(count):
        image, mask = _draw_synthetic_scene()
        name = f"synthetic_{i:04d}.png"
        cv2.imwrite(os.path.join(images_dir, name), image)
        cv2.imwrite(os.path.join(masks_dir, name), mask)

    print(f"Wrote {count} synthetic image/mask pairs to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic demo dataset")
    parser.add_argument("--out", default="../data", help="Output data/ directory")
    parser.add_argument("--count", type=int, default=200, help="Number of samples")
    args = parser.parse_args()
    generate_synthetic_dataset(args.out, args.count)
