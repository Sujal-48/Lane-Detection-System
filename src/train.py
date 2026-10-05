"""
train.py
--------
Trains LaneNet on (image, mask) pairs in data/images/ + data/masks/.
Saves the best checkpoint to checkpoints/lane_model.pth and a simple
loss-curve plot to checkpoints/loss_curve.png so you have something
visual to check progress with.

Run:
    python train.py --epochs 15
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split

from dataset import LaneDataset
from model import LaneNet


def dice_loss(pred_logits, target, eps=1e-6):
    """Dice loss — plays nicer than plain BCE on thin, sparse lane pixels."""
    pred = torch.sigmoid(pred_logits)
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = LaneDataset(
        images_dir=os.path.join(args.data, "images"),
        masks_dir=os.path.join(args.data, "masks"),
    )
    if len(dataset) == 0:
        raise SystemExit(
            f"No images found in {args.data}/images — generate a demo set first:\n"
            f"  python dataset.py --out {args.data} --count 200"
        )

    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = LaneNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()

    os.makedirs(args.checkpoints, exist_ok=True)
    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = bce(logits, y) + dice_loss(logits, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_ds)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"Epoch {epoch:02d}/{args.epochs} — train loss {train_loss:.4f} — val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.checkpoints, "lane_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved best checkpoint -> {ckpt_path}")

    _plot_history(history, os.path.join(args.checkpoints, "loss_curve.png"))


def _plot_history(history, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping loss curve plot")
        return
    plt.figure(figsize=(6, 4))
    plt.plot(history["train"], label="train loss")
    plt.plot(history["val"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("LaneNet training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved loss curve -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LaneNet")
    parser.add_argument("--data", default="../data", help="Directory with images/ and masks/")
    parser.add_argument("--checkpoints", default="../checkpoints", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
