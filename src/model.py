"""
model.py
--------
A deliberately small U-Net for binary, pixel-wise lane segmentation.
Takes the 4-channel (RGB + edge map) input from preprocess.py and predicts
a single-channel probability mask: how likely each pixel is "lane".

Kept small on purpose — this is meant to train fast on modest data/hardware
and to be easy to read end-to-end, not to chase state-of-the-art accuracy.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class LaneNet(nn.Module):
    """Tiny U-Net: 3 down-steps, bottleneck, 3 up-steps with skip connections."""

    def __init__(self, in_channels=4, base=16):
        super().__init__()
        self.enc1 = conv_block(in_channels, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.out_conv = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)  # raw logits, shape (B, 1, H, W)


if __name__ == "__main__":
    # quick sanity check: does a forward pass run and give the right shape?
    model = LaneNet()
    dummy = torch.randn(2, 4, 256, 256)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (2, 1, 256, 256)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
