"""
preprocess.py
-------------
The "traditional computer vision" half of the hybrid pipeline.

Turns a raw road frame into the inputs the ML model consumes:
  - a resized RGB frame
  - a Canny edge map, masked to the road region of interest (ROI)

Stacking edges onto the RGB frame gives the model an extra, cheap,
hand-engineered hint about where lane boundaries usually are, so it doesn't
have to learn edge-finding from scratch — that's the point of "combining"
classical CV with the model instead of using either alone.
"""

import cv2
import numpy as np

INPUT_SIZE = (256, 256)  # (width, height) fed to the model


def region_of_interest(edges):
    """Keep only a trapezoid over the road, discard sky/roadside noise."""
    h, w = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(0.05 * w), h),
        (int(0.40 * w), int(0.55 * h)),
        (int(0.60 * w), int(0.55 * h)),
        (int(0.95 * w), h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def edge_map(frame_bgr):
    """Grayscale -> blur -> Canny -> ROI mask. Returns a single-channel
    0-255 edge image the same size as the input frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return region_of_interest(edges)


def make_model_input(frame_bgr, size=INPUT_SIZE):
    """Build the 4-channel tensor-ready array: resized RGB + resized edges.

    Returns a float32 array of shape (4, H, W) in [0, 1], channel order
    [R, G, B, edge].
    """
    resized = cv2.resize(frame_bgr, size)
    edges = edge_map(resized)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    edges_norm = edges.astype(np.float32) / 255.0

    stacked = np.dstack([rgb, edges_norm])          # (H, W, 4)
    return np.transpose(stacked, (2, 0, 1))          # (4, H, W)
