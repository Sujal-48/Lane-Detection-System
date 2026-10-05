"""
infer.py
--------
The full hybrid pipeline, end to end:
    raw frame -> preprocess.py (CV: edges + ROI) -> LaneNet (ML: pixel mask)
    -> colored overlay on the original frame.

Run on a video:
    python infer.py --input road_clip.mp4 --checkpoint ../checkpoints/lane_model.pth

Run on an image:
    python infer.py --input road_photo.jpg --image --checkpoint ../checkpoints/lane_model.pth

Run on a webcam:
    python infer.py --input 0 --checkpoint ../checkpoints/lane_model.pth
"""

import argparse
import sys

import cv2
import numpy as np
import torch

from model import LaneNet
from preprocess import make_model_input, INPUT_SIZE

# Same accent colors as the HTML dashboard, so the two feel like one product.
LANE_COLOR = (66, 197, 245)     # amber-ish (BGR)
FILL_COLOR = (208, 217, 63)     # cyan-ish (BGR)


def load_model(checkpoint_path, device):
    model = LaneNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def predict_mask(model, frame_bgr, device, threshold=0.5):
    x = make_model_input(frame_bgr)
    x = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, 4, H, W)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H, W) in [0,1]
    mask = (prob > threshold).astype(np.uint8) * 255
    return mask, prob


def overlay_mask(frame_bgr, mask):
    h, w = frame_bgr.shape[:2]
    mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    color_layer = np.zeros_like(frame_bgr)
    color_layer[mask_full > 0] = FILL_COLOR

    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_layer = np.zeros_like(frame_bgr)
    cv2.drawContours(contour_layer, contours, -1, LANE_COLOR, 4)

    result = cv2.addWeighted(frame_bgr, 1.0, color_layer, 0.35, 0)
    result = cv2.addWeighted(result, 1.0, contour_layer, 0.9, 0)

    coverage = (mask_full > 0).mean() * 100
    cv2.putText(result, f"Lane pixels: {coverage:.1f}%", (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return result


def run_on_image(path, model, device):
    frame = cv2.imread(path)
    if frame is None:
        print(f"Could not read image: {path}")
        sys.exit(1)
    mask, _ = predict_mask(model, frame, device)
    result = overlay_mask(frame, mask)
    cv2.imshow("Hybrid Lane Detection - press any key to close", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_video(source, model, device, output_path=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open video source: {source}")
        sys.exit(1)

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask, _ = predict_mask(model, frame, device)
        result = overlay_mask(frame, mask)
        cv2.imshow("Hybrid Lane Detection - press 'q' to quit", result)
        if writer:
            writer.write(result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Hybrid CV + ML lane detection inference")
    parser.add_argument("--input", required=True, help="Video/image path, or '0' for webcam")
    parser.add_argument("--image", action="store_true", help="Treat --input as a single image")
    parser.add_argument("--checkpoint", default="../checkpoints/lane_model.pth")
    parser.add_argument("--output", default=None, help="Optional path to save processed video")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    print(f"Loaded model on {device}, expecting {INPUT_SIZE} inputs")

    if args.image:
        run_on_image(args.input, model, device)
    else:
        source = int(args.input) if args.input.isdigit() else args.input
        run_on_video(source, model, device, args.output)


if __name__ == "__main__":
    main()
