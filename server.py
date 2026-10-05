"""
server.py
---------
The bridge between the Python ML pipeline and the HTML/CSS/JS dashboard.

The browser can't run PyTorch directly, so this is a tiny local web server:
the dashboard sends it a video frame (as a base64 JPEG), it runs the same
hybrid CV+ML pipeline from src/infer.py, and sends back a lane mask plus a
few numbers for the telemetry panel.

Run it from the project root:
    python server.py --checkpoint checkpoints/lane_model.pth

Then open the dashboard (index.html / Live Server) in a browser tab — as
long as the server is running, the dashboard's "Start detection" button
will call this instead of showing fake demo numbers.
"""

import argparse
import base64
import os
import sys

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from model import LaneNet          # noqa: E402
from preprocess import make_model_input  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")

app = Flask(__name__)
CORS(app)  # the dashboard is served from a different origin/port

MODEL = None
DEVICE = None


def decode_data_url(data_url):
    """'data:image/jpeg;base64,...' -> BGR numpy frame."""
    header, encoded = data_url.split(",", 1)
    binary = base64.b64decode(encoded)
    arr = np.frombuffer(binary, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_mask_png(mask_uint8):
    """Single-channel 0/255 mask -> 'data:image/png;base64,...' string."""
    ok, buf = cv2.imencode(".png", mask_uint8)
    if not ok:
        raise RuntimeError("Failed to encode mask")
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/png;base64,{b64}"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


@app.route("/", methods=["GET"])
def dashboard():
    """Serve the integrated HTML dashboard from the same command as the API."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>", methods=["GET"])
def frontend_assets(filename):
    """Serve dashboard CSS and JavaScript files (but never API routes)."""
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not isinstance(image_data, str):
        return jsonify({"error": "Send JSON with an image data URL in the 'image' field."}), 400
    try:
        frame = decode_data_url(image_data)
    except (ValueError, TypeError, base64.binascii.Error):
        return jsonify({"error": "The image data URL is invalid."}), 400
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    h, w = frame.shape[:2]

    x = make_model_input(frame)
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(x_tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = (prob > 0.5).astype(np.uint8) * 255
    mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(mask_full > 0)
    coverage = float((mask_full > 0).mean() * 100)

    if len(xs) > 20:
        lane_center = float(np.mean(xs))
        offset_pct = ((w / 2) - lane_center) / (w / 2) * 100  # + = left of center
        # crude curvature proxy: spread of x for the upper vs lower half
        upper = xs[ys < h * 0.6]
        lower = xs[ys >= h * 0.6]
        curvature = float(abs(np.std(upper) - np.std(lower))) if len(upper) and len(lower) else 0.0
    else:
        offset_pct = 0.0
        curvature = 0.0

    return jsonify({
        "mask": encode_mask_png(mask),
        "coverage": round(coverage, 2),
        "offset_pct": round(offset_pct, 2),
        "curvature": round(curvature, 2),
    })


def main():
    global MODEL, DEVICE
    parser = argparse.ArgumentParser(description="Serve LaneNet for the HTML dashboard")
    parser.add_argument("--checkpoint", default="checkpoints/lane_model.pth")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = LaneNet().to(DEVICE)
    if not os.path.isfile(args.checkpoint):
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train a model first: py src/train.py --epochs 15"
        )
    MODEL.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    MODEL.eval()
    print(f"Loaded {args.checkpoint} on {DEVICE}")
    print(f"Open http://localhost:{args.port} in your browser")

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
