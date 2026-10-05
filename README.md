# LaneSight: combined frontend + ML lane detector

This folder combines the two ZIP files into one application. The website is
the user interface; the Python server runs the lane-detection model.

## What was in each ZIP?

| ZIP | What it is | Main files | How it works alone |
| --- | --- | --- | --- |
| `files.zip` | Frontend: HTML, CSS, and JavaScript (not Java) | `index.html`, `style.css`, `script.js` | Open `index.html` in a browser. It plays uploaded video and shows a visual demo overlay. |
| `lane_ml_project2.zip` | Python computer-vision and machine-learning project | `server.py`, `src/` | Train `LaneNet`, then run `server.py`; it exposes a Flask API for predictions. |

The backend uses OpenCV (grayscale, blur, Canny edges, and a road-region mask)
plus a small PyTorch U-Net (`LaneNet`) that makes a pixel-by-pixel lane mask.
Flask is the local web API. There is no Java code in either project.

## Combined layout

```text
lane-detection-combined/
├── frontend/              # website: HTML, CSS, JavaScript
├── server.py              # Flask API and website server
├── src/
│   ├── dataset.py         # load/generate training data
│   ├── train.py           # train and save the model
│   ├── infer.py           # optional webcam/image/video desktop test
│   ├── preprocess.py      # OpenCV preprocessing
│   └── model.py           # PyTorch LaneNet model
├── data/images/           # training road images
├── data/masks/            # matching white/black lane masks
├── checkpoints/           # trained lane_model.pth is created here
└── requirements.txt       # Python packages to install
```

## Start from the beginning (Windows)

### 1. Install a supported Python version

Install **Python 3.12 or 3.13 (64-bit)** from [python.org](https://www.python.org/downloads/windows/).
PyTorch in this project is not yet compatible with Python 3.14. During
installation, select **Add Python to PATH**. Close and reopen VS Code after it
finishes.

In VS Code, open this `lane-detection-combined` folder and choose **Terminal → New Terminal**.
Confirm that 3.12 is available:

```powershell
py -3.12 --version
```

If you installed 3.13 instead, replace every `py -3.12` below with `py -3.13`.

### 2. Create the project environment and install packages

Run these commands once, from this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell says that scripts are blocked, run this once in that terminal and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Create a small practice dataset

This makes 200 simple road images and matching lane masks, so no download is needed:

```powershell
python src\dataset.py --out data --count 200
```

### 4. Train the model

```powershell
python src\train.py --data data --checkpoints checkpoints --epochs 15
```

Wait until it creates `checkpoints\lane_model.pth`. Training on a normal CPU
can take a few minutes. This starter model learns the generated practice
images; use a real labelled lane dataset for meaningful road-video results.

### 5. Start the complete application

```powershell
python server.py --checkpoint checkpoints\lane_model.pth
```

Keep that terminal open. Open **http://localhost:5000** in a browser. Upload a
driving video, press Play, then press **Start detection**. About four video
frames per second are sent to `/predict`; the Python server returns a mask and
telemetry which the dashboard draws over the video.

## Run each original part separately

### Website only

Open `frontend\index.html` in a browser (or VS Code Live Server). Without the
Python server, it deliberately falls back to the animated demo overlay.

### ML pipeline only

After Steps 2–4, run a webcam, video, or still image without the website:

```powershell
python src\infer.py --input 0 --checkpoint checkpoints\lane_model.pth
python src\infer.py --input path\to\road-video.mp4 --checkpoint checkpoints\lane_model.pth
python src\infer.py --input path\to\road-image.jpg --image --checkpoint checkpoints\lane_model.pth
```

Press `q` to close the webcam/video window.

## How the connection works

```text
Browser dashboard → POST /predict with a JPEG video frame
                → Flask server → OpenCV preprocessing → PyTorch LaneNet
Browser dashboard ← JSON: lane mask, coverage, offset, curvature
```

`frontend/script.js` checks `/health` when the page opens. If the server is
available it uses real predictions; if not, the original frontend demo stays
usable. Flask-CORS is included so the frontend can also be opened through
VS Code Live Server during development.

## Important notes

- The ZIP does not include a trained checkpoint or real labelled road data,
  so Step 3 and Step 4 are required before live ML predictions can work.
- For a public deployment, restrict CORS and do not use Flask's development
  server. This setup is intended for local learning and testing.
