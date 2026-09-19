# coin-detection-and-counting-project

An automated Computer Vision project built in Python that detects, classifies, and counts both old and new Sri Lankan coins in real time or from static images. The system calculates individual coin values, quantities, total monetary value, and determines camera distance for accurate physical radius measurement.

## ✨ Features

- **Multi-Coin Classification**
  - Supports old and new Sri Lankan coin series (Rs. 1, Rs. 2, Rs. 5, Rs. 10) by comparing estimated physical diameters.
- **Real-time Coin Counting & Valuation**
  - Shows detected denomination and series.
  - Tracks quantity by denomination.
  - Calculates total monetary value.
- **Camera Distance / Depth Estimation**
  - Uses focal length and a known reference coin diameter to estimate camera distance when distance is not provided.
- **High Accuracy Visual Feedback**
  - Draws circles around detected coins.
  - Overlays denomination labels, per-value counts, total value, and camera distance.

## 🛠️ Built With

- Python 3.x
- OpenCV
- NumPy

## 📐 Calibration Formula

The project uses:

\[
\text{Actual Radius (mm)} = \frac{\text{Radius in Pixels} \times \text{Camera Distance (mm)}}{\text{Focal Length (px)}}
\]

Distance estimation (if you provide a known reference coin diameter):

\[
\text{Camera Distance (mm)} = \frac{\text{Known Radius (mm)} \times \text{Focal Length (px)}}{\text{Measured Radius (px)}}
\]

## 🚀 Setup

```bash
python -m pip install -r requirements.txt
```

## ▶️ Usage

### Static image mode

```bash
python coin_detection.py --image /absolute/path/to/image.jpg --focal-length 1200 --distance 400
```

Or estimate distance using a known reference coin diameter:

```bash
python coin_detection.py --image /absolute/path/to/image.jpg --focal-length 1200 --reference-diameter 26.4
```

### Real-time mode (webcam)

```bash
python coin_detection.py --realtime --focal-length 1200 --distance 400
```

Press `q` to quit realtime mode.
