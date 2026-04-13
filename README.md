# 📏 Container Depth Estimator

A production-grade computer vision pipeline that estimates the **physical interior depth** of containers from a **single RGB image** — and detects the **fill level** (filled vs remaining depth) automatically.

Built with **YOLO + SAM2 + Depth Anything V2 + RANSAC geometry** and served via a premium **Streamlit UI**.

---

## 🎬 Demo

Upload any container image (bowl, cup, bucket, specimen jar, water bottle…) and get:

| Output | Description |
|---|---|
| **Full Depth** | Total interior depth in cm ± error margin |
| **Filled Depth** | How much content is inside (cm) |
| **Remaining Depth** | Empty space to the rim (cm) |
| **Confidence** | HIGH / MEDIUM / LOW with per-component scores |
| **3D Point Cloud** | Interactive Plotly visualization of rim + base planes |
| **Depth Heatmap** | Inferno colormap overlaid on the container region |
| **Mask Overlay** | Rim (blue) and base (red) segmentation |

---

## 🧠 Pipeline Architecture

```
SINGLE IMAGE
     │
     ▼
[1] PREPROCESSING          CLAHE contrast + bilateral denoise + resize
     │
     ▼
[2] CONTAINER DETECTION    YOLOv8 coarse box → SAM2 fine mask
     │                     Fallback: SAM2 grid-point prompts
     ▼
[3] MASK SURGERY           Rim band extraction (dilate−erode)
     │                     Interior erosion (removes walls)
     │                     Bottom visibility check
     │                     Wall contamination test
     ▼
[4] METRIC DEPTH MAP       Depth Anything V2 (metric indoor)
     │                     Run on FULL image → then mask-crop
     │                     Sanity: base_depth > rim_depth
     ▼
[5] 2D → 3D LIFT           Pinhole camera back-projection
     │                     Intrinsics estimated: fx=fy=image_width
     │                     2-stage outlier removal (σ + IQR)
     ▼
[6] GEOMETRIC ENGINE       Method 1: Strip-Delta (primary — view-agnostic)
     │                       median(bottom_25%_depth) − median(top_25%_depth)
     │                     Method 2: RANSAC Plane Fitting (secondary)
     │                       only trusted when inlier ratio ≥ 25%
     │                       SVD refinement on inliers
     │                       Parallelism check on plane normals
     │                     Method 3: Visual Height (fallback)
     │                       pixel_span × depth / focal_length
     │                     Weighted combination of plausible estimates
     ▼
[7] FILL LEVEL DETECTION   Horizontal Sobel edge scan on interior mask
     │                     Finds liquid/content surface boundary
     │                     Splits depth → filled + remaining
     ▼
[8] CONFIDENCE SCORING     4-component weighted score:
     │                       plane_fit (35%) + point_count (20%)
     │                       + bottom_visibility (30%) + parallelism (15%)
     │                     Labels: HIGH (≥0.80) / MEDIUM (≥0.55) / LOW
     ▼
[9] OUTPUT + DEBUG VIZ     Annotated image, depth heatmap, 3D point cloud,
                           confidence bar chart, fill-level gauge
```

---

## 📁 Project Structure

```
container-depth/
│
├── app.py                      # Streamlit UI — pure presentation layer
│
├── models/
│   ├── detector.py             # YOLO + SAM2 hybrid detection
│   ├── depth.py                # Depth Anything V2 metric depth
│   └── geometry_engine.py      # Multi-method depth engine (strip-delta + RANSAC)
│
├── utils/
│   ├── preprocessing.py        # CLAHE, bilateral denoise, resize
│   ├── mask_surgery.py         # Rim/base split, fill-level detection
│   ├── geometry.py             # 2D→3D back-projection, outlier removal
│   ├── confidence.py           # 4-component confidence scoring
│   └── viz.py                  # All Plotly + OpenCV visualizations
│
├── config.py                   # All constants — no magic numbers elsewhere
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/AbdulRahim-G/Container_depth_Analysis.git
cd Container_depth_Analysis
```

### 2. Install dependencies

```bash
# Core packages
pip install -r requirements.txt

# PyTorch — choose one:
# CPU (recommended for most users):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1 (if you have an NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# SAM2 (Segment Anything 2)
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 3. Run the app

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **First run**: YOLO, SAM2, and Depth Anything V2 weights download automatically (~500MB total). Subsequent runs are instant.

---

## 📦 Requirements

| Package | Version | Purpose |
|---|---|---|
| streamlit | ≥1.32 | Web UI |
| torch + torchvision | ≥2.1 | Deep learning backend |
| ultralytics | ≥8.2 | YOLOv8 detection |
| transformers | ≥4.40 | Depth Anything V2 |
| sam2 | latest | SAM2 segmentation |
| opencv-python | ≥4.9 | Image processing |
| scikit-learn | ≥1.4 | RANSAC regression |
| plotly | ≥5.20 | 3D visualization |
| numpy, scipy, Pillow | latest | Numerical/image utils |

---

## 🔬 Key Technical Decisions

### Why Strip-Delta over pure RANSAC?

For **side-view** container images, the rim mask captures all four silhouette edges (top + sides + bottom), not just the opening. The resulting RANSAC plane normals are meaningless → wrong perpendicular distance.

The **strip-delta method** fixes this:
```python
depth = median(bottom_25%_of_mask_depth) − median(top_25%_of_mask_depth)
```
This works for any view angle — side-on, top-down, or angled.

### Why run depth on the full image?

Cropping to the container region before depth estimation removes global scene context and degrades metric accuracy significantly. Depth Anything V2 relies on the full scene to anchor its scale.

### Why CLAHE before SAM2?

Metallic, transparent, or wet containers create low-contrast regions that confuse SAM2. CLAHE on the L-channel (LAB colorspace) boosts local contrast while preserving color relationships.

### Fill Level Detection

The liquid surface creates a strong horizontal brightness/color discontinuity. We scan row-wise Sobel gradients (intensity + hue channels) inside the interior mask and find the row with peak gradient — the fill line. An SNR check (peak ≥ 2.5× mean) rejects false positives.

---

## 🎛 Configuration

All tunable parameters live in `config.py`. Key ones:

```python
# Model selection
YOLO_MODEL_NAME = "yolov8m.pt"
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SAM2_MODEL_NAME = "facebook/sam2-hiera-small"

# Geometry
RANSAC_RESIDUAL_THRESHOLD = 0.025   # 2.5cm inlier threshold
MIN_INLIER_RATIO_FOR_PLANE = 0.25   # Below this → use strip-delta
DEPTH_SANITY_MAX_CM = 150.0         # Physical plausibility bound

# Confidence weights
CONFIDENCE_WEIGHTS = {
    "plane_fit": 0.35,
    "point_count": 0.20,
    "bottom_visibility": 0.30,
    "parallelism": 0.15,
}
```

---

## 🐛 Troubleshooting

| Error | Fix |
|---|---|
| `Torch not compiled with CUDA enabled` | Install CPU PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `No module named 'sam2'` | `pip install git+https://github.com/facebookresearch/segment-anything-2.git` |
| `No container detected` | Ensure the container is clearly visible; try a clearer/closer shot |
| Depth very large (>1m) | Depth model may not be metric — check `DEPTH_MODEL_NAME` in config.py |
| Fill level wrong | Ensure the liquid surface is visible; angled shots work better than top-down |

---

## 📊 Accuracy Notes

- **Depth Anything V2 Metric Indoor** is calibrated for indoor close-range scenes (typical accuracy: ±5–15% at 0.5–2m)
- **Strip-delta** cancels out the absolute depth error — only the *relative* accuracy between rim and base matters
- Expected accuracy for containers: **±1–3cm** for clear, well-lit images
- For best results: photograph containers at a **slight angle** (15–45°) so both the rim and base are visible

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — metric monocular depth estimation
- [SAM 2](https://github.com/facebookresearch/segment-anything-2) — Segment Anything Model 2 by Meta AI
- [YOLOv8](https://github.com/ultralytics/ultralytics) — real-time object detection by Ultralytics
- [Streamlit](https://streamlit.io/) — web app framework
