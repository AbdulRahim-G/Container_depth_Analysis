"""
config.py — Global constants, thresholds, model paths, and scoring weights.
All magic numbers live here, nowhere else.
"""

# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# YOLO model (ultralytics) — using pretrained COCO weights
YOLO_MODEL_NAME = "yolov8m.pt"  # medium-size; good speed/accuracy tradeoff

# Depth Anything V2 — metric indoor (best for close-range objects)
# NOTE: Must use the Metric variant — the plain Small-hf is RELATIVE depth
#       (arbitrary scale) and will NOT give correct metric values.
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
# Options: Metric-Indoor-Small / Metric-Indoor-Base / Metric-Indoor-Large

# SAM2 checkpoint
SAM2_MODEL_NAME = "facebook/sam2-hiera-small"
# Options: tiny / small / base_plus / large

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

PREPROCESS_MAX_SIZE = 1024        # Max longest side in pixels
CLAHE_CLIP_LIMIT = 2.0            # CLAHE contrast limit
CLAHE_TILE_GRID_SIZE = (8, 8)     # CLAHE tile grid
BILATERAL_D = 9                   # Bilateral filter diameter
BILATERAL_SIGMA_COLOR = 75        # Bilateral color sigma
BILATERAL_SIGMA_SPACE = 75        # Bilateral space sigma

# ─────────────────────────────────────────────
# YOLO DETECTION
# ─────────────────────────────────────────────

# COCO class IDs for container-like objects
# 41=cup, 42=fork(skip), 43=knife(skip), 44=spoon(skip), 45=bowl,
# 39=bottle, 74=vase, 75=scissors(skip)
CONTAINER_COCO_CLASSES = [39, 41, 45, 74]  # bottle, cup, bowl, vase
YOLO_CONF_THRESHOLD = 0.25         # Minimum detection confidence
YOLO_IOU_THRESHOLD = 0.45          # NMS IoU threshold
YOLO_BOX_EXPAND_RATIO = 0.05       # Expand bbox by 5% before SAM prompt

# ─────────────────────────────────────────────
# SAM2 SEGMENTATION
# ─────────────────────────────────────────────

SAM2_NUM_GRID_POINTS = 16          # Grid points for fallback prompting
SAM2_MULTIMASK = True              # Return multiple mask candidates
SAM2_SCORE_THRESHOLD = 0.5        # Min SAM mask confidence score

# ─────────────────────────────────────────────
# MASK SURGERY
# ─────────────────────────────────────────────

RIM_DILATION_RATIO = 0.03          # Rim dilation as fraction of mask width
RIM_EROSION_RATIO = 0.005          # Rim inner erosion
INTERIOR_EROSION_RATIO = 0.12      # Interior erosion (removes walls)
BOTTOM_REGION_FRACTION = 0.30      # Bottom 30% of interior = base region
BOTTOM_VISIBILITY_THRESHOLD = 0.35 # Min ratio of valid base pixels (else warn)
DEPTH_VARIANCE_THRESHOLD = 0.08    # Max acceptable depth variance in base (m²)
WALL_DEPTH_OVERLAP_THRESHOLD = 0.5 # Max wall/interior depth histogram overlap

# ─────────────────────────────────────────────
# 3D GEOMETRY — CAMERA INTRINSICS
# ─────────────────────────────────────────────

# Default focal length estimate: fx = fy ≈ image_width
# Valid for ~60° horizontal FOV cameras (most smartphones/webcams)
FOCAL_LENGTH_RATIO = 1.0           # fx = image_width * ratio
MAX_POINTS_PER_REGION = 5000       # Subsample cap for point cloud

# ─────────────────────────────────────────────
# RANSAC PLANE FITTING
# ─────────────────────────────────────────────

RANSAC_MIN_SAMPLES = 3             # Minimum points to fit a plane
RANSAC_RESIDUAL_THRESHOLD = 0.025  # Inlier threshold in meters (2.5 cm)
                                   # Was 5 mm — too tight for metric depth model
                                   # Depth-Anything V2 has ~2-5% relative error
                                   # at 0.5 m distance → noise ≈ 10-25 mm
RANSAC_MAX_TRIALS = 1000           # Maximum RANSAC iterations
RANSAC_STOP_PROBABILITY = 0.9999   # Early stopping confidence

# Minimum inlier ratio to trust a RANSAC plane fit
# Below this → fall back to direct depth-delta estimate
MIN_INLIER_RATIO_FOR_PLANE = 0.25

# Outlier removal
OUTLIER_SIGMA = 2.5                # σ threshold for centroid distance
OUTLIER_IQR_MULTIPLIER = 1.5       # IQR multiplier for depth outliers

# ─────────────────────────────────────────────
# GEOMETRIC VALIDATION
# ─────────────────────────────────────────────

PARALLELISM_COS_THRESHOLD = 0.7    # Min cosine similarity of plane normals
MIN_POINTS_FOR_FIT = 10            # Minimum points to attempt plane fit
DEPTH_SANITY_MIN_CM = 1.0          # Minimum plausible depth (cm)
DEPTH_SANITY_MAX_CM = 150.0        # Maximum plausible depth (cm)

# ─────────────────────────────────────────────
# CONFIDENCE SCORING
# ─────────────────────────────────────────────

CONFIDENCE_WEIGHTS = {
    "plane_fit": 0.35,
    "point_count": 0.20,
    "bottom_visibility": 0.30,
    "parallelism": 0.15,
}

CONFIDENCE_POINT_TARGET = 500      # Points considered "enough" for fitting
CONFIDENCE_INLIER_THRESHOLD = 0.85 # Inlier ratio for "good" fit

CONFIDENCE_THRESHOLDS = {
    "HIGH": 0.80,
    "MEDIUM": 0.55,
    # Below MEDIUM → LOW
}

# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

VIZ_RIM_COLOR = (0, 120, 255)      # Blue (BGR) for rim overlay
VIZ_BASE_COLOR = (0, 0, 220)       # Red (BGR) for base overlay
VIZ_ALPHA = 0.45                   # Mask overlay transparency
VIZ_DEPTH_COLORMAP = "inferno"     # Plotly colorscale for depth heatmap
VIZ_PLANE_OPACITY = 0.25           # 3D plane mesh opacity

# Error margin estimation: ± based on RANSAC residual spread
ERROR_MARGIN_MULTIPLIER = 2.0      # 2× mean residual as ± range
