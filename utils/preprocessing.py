"""
utils/preprocessing.py — Module 1
Resize, CLAHE contrast enhancement, bilateral denoising.
Prepares the image for SAM2 and Depth Anything without losing edge info.
"""

import cv2
import numpy as np
from typing import Tuple
import logging

from config import (
    PREPROCESS_MAX_SIZE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
)

logger = logging.getLogger(__name__)


def preprocess_image(
    image_bgr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Preprocess a raw BGR image for the depth pipeline.

    Steps:
      1. Resize so longest side ≤ PREPROCESS_MAX_SIZE (aspect-ratio safe)
      2. CLAHE on the L-channel (LAB colorspace) for contrast enhancement
      3. Bilateral filter for edge-preserving denoising

    Args:
        image_bgr: Raw image in BGR format (H×W×3, uint8)

    Returns:
        processed_bgr : Enhanced image in BGR (resized)
        original_bgr  : Resized (but NOT enhanced) image — for depth model
        scale_factor  : Ratio = new_size / original_size (same for H and W)
    """
    h, w = image_bgr.shape[:2]

    # ── Step 1: Resize ──────────────────────────────────────────────────────
    scale_factor = 1.0
    if max(h, w) > PREPROCESS_MAX_SIZE:
        scale_factor = PREPROCESS_MAX_SIZE / max(h, w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"Resized {w}×{h} → {new_w}×{new_h} (scale={scale_factor:.3f})")

    original_bgr = image_bgr.copy()

    # ── Step 2: CLAHE in LAB space ───────────────────────────────────────────
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    logger.debug("CLAHE applied on L-channel.")

    # ── Step 3: Bilateral denoising ──────────────────────────────────────────
    denoised = cv2.bilateralFilter(
        enhanced_bgr,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )
    logger.debug("Bilateral filter applied.")

    return denoised, original_bgr, scale_factor


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as BGR numpy array. Raises on failure."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Load image from raw bytes (e.g. Streamlit UploadedFile.read())."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from bytes.")
    return img


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR → RGB (for display / model input)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB → BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
