"""
models/depth.py — Module 4
Metric depth estimation using Depth Anything V2.
CRITICAL: Run on FULL image first, then mask-crop. Never crop then depth.
"""

import numpy as np
import torch
import logging
from typing import Tuple, Optional
import cv2

from config import DEPTH_MODEL_NAME

logger = logging.getLogger(__name__)


class MetricDepthEstimator:
    """
    Depth Anything V2 (metric) wrapper.
    Uses transformers pipeline for clean integration.
    Lazy-loads model on first inference call.
    """

    def __init__(self):
        self._pipe = None
        self._device = "cpu"  # Force CPU — CPU-only PyTorch build
        logger.info(f"MetricDepthEstimator on device: {self._device}")

    def _load_model(self):
        if self._pipe is not None:
            return

        from transformers import pipeline as hf_pipeline
        logger.info(f"Loading Depth Anything V2: {DEPTH_MODEL_NAME}")

        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=DEPTH_MODEL_NAME,
            device=-1,  # -1 = CPU for HuggingFace pipeline
        )
        logger.info("Depth Anything V2 loaded.")

    def estimate_depth(
        self,
        image_rgb: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate metric depth for the FULL image.

        IMPORTANT: Always pass the full image, not a cropped container region.
        Cropping removes global scene context and degrades metric accuracy significantly.

        Args:
            image_rgb : Full RGB image (H×W×3, uint8)

        Returns:
            depth_map : Metric depth in meters (H×W, float32)
                        Same spatial resolution as input image.
        """
        self._load_model()

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image_rgb)

        logger.info(f"Running depth estimation on {image_rgb.shape[1]}×{image_rgb.shape[0]} image")

        result = self._pipe(pil_img)
        depth_pil = result["predicted_depth"]  # torch.Tensor or PIL

        # Convert to numpy
        if hasattr(depth_pil, "numpy"):
            depth_np = depth_pil.squeeze().numpy()
        else:
            depth_np = np.array(depth_pil)

        depth_np = depth_np.astype(np.float32)

        # Resize to match original image resolution (model may downsample)
        target_h, target_w = image_rgb.shape[:2]
        if depth_np.shape != (target_h, target_w):
            depth_np = cv2.resize(
                depth_np,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )
            logger.debug(f"Depth map resized to {target_w}×{target_h}")

        logger.info(
            f"Depth map: min={depth_np.min():.3f}m, max={depth_np.max():.3f}m, "
            f"mean={depth_np.mean():.3f}m"
        )
        return depth_np

    def apply_masks(
        self,
        depth_map: np.ndarray,
        rim_mask: np.ndarray,
        interior_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Apply rim and interior masks to the depth map. Also run sanity check.

        Sanity: mean(base_depth) should be > mean(rim_depth)
        because the base is farther from the camera than the opening (rim).

        Args:
            depth_map     : Full metric depth map (H×W, float32, meters)
            rim_mask      : Binary rim mask (H×W, uint8)
            interior_mask : Binary interior (base) mask (H×W, uint8)

        Returns:
            rim_depth    : Depth map with only rim region (zeros elsewhere)
            base_depth   : Depth map with only interior region (zeros elsewhere)
            depth_sane   : True if base is deeper than rim (physics OK)
        """
        rim_depth = depth_map * (rim_mask > 0).astype(np.float32)
        base_depth = depth_map * (interior_mask > 0).astype(np.float32)

        # Compute means (only over valid non-zero pixels)
        rim_vals = rim_depth[rim_mask > 0]
        base_vals = base_depth[interior_mask > 0]

        rim_mean = float(rim_vals.mean()) if len(rim_vals) > 0 else 0.0
        base_mean = float(base_vals.mean()) if len(base_vals) > 0 else 0.0

        depth_sane = base_mean >= rim_mean

        if not depth_sane:
            logger.warning(
                f"Depth sanity FAILED: base_mean={base_mean:.3f}m < rim_mean={rim_mean:.3f}m. "
                "Container may be upside-down, occluded, or depth model struggling."
            )
        else:
            logger.info(
                f"Depth sanity OK: rim_mean={rim_mean:.3f}m, base_mean={base_mean:.3f}m, "
                f"Δ={base_mean - rim_mean:.3f}m"
            )

        return rim_depth, base_depth, depth_sane

    def run_full_depth_pipeline(
        self,
        image_rgb: np.ndarray,
        rim_mask: np.ndarray,
        interior_mask: np.ndarray,
    ) -> dict:
        """
        Full depth pipeline. Entry point for Module 4.

        Args:
            image_rgb     : Preprocessed full RGB image
            rim_mask      : Binary rim mask
            interior_mask : Binary interior mask

        Returns:
            dict with:
                depth_map    : Full depth map (H×W, float32, meters)
                rim_depth    : Masked rim depth
                base_depth   : Masked base depth
                depth_sane   : bool
                rim_mean_m   : Mean rim depth in meters
                base_mean_m  : Mean base depth in meters
        """
        depth_map = self.estimate_depth(image_rgb)
        rim_depth, base_depth, depth_sane = self.apply_masks(depth_map, rim_mask, interior_mask)

        rim_vals = depth_map[rim_mask > 0]
        base_vals = depth_map[interior_mask > 0]

        return {
            "depth_map": depth_map,
            "rim_depth": rim_depth,
            "base_depth": base_depth,
            "depth_sane": depth_sane,
            "rim_mean_m": float(rim_vals.mean()) if len(rim_vals) > 0 else 0.0,
            "base_mean_m": float(base_vals.mean()) if len(base_vals) > 0 else 0.0,
        }
