"""
models/detector.py — Module 2
YOLO + SAM2 hybrid detection.
YOLO gives coarse bounding boxes; SAM2 refines them into precise masks.
Fallback: grid-point SAM2 prompts when YOLO finds nothing.
"""

import numpy as np
import torch
import logging
from typing import Optional, Tuple, List, Dict

from config import (
    YOLO_MODEL_NAME,
    SAM2_MODEL_NAME,
    CONTAINER_COCO_CLASSES,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_BOX_EXPAND_RATIO,
    SAM2_NUM_GRID_POINTS,
    SAM2_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class ContainerDetector:
    """
    Hybrid YOLO + SAM2 detector.
    Lazy-loads models on first call to avoid slow startup.
    """

    def __init__(self):
        self._yolo = None
        self._sam2_predictor = None
        # Force CPU — CPU-only PyTorch build raises errors if CUDA is attempted
        self._device = "cpu"
        logger.info(f"Detector using device: {self._device}")

    # ── Lazy loaders ─────────────────────────────────────────────────────────

    def _load_yolo(self):
        if self._yolo is not None:
            return
        from ultralytics import YOLO
        logger.info(f"Loading YOLO model: {YOLO_MODEL_NAME}")
        self._yolo = YOLO(YOLO_MODEL_NAME)
        self._yolo.to("cpu")  # explicit CPU — prevents CUDA errors on CPU-only builds
        logger.info("YOLO loaded.")

    def _load_sam2(self):
        if self._sam2_predictor is not None:
            return
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import torch as _torch
            logger.info(f"Loading SAM2 from HuggingFace: {SAM2_MODEL_NAME}")
            self._sam2_predictor = SAM2ImagePredictor.from_pretrained(
                SAM2_MODEL_NAME,
                device=_torch.device("cpu"),  # explicit CPU
            )
            logger.info("SAM2 loaded.")
        except ImportError:
            logger.warning(
                "SAM2 (sam2 package) not installed. "
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )
            self._sam2_predictor = None
        except Exception as e:
            logger.warning(f"SAM2 failed to load ({e}) — will use box-mask fallback")
            self._sam2_predictor = None

    # ── YOLO detection ───────────────────────────────────────────────────────

    def detect_with_yolo(self, image_rgb: np.ndarray) -> List[Dict]:
        """
        Run YOLO on the image and return bounding boxes for container classes.

        Args:
            image_rgb : RGB image (H×W×3, uint8)

        Returns:
            List of dicts: [{"box": [x1,y1,x2,y2], "conf": float, "cls": int}]
            Sorted by confidence (descending).
        """
        self._load_yolo()

        results = self._yolo.predict(
            source=image_rgb,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            classes=CONTAINER_COCO_CLASSES,
            device="cpu",
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": conf,
                    "cls": cls_id,
                })

        detections.sort(key=lambda d: d["conf"], reverse=True)
        logger.info(f"YOLO found {len(detections)} container detections")
        return detections

    def _expand_box(self, box: List[int], image_h: int, image_w: int) -> List[int]:
        """Expand bounding box by YOLO_BOX_EXPAND_RATIO to ensure full container is included."""
        x1, y1, x2, y2 = box
        dw = int((x2 - x1) * YOLO_BOX_EXPAND_RATIO)
        dh = int((y2 - y1) * YOLO_BOX_EXPAND_RATIO)
        x1 = max(0, x1 - dw)
        y1 = max(0, y1 - dh)
        x2 = min(image_w, x2 + dw)
        y2 = min(image_h, y2 + dh)
        return [x1, y1, x2, y2]

    # ── SAM2 segmentation ────────────────────────────────────────────────────

    def segment_with_sam2_box(
        self,
        image_rgb: np.ndarray,
        box: List[int],
    ) -> Optional[np.ndarray]:
        """
        Prompt SAM2 with a bounding box to get a fine segmentation mask.

        Args:
            image_rgb : RGB image (H×W×3, uint8)
            box       : [x1, y1, x2, y2] bounding box

        Returns:
            Binary mask (H×W, uint8, 0 or 255) or None if SAM2 unavailable
        """
        self._load_sam2()
        if self._sam2_predictor is None:
            logger.warning("SAM2 unavailable — returning YOLO box mask as fallback")
            return self._box_to_mask(box, image_rgb.shape[:2])

        h, w = image_rgb.shape[:2]
        expanded_box = self._expand_box(box, h, w)

        with torch.inference_mode():
            self._sam2_predictor.set_image(image_rgb)
            box_np = np.array(expanded_box, dtype=np.float32)
            masks, scores, _ = self._sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_np[None, :],
                multimask_output=True,
            )

        # Pick the highest-scored mask
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_mask = masks[best_idx]

        if best_score < SAM2_SCORE_THRESHOLD:
            logger.warning(f"SAM2 best mask score {best_score:.2f} below threshold — using box fallback")
            return self._box_to_mask(box, image_rgb.shape[:2])

        # Convert bool → uint8 binary
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        logger.debug(f"SAM2 box-prompted mask: score={best_score:.3f}, pixels={mask_uint8.sum()//255}")
        return mask_uint8

    def segment_with_sam2_grid(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback: prompt SAM2 with a grid of evenly-spaced positive points
        and merge the result into a single mask.

        Args:
            image_rgb : RGB image (H×W×3, uint8)

        Returns:
            Best binary mask or None
        """
        self._load_sam2()
        if self._sam2_predictor is None:
            logger.warning("SAM2 unavailable — cannot run grid fallback")
            return None

        h, w = image_rgb.shape[:2]
        n = SAM2_NUM_GRID_POINTS
        xs = np.linspace(w * 0.2, w * 0.8, int(n ** 0.5))
        ys = np.linspace(h * 0.2, h * 0.8, int(n ** 0.5))
        grid_xs, grid_ys = np.meshgrid(xs, ys)
        points = np.stack([grid_xs.flatten(), grid_ys.flatten()], axis=1)
        labels = np.ones(len(points), dtype=np.int64)

        with torch.inference_mode():
            self._sam2_predictor.set_image(image_rgb)
            masks, scores, _ = self._sam2_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )

        # Pick the largest mask (by area) with score above threshold
        valid = [(i, s, masks[i]) for i, s in enumerate(scores) if s >= SAM2_SCORE_THRESHOLD]
        if not valid:
            logger.warning("SAM2 grid fallback found no valid masks")
            return None

        # Largest-area mask
        best = max(valid, key=lambda x: x[2].sum())
        mask_uint8 = (best[2] * 255).astype(np.uint8)
        logger.info(f"SAM2 grid fallback mask: score={best[1]:.3f}, pixels={mask_uint8.sum()//255}")
        return mask_uint8

    @staticmethod
    def _box_to_mask(box: List[int], shape: Tuple[int, int]) -> np.ndarray:
        """Create a filled rectangle mask from a bounding box."""
        mask = np.zeros(shape, dtype=np.uint8)
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 255
        return mask

    # ── Main entry point ─────────────────────────────────────────────────────

    def detect_and_segment(
        self,
        image_rgb: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Full detection pipeline. Entry point for Module 2.

        1. YOLO → bounding box
        2. SAM2 → fine mask prompted by YOLO box
        3. Fallback to SAM2 grid if YOLO detects nothing

        Args:
            image_rgb : Preprocessed RGB image (H×W×3, uint8)

        Returns:
            mask     : Binary mask (H×W, uint8) or None if total failure
            metadata : Detection info (boxes, scores, strategy used)
        """
        metadata = {"strategy": None, "yolo_detections": [], "sam2_score": None}

        # Try YOLO first
        detections = self.detect_with_yolo(image_rgb)
        metadata["yolo_detections"] = detections

        if detections:
            best = detections[0]
            logger.info(f"Using top YOLO detection: cls={best['cls']}, conf={best['conf']:.2f}")
            mask = self.segment_with_sam2_box(image_rgb, best["box"])
            metadata["strategy"] = "YOLO+SAM2"
            metadata["yolo_box"] = best["box"]
        else:
            logger.warning("YOLO found no containers — falling back to SAM2 grid prompting")
            mask = self.segment_with_sam2_grid(image_rgb)
            metadata["strategy"] = "SAM2-grid-fallback"

        if mask is None or mask.max() == 0:
            logger.error("Detection completely failed — no valid mask produced")
            return None, metadata

        logger.info(
            f"Detection done: strategy={metadata['strategy']}, "
            f"mask_pixels={int(mask.sum())//255}"
        )
        return mask, metadata
