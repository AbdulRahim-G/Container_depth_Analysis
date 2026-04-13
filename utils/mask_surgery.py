"""
utils/mask_surgery.py — Module 3
Transform raw SAM2 mask into semantically distinct rim and interior regions.
The single most underrated step — raw SAM mask = outer silhouette, useless for depth.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List
import logging

from config import (
    RIM_DILATION_RATIO,
    RIM_EROSION_RATIO,
    INTERIOR_EROSION_RATIO,
    BOTTOM_REGION_FRACTION,
    BOTTOM_VISIBILITY_THRESHOLD,
    DEPTH_VARIANCE_THRESHOLD,
    WALL_DEPTH_OVERLAP_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _make_elliptical_kernel(size: int) -> np.ndarray:
    """Create elliptical morphological kernel of given pixel size."""
    size = max(1, size)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def extract_rim_mask(
    sam_mask: np.ndarray,
    mask_width: int,
) -> np.ndarray:
    """
    Extract the rim annular band from a binary SAM mask.

    Method: dilate(mask) − erode(mask) = thin boundary ring
    The dilation captures the outer edge; the erosion keeps only the ring.

    Args:
        sam_mask   : Binary mask (H×W, uint8, values 0 or 255)
        mask_width : Width of mask (for ratio-based kernel sizing)

    Returns:
        rim_mask : Binary rim annular band (H×W, uint8)
    """
    dil_size = max(3, int(mask_width * RIM_DILATION_RATIO))
    ero_size = max(1, int(mask_width * RIM_EROSION_RATIO))

    kernel_dil = _make_elliptical_kernel(dil_size)
    kernel_ero = _make_elliptical_kernel(ero_size)

    dilated = cv2.dilate(sam_mask, kernel_dil, iterations=2)
    eroded = cv2.erode(sam_mask, kernel_ero, iterations=2)

    rim_mask = cv2.bitwise_and(dilated, cv2.bitwise_not(eroded))
    logger.debug(
        f"Rim mask: dil_kernel={dil_size}, ero_kernel={ero_size}, "
        f"rim_pixels={np.sum(rim_mask > 0)}"
    )
    return rim_mask


def extract_interior_mask(
    sam_mask: np.ndarray,
    mask_width: int,
) -> np.ndarray:
    """
    Extract the interior region by aggressively eroding the SAM mask.

    This removes the wall areas and leaves only the central floor/base region
    visible through the opening.

    Args:
        sam_mask   : Binary mask (H×W, uint8)
        mask_width : Width of mask for ratio-based sizing

    Returns:
        interior_mask : Eroded interior region (H×W, uint8)
    """
    ero_size = max(5, int(mask_width * INTERIOR_EROSION_RATIO))
    kernel = _make_elliptical_kernel(ero_size)
    interior = cv2.erode(sam_mask, kernel, iterations=3)
    logger.debug(
        f"Interior mask: ero_kernel={ero_size}, "
        f"interior_pixels={np.sum(interior > 0)}"
    )
    return interior


def check_bottom_visibility(
    interior_mask: np.ndarray,
    depth_map: np.ndarray,
) -> Tuple[float, bool, List[str]]:
    """
    Analyze whether the container's bottom is actually visible.

    Checks the bottom 30% of the interior mask for:
      - Ratio of valid (non-zero) depth pixels
      - Depth variance (high variance = occluded or multipath noise)

    Args:
        interior_mask : Binary interior region (H×W, uint8)
        depth_map     : Full image depth map in meters (H×W, float32)

    Returns:
        visibility_ratio  : Fraction of base pixels with valid depth [0,1]
        is_visible        : True if bottom appears reliably visible
        warnings          : List of warning strings
    """
    warnings = []
    h = interior_mask.shape[0]

    # Take bottom 30% of interior region
    bottom_start = int(h * (1.0 - BOTTOM_REGION_FRACTION))
    bottom_mask = interior_mask.copy()
    bottom_mask[:bottom_start, :] = 0

    total_interior = int(np.sum(interior_mask > 0))
    total_bottom_interior = int(np.sum(bottom_mask > 0))

    if total_bottom_interior == 0:
        warnings.append("No interior pixels found in bottom region — mask may be too small.")
        return 0.0, False, warnings

    # Valid depth pixels in bottom region
    bottom_depth_vals = depth_map[bottom_mask > 0]
    valid_depth = bottom_depth_vals[bottom_depth_vals > 0.01]
    visibility_ratio = len(valid_depth) / total_bottom_interior

    if visibility_ratio < BOTTOM_VISIBILITY_THRESHOLD:
        warnings.append(
            f"Bottom visibility low ({visibility_ratio:.0%}) — "
            "container base may be occluded or not visible."
        )

    # Depth variance check in bottom region
    if len(valid_depth) > 10:
        variance = float(np.var(valid_depth))
        if variance > DEPTH_VARIANCE_THRESHOLD:
            warnings.append(
                f"High depth variance in base region ({variance:.4f} m²) — "
                "depth readings may be noisy or base is curved."
            )

    is_visible = visibility_ratio >= BOTTOM_VISIBILITY_THRESHOLD
    logger.debug(
        f"Bottom visibility: {visibility_ratio:.2%}, "
        f"valid_px={len(valid_depth)}, visible={is_visible}"
    )
    return visibility_ratio, is_visible, warnings


def check_wall_contamination(
    sam_mask: np.ndarray,
    interior_mask: np.ndarray,
    depth_map: np.ndarray,
) -> Tuple[float, List[str]]:
    """
    Check if the wall region depth overlaps too much with the interior depth.
    Wall contamination → rim and base will be at similar depths → bad geometry.

    Args:
        sam_mask      : Full SAM binary mask
        interior_mask : Eroded interior binary mask
        depth_map     : Full depth map in meters

    Returns:
        overlap_score : Histogram intersection score [0,1], higher=more overlap
        warnings      : List of warning strings
    """
    warnings = []

    # Wall region = SAM mask minus interior
    wall_mask = cv2.bitwise_and(sam_mask, cv2.bitwise_not(interior_mask))

    wall_depths = depth_map[wall_mask > 0]
    interior_depths = depth_map[interior_mask > 0]

    wall_depths = wall_depths[wall_depths > 0.01]
    interior_depths = interior_depths[interior_depths > 0.01]

    if len(wall_depths) < 10 or len(interior_depths) < 10:
        return 0.0, warnings

    # Normalized histogram intersection
    bins = 50
    range_min = min(wall_depths.min(), interior_depths.min())
    range_max = max(wall_depths.max(), interior_depths.max())
    hist_range = (range_min, range_max)

    wall_hist, _ = np.histogram(wall_depths, bins=bins, range=hist_range, density=True)
    int_hist, _ = np.histogram(interior_depths, bins=bins, range=hist_range, density=True)

    overlap = float(np.sum(np.minimum(wall_hist, int_hist))) / bins

    if overlap > WALL_DEPTH_OVERLAP_THRESHOLD:
        warnings.append(
            f"High wall/interior depth overlap ({overlap:.2f}) — "
            "container walls and base may be at similar depths. Geometry may be unreliable."
        )

    logger.debug(f"Wall contamination overlap score: {overlap:.3f}")
    return overlap, warnings


def run_mask_surgery(
    sam_mask: np.ndarray,
    depth_map: np.ndarray,
) -> Dict:
    """
    Full mask surgery pipeline. Entry point for Module 3.

    Args:
        sam_mask  : Raw SAM2 binary mask (H×W, uint8, 0 or 255)
        depth_map : Full image depth map from Depth Anything (H×W, float32, meters)

    Returns:
        dict with keys:
            rim_mask        : np.ndarray (H×W, uint8)
            interior_mask   : np.ndarray (H×W, uint8)
            visibility_ratio: float
            is_bottom_visible:bool
            wall_overlap    : float
            warnings        : List[str]
    """
    all_warnings: List[str] = []

    h, w = sam_mask.shape[:2]
    mask_width = w  # Use full image width since SAM mask fills the container

    # Compute bounding box width for relative sizing
    cols = np.where(sam_mask.max(axis=0) > 0)[0]
    if len(cols) > 0:
        mask_width = int(cols[-1] - cols[0])
    mask_width = max(mask_width, 50)  # Safety floor

    # Step A: Rim extraction
    rim_mask = extract_rim_mask(sam_mask, mask_width)

    # Step B: Interior extraction
    interior_mask = extract_interior_mask(sam_mask, mask_width)

    # Sanity: interior must have some pixels
    if np.sum(interior_mask > 0) < 20:
        logger.warning("Interior mask is nearly empty — container may be too small in frame.")
        all_warnings.append(
            "Interior mask is nearly empty. Container may be too small or SAM mask too tight."
        )
        # Fallback: use bottom half of SAM mask as interior
        interior_mask = sam_mask.copy()
        interior_mask[:h // 2, :] = 0

    # Step C: Bottom visibility
    vis_ratio, is_visible, vis_warnings = check_bottom_visibility(interior_mask, depth_map)
    all_warnings.extend(vis_warnings)

    # Step D: Wall contamination
    overlap, overlap_warnings = check_wall_contamination(sam_mask, interior_mask, depth_map)
    all_warnings.extend(overlap_warnings)

    return {
        "rim_mask": rim_mask,
        "interior_mask": interior_mask,
        "visibility_ratio": vis_ratio,
        "is_bottom_visible": is_visible,
        "wall_overlap": overlap,
        "warnings": all_warnings,
    }


def detect_fill_level(
    interior_mask: np.ndarray,
    image_bgr: np.ndarray,
    full_depth_cm: float,
) -> Dict:
    """
    Detect the liquid/content fill level inside a container.

    Strategy — horizontal edge scan on grayscale interior:
      1. Convert image to grayscale, apply interior mask
      2. Compute per-row horizontal Sobel gradient magnitude
      3. The fill surface = the row with the strongest horizontal edge
         inside the interior bounds (excluding top/bottom 15% margins)
      4. Split full_depth_cm proportionally by the fill row position

    Why edges: Contents (liquid, solid, powder) always create a stark
    horizontal boundary with the air above them. Depth cues alone can
    be ambiguous (transparent liquid ≈ air depth), but brightness/color
    discontinuity is virtually always present.

    Args:
        interior_mask : Binary interior mask (H×W, uint8)
        image_bgr     : Original BGR image (H×W×3, uint8)
        full_depth_cm : Total container depth in cm (from geometry engine)

    Returns:
        dict with:
            filled_depth_cm   : cm from bottom to fill surface
            empty_depth_cm    : cm from fill surface to rim (air gap)
            fill_ratio        : fraction filled [0,1]
            fill_row          : image row index of detected fill line
            fill_detected     : bool — False if no clear line found
    """
    # Find vertical extent of interior mask
    interior_rows = np.where(interior_mask.max(axis=1) > 0)[0]
    if len(interior_rows) < 10:
        logger.warning("Interior mask too small for fill-level detection")
        return {
            "filled_depth_cm": None,
            "empty_depth_cm": None,
            "fill_ratio": None,
            "fill_row": None,
            "fill_detected": False,
        }

    top_row = int(interior_rows[0])
    bottom_row = int(interior_rows[-1])
    total_rows = bottom_row - top_row
    if total_rows < 6:
        return {
            "filled_depth_cm": None,
            "empty_depth_cm": None,
            "fill_ratio": None,
            "fill_row": None,
            "fill_detected": False,
        }

    # Grayscale + interior mask
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_masked = gray * (interior_mask > 0).astype(np.float32)

    # Horizontal Sobel (detects horizontal edges = fill surface)
    sobel_y = cv2.Sobel(gray_masked, cv2.CV_32F, 0, 1, ksize=3)
    sobel_y = np.abs(sobel_y) * (interior_mask > 0)

    # Also compute horizontal color gradient (Hue channel captures liquid colour)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[:, :, 0] * (interior_mask > 0)
    sobel_hue = cv2.Sobel(hue, cv2.CV_32F, 0, 1, ksize=3)
    sobel_hue = np.abs(sobel_hue) * (interior_mask > 0)

    # Per-row gradient scores (intensity + hue combined)
    row_score = sobel_y.sum(axis=1) + 0.5 * sobel_hue.sum(axis=1)

    # Search window: exclude top 15% and bottom 15% of interior (rim/base noise)
    margin = max(2, int(total_rows * 0.15))
    search_start = top_row + margin
    search_end = bottom_row - margin

    if search_start >= search_end:
        return {
            "filled_depth_cm": None,
            "empty_depth_cm": None,
            "fill_ratio": None,
            "fill_row": None,
            "fill_detected": False,
        }

    search_scores = row_score[search_start:search_end]
    best_offset = int(np.argmax(search_scores))
    fill_row = search_start + best_offset
    best_score = float(search_scores[best_offset])

    # Confidence check: is the edge strong enough to be a real fill line?
    # Compare peak against mean — a real fill line is much stronger than noise
    mean_score = float(search_scores.mean()) + 1e-6
    edge_snr = best_score / mean_score
    fill_detected = edge_snr >= 2.5  # peak must be 2.5× the mean

    logger.info(
        f"Fill detection: fill_row={fill_row}, SNR={edge_snr:.2f}, "
        f"detected={fill_detected}"
    )

    # Position ratio: 0=top/rim, 1=bottom/base
    fill_fraction_from_top = (fill_row - top_row) / total_rows

    # Air gap = from rim down to fill surface (top portion)
    # Filled = from fill surface down to base (bottom portion)
    empty_ratio = float(np.clip(fill_fraction_from_top, 0.0, 1.0))
    filled_ratio = 1.0 - empty_ratio

    empty_depth_cm = round(full_depth_cm * empty_ratio, 1)
    filled_depth_cm = round(full_depth_cm * filled_ratio, 1)

    return {
        "filled_depth_cm": filled_depth_cm,
        "empty_depth_cm": empty_depth_cm,
        "fill_ratio": round(filled_ratio, 3),
        "fill_row": fill_row,
        "fill_detected": fill_detected,
    }

