"""
utils/geometry.py — Module 5
Lift 2D depth pixels into 3D point clouds using camera intrinsics.
Handles intrinsics estimation, projection, and smart subsampling.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging

from config import FOCAL_LENGTH_RATIO, MAX_POINTS_PER_REGION

logger = logging.getLogger(__name__)


def estimate_intrinsics(
    image_h: int,
    image_w: int,
    focal_length_px: Optional[float] = None,
) -> Dict[str, float]:
    """
    Estimate camera intrinsic parameters.

    If focal_length_px is not provided, we use the heuristic:
        fx = fy ≈ image_width
    This corresponds to roughly a 53° horizontal FOV — typical of smartphones
    and standard webcams. Good enough for metric depth from Depth Anything.

    Args:
        image_h         : Image height in pixels
        image_w         : Image width in pixels
        focal_length_px : Override focal length (pixels), None = estimate

    Returns:
        dict with fx, fy, cx, cy (all in pixels)
    """
    if focal_length_px is not None:
        fx = fy = focal_length_px
    else:
        fx = fy = image_w * FOCAL_LENGTH_RATIO

    cx = image_w / 2.0
    cy = image_h / 2.0

    logger.debug(f"Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def pixels_to_3d(
    mask: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: Dict[str, float],
    max_points: int = MAX_POINTS_PER_REGION,
) -> np.ndarray:
    """
    Back-project masked 2D pixels into 3D space using pinhole camera model.

    For each pixel (u, v) in the mask:
        Z = depth_map[v, u]        (meters)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    Args:
        mask        : Binary mask (H×W, uint8, non-zero = include)
        depth_map   : Metric depth map (H×W, float32, meters)
        intrinsics  : Camera intrinsics dict from estimate_intrinsics()
        max_points  : Subsample cap (random sampling if exceeded)

    Returns:
        points : (N, 3) float32 array of 3D points [X, Y, Z]
                 Returns empty array (0,3) if no valid points.
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    # Get pixel coordinates where mask is active
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        logger.warning("Empty mask passed to pixels_to_3d — returning empty point cloud.")
        return np.zeros((0, 3), dtype=np.float32)

    # Extract depth values
    z_vals = depth_map[ys, xs].astype(np.float32)

    # Filter out invalid (zero or negative) depth readings
    valid = z_vals > 0.01
    xs, ys, z_vals = xs[valid], ys[valid], z_vals[valid]

    if len(xs) == 0:
        logger.warning("No valid depth values in masked region.")
        return np.zeros((0, 3), dtype=np.float32)

    # Back-project to 3D
    x_3d = (xs - cx) * z_vals / fx
    y_3d = (ys - cy) * z_vals / fy
    z_3d = z_vals

    points = np.stack([x_3d, y_3d, z_3d], axis=1)

    # Subsample if needed
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        logger.debug(f"Subsampled point cloud: {len(points)} → {max_points} points")

    logger.debug(f"pixels_to_3d: {len(points)} 3D points generated")
    return points


def remove_outliers(
    points: np.ndarray,
    sigma: float = 2.5,
    iqr_multiplier: float = 1.5,
) -> Tuple[np.ndarray, int]:
    """
    Two-stage outlier removal on a 3D point cloud.

    Stage 1 — centroid distance filter (σ-based):
        Remove points farther than `sigma` standard deviations from centroid.

    Stage 2 — IQR filter on Z-axis (depth):
        Remove points with Z outside [Q1 - k*IQR, Q3 + k*IQR].

    Args:
        points         : (N, 3) float32 array
        sigma          : σ threshold for centroid distance
        iqr_multiplier : IQR multiplier for depth axis filter

    Returns:
        filtered_points : (M, 3) cleaned point cloud
        n_removed       : Number of points removed
    """
    if len(points) < 4:
        return points, 0

    n_original = len(points)

    # Stage 1: centroid σ filter
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    mean_d, std_d = distances.mean(), distances.std()
    mask1 = distances < (mean_d + sigma * std_d)
    points = points[mask1]

    if len(points) < 4:
        return points, n_original - len(points)

    # Stage 2: IQR filter on Z (depth axis)
    z = points[:, 2]
    q1, q3 = np.percentile(z, 25), np.percentile(z, 75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    mask2 = (z >= lower) & (z <= upper)
    points = points[mask2]

    n_removed = n_original - len(points)
    logger.debug(
        f"Outlier removal: {n_original} → {len(points)} points ({n_removed} removed)"
    )
    return points, n_removed


def lift_masks_to_3d(
    rim_mask: np.ndarray,
    interior_mask: np.ndarray,
    depth_map: np.ndarray,
    image_h: int,
    image_w: int,
    focal_length_px: Optional[float] = None,
) -> Dict:
    """
    Full 2D→3D lifting pipeline. Entry point for Module 5.

    Args:
        rim_mask        : Binary rim mask (H×W, uint8)
        interior_mask   : Binary interior mask (H×W, uint8)
        depth_map       : Metric depth map in meters (H×W, float32)
        image_h, image_w: Image dimensions
        focal_length_px : Optional focal length override

    Returns:
        dict with:
            intrinsics   : Camera intrinsic parameters
            rim_points   : (N, 3) float32
            base_points  : (M, 3) float32
            n_rim        : int, count before subsampling
            n_base       : int, count before subsampling
    """
    intrinsics = estimate_intrinsics(image_h, image_w, focal_length_px)

    rim_pts_raw = pixels_to_3d(rim_mask, depth_map, intrinsics, max_points=999999)
    base_pts_raw = pixels_to_3d(interior_mask, depth_map, intrinsics, max_points=999999)

    n_rim_raw = len(rim_pts_raw)
    n_base_raw = len(base_pts_raw)

    rim_pts, rim_removed = remove_outliers(rim_pts_raw)
    base_pts, base_removed = remove_outliers(base_pts_raw)

    # Final subsample
    if len(rim_pts) > MAX_POINTS_PER_REGION:
        idx = np.random.choice(len(rim_pts), MAX_POINTS_PER_REGION, replace=False)
        rim_pts = rim_pts[idx]
    if len(base_pts) > MAX_POINTS_PER_REGION:
        idx = np.random.choice(len(base_pts), MAX_POINTS_PER_REGION, replace=False)
        base_pts = base_pts[idx]

    logger.info(
        f"3D lift: rim {n_rim_raw}→{len(rim_pts)} pts, "
        f"base {n_base_raw}→{len(base_pts)} pts"
    )

    return {
        "intrinsics": intrinsics,
        "rim_points": rim_pts,
        "base_points": base_pts,
        "n_rim_raw": n_rim_raw,
        "n_base_raw": n_base_raw,
    }
