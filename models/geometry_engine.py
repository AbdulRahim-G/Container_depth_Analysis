"""
models/geometry_engine.py — Module 6
THE REAL ENGINE. Multi-method depth estimation with automatic fallback.

Why RANSAC alone fails for side-view containers:
  - The rim_mask captures ALL silhouette edges (top+sides+bottom), not just the opening
  - The fitted plane normal is meaningless → wrong perpendicular distance
  - Inlier ratios are low → planes are fit to noise

Three-method approach:
  1. Strip-Delta (PRIMARY): top-25% vs bottom-25% of mask depth median difference
     Works for ANY view angle — side-on, top-down, angled
  2. RANSAC Planes (SECONDARY): only trusted when inlier ratios ≥ 0.35
  3. Visual Height (FALLBACK): pixel_span × depth / focal_length
     Gives exterior height ≈ interior depth for most containers

Selection: weighted combination based on per-method confidence scores.
"""

import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import logging
from typing import Tuple, Optional, Dict

from config import (
    RANSAC_MIN_SAMPLES,
    RANSAC_RESIDUAL_THRESHOLD,
    RANSAC_MAX_TRIALS,
    RANSAC_STOP_PROBABILITY,
    PARALLELISM_COS_THRESHOLD,
    MIN_INLIER_RATIO_FOR_PLANE,
    MIN_POINTS_FOR_FIT,
    DEPTH_SANITY_MIN_CM,
    DEPTH_SANITY_MAX_CM,
    ERROR_MARGIN_MULTIPLIER,
    OUTLIER_SIGMA,
    OUTLIER_IQR_MULTIPLIER,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — STRIP DELTA (primary, view-angle agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def strip_delta_depth(
    sam_mask: np.ndarray,
    depth_map: np.ndarray,
    strip_fraction: float = 0.25,
) -> Optional[Dict]:
    """
    Compute depth as:  median(bottom_strip_depth) − median(top_strip_depth)

    Works for ANY view angle:
      • Side view  → top strip = opening rim, bottom strip = container base
      • Top-down   → top strip = far rim, bottom strip = near base
      • Angled     → interpolates correctly between the two

    This avoids all 3D geometry complications — purely a depth-map measurement
    on the two ends of the container.

    Args:
        sam_mask       : Full container binary mask (H×W, uint8)
        depth_map      : Metric depth map in meters (H×W, float32)
        strip_fraction : Fraction of vertical span for each strip (default 25%)

    Returns:
        dict with depth_m, depth_cm, top_median_m, bot_median_m, n_top, n_bot
        or None if insufficient pixels
    """
    rows_with_pixels = np.where(sam_mask.max(axis=1) > 0)[0]
    if len(rows_with_pixels) < 6:
        logger.warning("strip_delta_depth: SAM mask has too few rows")
        return None

    top_row = int(rows_with_pixels[0])
    bot_row = int(rows_with_pixels[-1])
    span = bot_row - top_row

    if span < 4:
        return None

    margin = max(2, int(span * strip_fraction))

    # Top strip mask (opening / rim side)
    top_strip = np.zeros_like(sam_mask)
    top_strip[top_row: top_row + margin, :] = sam_mask[top_row: top_row + margin, :]

    # Bottom strip mask (base side)
    bot_strip = np.zeros_like(sam_mask)
    bot_strip[max(0, bot_row - margin + 1): bot_row + 1, :] = \
        sam_mask[max(0, bot_row - margin + 1): bot_row + 1, :]

    top_depths = depth_map[top_strip > 0]
    bot_depths = depth_map[bot_strip > 0]

    top_valid = top_depths[(top_depths > 0.01) & (top_depths < 30.0)]
    bot_valid = bot_depths[(bot_depths > 0.01) & (bot_depths < 30.0)]

    if len(top_valid) < 5 or len(bot_valid) < 5:
        logger.warning(
            f"strip_delta_depth: insufficient valid pixels "
            f"(top={len(top_valid)}, bot={len(bot_valid)})"
        )
        return None

    top_med = float(np.median(top_valid))
    bot_med = float(np.median(bot_valid))
    delta = bot_med - top_med

    if delta <= 0.001:
        logger.warning(
            f"strip_delta_depth: negative/zero delta={delta:.4f}m "
            "(top deeper than bottom — depth model issue or wrong orientation)"
        )
        # Try the absolute value; depth map might be inverted for this model
        delta = abs(delta)
        if delta <= 0.001:
            return None

    depth_cm = delta * 100.0
    logger.info(
        f"Strip-delta: top={top_med:.3f}m, bot={bot_med:.3f}m, "
        f"Δ={delta:.3f}m = {depth_cm:.1f}cm"
    )

    return {
        "depth_m": round(delta, 4),
        "depth_cm": round(depth_cm, 2),
        "top_median_m": top_med,
        "bot_median_m": bot_med,
        "n_top": len(top_valid),
        "n_bot": len(bot_valid),
        # Error estimate: IQR-based spread of the two strips
        "error_m": float(
            np.percentile(bot_valid, 75) - np.percentile(bot_valid, 25) +
            np.percentile(top_valid, 75) - np.percentile(top_valid, 25)
        ) / 2.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — RANSAC PLANE FITTING (secondary, good for top-down/angled)
# ─────────────────────────────────────────────────────────────────────────────

class PlaneEstimator(BaseEstimator, RegressorMixin):
    """Fits Z = aX + bY + c via least-squares. RANSAC wraps this."""
    def fit(self, X, y):
        ones = np.ones((X.shape[0], 1))
        A = np.hstack([X, ones])
        self.coef_, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        A = np.hstack([X, ones])
        return A @ self.coef_


def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SVD plane fit. Returns unit normal and centroid."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal) + 1e-9
    return normal, centroid


def ransac_plane_fit(
    points: np.ndarray,
    residual_threshold: float = RANSAC_RESIDUAL_THRESHOLD,
) -> Optional[Dict]:
    """RANSAC plane fit on a 3D point cloud. Returns None if fitting fails."""
    if len(points) < MIN_POINTS_FOR_FIT:
        return None

    X_2d = points[:, :2]
    z = points[:, 2]

    try:
        ransac = RANSACRegressor(
            estimator=PlaneEstimator(),
            min_samples=RANSAC_MIN_SAMPLES,
            residual_threshold=residual_threshold,
            max_trials=RANSAC_MAX_TRIALS,
            stop_probability=RANSAC_STOP_PROBABILITY,
            random_state=42,
        )
        ransac.fit(X_2d, z)
    except Exception as e:
        logger.error(f"RANSAC fitting failed: {e}")
        return None

    inlier_mask = ransac.inlier_mask_
    inlier_ratio = float(inlier_mask.sum()) / len(points)
    n_inliers = int(inlier_mask.sum())

    if n_inliers < MIN_POINTS_FOR_FIT:
        return None

    inlier_points = points[inlier_mask]
    normal, centroid = fit_plane_svd(inlier_points)
    d = float(np.dot(normal, centroid))
    plane_params = np.array([normal[0], normal[1], normal[2], -d], dtype=np.float64)
    residuals = np.abs(inlier_points @ normal - d)
    mean_residual = float(residuals.mean())

    logger.info(
        f"RANSAC: inlier_ratio={inlier_ratio:.2%}, "
        f"n={n_inliers}, residual={mean_residual*100:.2f}cm"
    )
    return {
        "plane_params": plane_params,
        "normal": normal,
        "centroid": centroid,
        "inlier_mask": inlier_mask,
        "inlier_ratio": inlier_ratio,
        "mean_residual": mean_residual,
        "n_inliers": n_inliers,
    }


def ransac_depth(rim_points, base_points) -> Optional[Dict]:
    """
    Try RANSAC plane fitting on rim and base point clouds.
    Returns depth result or None if either fit is poor.
    """
    rim_fit = ransac_plane_fit(rim_points)
    base_fit = ransac_plane_fit(base_points)

    if rim_fit is None or base_fit is None:
        return None

    rim_ir = rim_fit["inlier_ratio"]
    base_ir = base_fit["inlier_ratio"]

    # Reject if EITHER plane has very low inlier ratio
    if rim_ir < MIN_INLIER_RATIO_FOR_PLANE or base_ir < MIN_INLIER_RATIO_FOR_PLANE:
        logger.warning(
            f"RANSAC inlier ratios too low: rim={rim_ir:.2%}, base={base_ir:.2%} "
            f"(threshold={MIN_INLIER_RATIO_FOR_PLANE:.2%}) — discarding RANSAC result"
        )
        return None

    cos_angle = float(abs(np.dot(rim_fit["normal"], base_fit["normal"])))
    is_parallel = cos_angle >= PARALLELISM_COS_THRESHOLD

    avg_normal = (rim_fit["normal"] + base_fit["normal"]) / 2.0
    avg_normal /= np.linalg.norm(avg_normal) + 1e-9

    vec = base_fit["centroid"] - rim_fit["centroid"]
    depth_m = float(abs(np.dot(vec, avg_normal)))
    depth_cm = depth_m * 100.0

    mean_residual = (rim_fit["mean_residual"] + base_fit["mean_residual"]) / 2.0
    error_cm = mean_residual * ERROR_MARGIN_MULTIPLIER * 100.0

    logger.info(
        f"RANSAC depth: {depth_cm:.2f}cm ± {error_cm:.2f}cm "
        f"(parallel={is_parallel}, cos={cos_angle:.2f})"
    )

    return {
        "depth_m": round(depth_m, 4),
        "depth_cm": round(depth_cm, 2),
        "error_cm": round(error_cm, 2),
        "rim_fit": rim_fit,
        "base_fit": base_fit,
        "parallelism_cos": cos_angle,
        "is_parallel": is_parallel,
    }


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3 — VISUAL HEIGHT FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def visual_height_estimate(
    sam_mask: np.ndarray,
    depth_map: np.ndarray,
    fy: float,
    interior_fraction: float = 0.80,
) -> Optional[float]:
    """
    Estimate depth from apparent pixel height of container × metric depth ÷ focal_length.
      real_height = pixel_span × depth_to_object / fy

    interior_fraction: scale factor to convert exterior height → interior depth
    (0.80 accounts for caps, bases, walls — conservative estimate)

    Returns depth_m or None.
    """
    rows = np.where(sam_mask.max(axis=1) > 0)[0]
    if len(rows) < 4:
        return None

    pixel_span = float(rows[-1] - rows[0])
    if pixel_span < 2:
        return None

    mask_depths = depth_map[sam_mask > 0]
    valid = mask_depths[(mask_depths > 0.01) & (mask_depths < 30.0)]
    if len(valid) < 10:
        return None

    # Use 25th percentile depth = front surface of container
    depth_to_obj = float(np.percentile(valid, 25))
    real_height_m = (pixel_span * depth_to_obj / fy) * interior_fraction

    logger.info(
        f"Visual height: pixel_span={pixel_span:.0f}px, "
        f"depth={depth_to_obj:.3f}m, fy={fy:.0f}, "
        f"result={real_height_m*100:.1f}cm"
    )
    return real_height_m


# ─────────────────────────────────────────────────────────────────────────────
# MASTER ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_geometry_engine(
    rim_points: np.ndarray,
    base_points: np.ndarray,
    sam_mask: Optional[np.ndarray] = None,
    depth_map: Optional[np.ndarray] = None,
    intrinsics: Optional[Dict] = None,
) -> Dict:
    """
    Full geometric depth computation. Entry point for Module 6.

    Method hierarchy:
      1. Strip-Delta  — most robust, always attempted
      2. RANSAC       — trusted only when inlier ratios ≥ MIN_INLIER_RATIO_FOR_PLANE
      3. Visual-Size  — fallback when both above are outside plausible range

    Final depth = weighted combination of methods that fall in [0.5cm, 100cm].

    Args:
        rim_points  : (N, 3) rim 3D point cloud
        base_points : (M, 3) base 3D point cloud
        sam_mask    : (H, W) uint8 binary mask of full container
        depth_map   : (H, W) float32 metric depth map in meters
        intrinsics  : camera intrinsics dict (fx, fy, cx, cy)

    Returns:
        dict with depth_cm, depth_m, error_margin_cm, rim_plane, base_plane,
                    parallelism_cos, is_parallel, warnings, success, method_used
    """
    warnings_list = []
    result = {
        "depth_cm": None,
        "depth_m": None,
        "error_margin_cm": None,
        "rim_plane": None,
        "base_plane": None,
        "parallelism_cos": 0.0,
        "is_parallel": False,
        "warnings": warnings_list,
        "success": False,
        "method_used": "none",
    }

    PLAUSIBLE_MIN = DEPTH_SANITY_MIN_CM
    PLAUSIBLE_MAX = DEPTH_SANITY_MAX_CM

    def plausible(depth_cm):
        return PLAUSIBLE_MIN <= depth_cm <= PLAUSIBLE_MAX

    candidates = []  # list of (depth_m, error_m, weight, method_name)

    # ── METHOD 1: Strip-delta ─────────────────────────────────────────────────
    strip_result = None
    if sam_mask is not None and depth_map is not None:
        strip_result = strip_delta_depth(sam_mask, depth_map)
        if strip_result is not None:
            sd = strip_result["depth_cm"]
            if plausible(sd):
                # Weight: based on pixel count and depth consistency
                n_pixels = strip_result["n_top"] + strip_result["n_bot"]
                pixel_weight = min(1.0, n_pixels / 200)
                candidates.append((
                    strip_result["depth_m"],
                    strip_result["error_m"],
                    0.7 * pixel_weight,   # strip-delta is our most reliable
                    "strip_delta",
                ))
                logger.info(f"Strip-delta depth: {sd:.1f}cm ✓")
            else:
                logger.warning(f"Strip-delta depth {sd:.1f}cm outside plausible range")
                warnings_list.append(
                    f"Strip-delta depth {sd:.1f}cm is outside expected range "
                    f"({PLAUSIBLE_MIN}–{PLAUSIBLE_MAX}cm) — depth model may have scaling issues."
                )

    # ── METHOD 2: RANSAC planes ───────────────────────────────────────────────
    ransac_res = None
    try:
        ransac_res = ransac_depth(rim_points, base_points)
    except Exception as e:
        logger.warning(f"RANSAC failed entirely: {e}")

    if ransac_res is not None:
        result["rim_plane"] = ransac_res["rim_fit"]
        result["base_plane"] = ransac_res["base_fit"]
        result["parallelism_cos"] = ransac_res["parallelism_cos"]
        result["is_parallel"] = ransac_res["is_parallel"]

        rd = ransac_res["depth_cm"]
        if plausible(rd):
            # Trust RANSAC more when both planes have high inlier ratios
            rim_ir = ransac_res["rim_fit"]["inlier_ratio"]
            base_ir = ransac_res["base_fit"]["inlier_ratio"]
            par_boost = ransac_res["parallelism_cos"]
            ransac_weight = 0.5 * rim_ir * base_ir * par_boost
            candidates.append((
                ransac_res["depth_m"],
                ransac_res["error_cm"] / 100.0,
                ransac_weight,
                "ransac",
            ))
            logger.info(f"RANSAC depth: {rd:.1f}cm ✓ (weight={ransac_weight:.2f})")
        else:
            logger.warning(f"RANSAC depth {rd:.1f}cm outside plausible range — ignoring")
            warnings_list.append(
                f"RANSAC plane geometry gave {rd:.1f}cm — outside plausible range. "
                "Likely caused by poor plane normal estimation (low inlier ratios)."
            )
    else:
        # Provide stub metadata for UI
        warnings_list.append(
            "RANSAC plane fitting skipped or failed (low inlier ratios). "
            "Using direct depth-delta estimation instead."
        )
        # Populate stub plane data so UI doesn't crash
        result["rim_plane"] = {
            "inlier_ratio": 0.0, "n_inliers": 0,
            "mean_residual": 0.0, "plane_params": np.zeros(4),
            "normal": np.array([0., 0., 1.]), "centroid": np.zeros(3),
            "inlier_mask": np.zeros(max(len(rim_points), 1), dtype=bool),
        }
        result["base_plane"] = {
            "inlier_ratio": 0.0, "n_inliers": 0,
            "mean_residual": 0.0, "plane_params": np.zeros(4),
            "normal": np.array([0., 0., 1.]), "centroid": np.zeros(3),
            "inlier_mask": np.zeros(max(len(base_points), 1), dtype=bool),
        }

    # ── METHOD 3: Visual height fallback ─────────────────────────────────────
    if sam_mask is not None and depth_map is not None and intrinsics is not None:
        fy = intrinsics.get("fy", depth_map.shape[1])
        vh = visual_height_estimate(sam_mask, depth_map, fy)
        if vh is not None:
            vh_cm = vh * 100.0
            if plausible(vh_cm):
                # Low weight — relies on focal length assumption
                candidates.append((vh, 0.03, 0.20, "visual_height"))
                logger.info(f"Visual height estimate: {vh_cm:.1f}cm")

    # ── SELECTION ─────────────────────────────────────────────────────────────
    if not candidates:
        # Complete failure — nothing is in plausible range
        # Last resort: use strip_result raw even if out of range
        if strip_result is not None:
            sd = strip_result["depth_cm"]
            logger.error(f"All methods out of range — using raw strip-delta: {sd:.1f}cm")
            warnings_list.append(
                f"⚠️ All depth methods returned values outside {PLAUSIBLE_MIN}–{PLAUSIBLE_MAX}cm. "
                f"Raw strip-delta: {sd:.1f}cm. Depth model may need recalibration."
            )
            result.update({
                "depth_cm": round(sd, 2),
                "depth_m": round(strip_result["depth_m"], 4),
                "error_margin_cm": round(strip_result["error_m"] * 100, 2),
                "warnings": warnings_list,
                "success": True,
                "method_used": "strip_delta_raw",
            })
            return result

        warnings_list.append(
            "Depth estimation completely failed. "
            "Ensure the container is clearly visible and the image is in focus."
        )
        result["warnings"] = warnings_list
        return result

    # Weighted combination of plausible candidates
    total_weight = sum(w for _, _, w, _ in candidates)
    depth_m_combined = sum(d * w for d, _, w, _ in candidates) / total_weight
    error_m_combined = sum(e * w for _, e, w, _ in candidates) / total_weight
    methods_used = "+".join(sorted(set(m for _, _, _, m in candidates)))

    depth_cm_combined = depth_m_combined * 100.0
    error_cm_combined = error_m_combined * 100.0

    # Sanity clamp
    depth_cm_combined = float(np.clip(depth_cm_combined, PLAUSIBLE_MIN, PLAUSIBLE_MAX))

    logger.info(
        f"FINAL DEPTH: {depth_cm_combined:.2f}cm ± {error_cm_combined:.2f}cm "
        f"[methods: {methods_used}]"
    )

    result.update({
        "depth_cm": round(depth_cm_combined, 2),
        "depth_m": round(depth_m_combined, 4),
        "error_margin_cm": round(max(error_cm_combined, 0.5), 2),
        "warnings": warnings_list,
        "success": True,
        "method_used": methods_used,
    })
    return result
