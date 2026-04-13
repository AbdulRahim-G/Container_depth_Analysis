"""
utils/confidence.py — Module 7
Weighted confidence scoring system combining plane fit quality,
point count adequacy, bottom visibility, and plane parallelism.
"""

import numpy as np
from typing import Dict, Tuple
import logging

from config import (
    CONFIDENCE_WEIGHTS,
    CONFIDENCE_POINT_TARGET,
    CONFIDENCE_INLIER_THRESHOLD,
    CONFIDENCE_THRESHOLDS,
)

logger = logging.getLogger(__name__)


def score_plane_fit(
    rim_inlier_ratio: float,
    base_inlier_ratio: float,
) -> float:
    """
    Score based on RANSAC inlier ratios for both planes.
    Both planes need high inlier ratios for a reliable fit.

    Score = rim_inlier_ratio * base_inlier_ratio
    Normalized so that (0.85 * 0.85) ≈ 0.72 gives ~0.72 raw score.

    Args:
        rim_inlier_ratio  : Fraction of rim points that are RANSAC inliers [0,1]
        base_inlier_ratio : Fraction of base points that are RANSAC inliers [0,1]

    Returns:
        score : float [0, 1]
    """
    score = rim_inlier_ratio * base_inlier_ratio
    logger.debug(f"Plane fit score: {score:.3f} (rim={rim_inlier_ratio:.2f}, base={base_inlier_ratio:.2f})")
    return float(np.clip(score, 0.0, 1.0))


def score_point_count(
    n_rim_points: int,
    n_base_points: int,
    target: int = CONFIDENCE_POINT_TARGET,
) -> float:
    """
    Score based on how many points are available for plane fitting.
    Uses the bottleneck (minimum of rim/base counts).

    Args:
        n_rim_points  : Number of rim 3D points after outlier removal
        n_base_points : Number of base 3D points after outlier removal
        target        : Target point count considered "enough" (default 500)

    Returns:
        score : float [0, 1]
    """
    bottleneck = min(n_rim_points, n_base_points)
    score = float(np.clip(bottleneck / target, 0.0, 1.0))
    logger.debug(f"Point count score: {score:.3f} (rim={n_rim_points}, base={n_base_points}, target={target})")
    return score


def score_bottom_visibility(visibility_ratio: float) -> float:
    """
    Score based on what fraction of the base region has valid depth.
    If the bottom is barely visible, we can't trust the base plane.

    Args:
        visibility_ratio : Fraction of base pixels with valid depth [0, 1]

    Returns:
        score : float [0, 1]
    """
    logger.debug(f"Visibility score: {visibility_ratio:.3f}")
    return float(np.clip(visibility_ratio, 0.0, 1.0))


def score_parallelism(cos_angle: float) -> float:
    """
    Score based on cosine similarity between rim and base plane normals.
    For a real container, rim and base MUST be approximately parallel.
    cos_angle close to 1.0 → perfectly parallel → high score.
    cos_angle < 0 → planes are diverging → nonsensical fit.

    Args:
        cos_angle : Absolute cosine of angle between plane normals [0, 1]

    Returns:
        score : float [0, 1]
    """
    score = float(np.clip(abs(cos_angle), 0.0, 1.0))
    logger.debug(f"Parallelism score: {score:.3f} (cos={cos_angle:.3f})")
    return score


def compute_confidence(
    rim_inlier_ratio: float,
    base_inlier_ratio: float,
    n_rim_points: int,
    n_base_points: int,
    visibility_ratio: float,
    parallelism_cos: float,
) -> Dict:
    """
    Compute the final weighted confidence score. Entry point for Module 7.

    Args:
        rim_inlier_ratio  : RANSAC rim inlier ratio
        base_inlier_ratio : RANSAC base inlier ratio
        n_rim_points      : Number of rim 3D points (after filtering)
        n_base_points     : Number of base 3D points (after filtering)
        visibility_ratio  : Bottom visibility fraction
        parallelism_cos   : Cosine between plane normals

    Returns:
        dict with:
            scores          : per-component scores dict
            weights         : component weights dict
            confidence_raw  : float [0, 1]
            confidence_label: "HIGH" | "MEDIUM" | "LOW"
    """
    scores = {
        "plane_fit": score_plane_fit(rim_inlier_ratio, base_inlier_ratio),
        "point_count": score_point_count(n_rim_points, n_base_points),
        "bottom_visibility": score_bottom_visibility(visibility_ratio),
        "parallelism": score_parallelism(parallelism_cos),
    }

    weights = CONFIDENCE_WEIGHTS
    confidence_raw = sum(scores[k] * weights[k] for k in scores)
    confidence_raw = float(np.clip(confidence_raw, 0.0, 1.0))

    if confidence_raw >= CONFIDENCE_THRESHOLDS["HIGH"]:
        confidence_label = "HIGH"
    elif confidence_raw >= CONFIDENCE_THRESHOLDS["MEDIUM"]:
        confidence_label = "MEDIUM"
    else:
        confidence_label = "LOW"

    logger.info(
        f"Confidence: {confidence_raw:.3f} → {confidence_label} "
        f"| scores={scores}"
    )

    return {
        "scores": scores,
        "weights": weights,
        "confidence_raw": confidence_raw,
        "confidence_label": confidence_label,
    }


def get_confidence_emoji(label: str) -> str:
    """Return emoji indicator for confidence label."""
    return {"HIGH": "✅", "MEDIUM": "⚠️", "LOW": "❌"}.get(label, "❓")


def get_confidence_color(label: str) -> str:
    """Return hex color for confidence label (for Streamlit display)."""
    return {
        "HIGH": "#22c55e",    # green
        "MEDIUM": "#f59e0b",  # amber
        "LOW": "#ef4444",     # red
    }.get(label, "#94a3b8")
