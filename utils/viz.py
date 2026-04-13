"""
utils/viz.py — Module 8
All visualizations: annotated image, depth heatmap, 3D point cloud,
confidence bar chart, and result card. All Streamlit-ready (Plotly + OpenCV).
"""

import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import logging

from config import (
    VIZ_RIM_COLOR,
    VIZ_BASE_COLOR,
    VIZ_ALPHA,
    VIZ_PLANE_OPACITY,
    ERROR_MARGIN_MULTIPLIER,
)
from utils.confidence import get_confidence_color, get_confidence_emoji

logger = logging.getLogger(__name__)


# ─── Color helpers ────────────────────────────────────────────────────────────

def _bgr_to_rgb_tuple(bgr: Tuple) -> Tuple:
    return (bgr[2], bgr[1], bgr[0])


# ─── VIZ 1: Annotated Image ───────────────────────────────────────────────────

def draw_mask_overlay(
    image_bgr: np.ndarray,
    rim_mask: np.ndarray,
    interior_mask: np.ndarray,
) -> np.ndarray:
    """
    Overlay rim (blue) and interior/base (red) masks on the image.

    Args:
        image_bgr     : Original BGR image (H×W×3, uint8)
        rim_mask      : Binary rim mask (H×W, uint8)
        interior_mask : Binary interior (base) mask (H×W, uint8)

    Returns:
        annotated_rgb : RGB image with colored overlays
    """
    overlay = image_bgr.copy()

    # Rim → Blue
    rim_colored = np.zeros_like(overlay)
    rim_colored[rim_mask > 0] = VIZ_RIM_COLOR    # BGR blue

    # Interior → Red
    base_colored = np.zeros_like(overlay)
    base_colored[interior_mask > 0] = VIZ_BASE_COLOR  # BGR red

    cv2.addWeighted(rim_colored, VIZ_ALPHA, overlay, 1.0, 0, overlay)
    cv2.addWeighted(base_colored, VIZ_ALPHA, overlay, 1.0, 0, overlay)

    # Draw contours for crisp edges
    rim_contours, _ = cv2.findContours(rim_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    base_contours, _ = cv2.findContours(interior_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(overlay, rim_contours, -1, VIZ_RIM_COLOR, 2)
    cv2.drawContours(overlay, base_contours, -1, VIZ_BASE_COLOR, 2)

    # Legend
    h = overlay.shape[0]
    cv2.rectangle(overlay, (10, h - 60), (200, h - 10), (30, 30, 30), -1)
    cv2.putText(overlay, "RIM", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, VIZ_RIM_COLOR, 2)
    cv2.putText(overlay, "BASE", (100, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, VIZ_BASE_COLOR, 2)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ─── VIZ 2: Depth Heatmap ────────────────────────────────────────────────────

def create_depth_heatmap(
    depth_map: np.ndarray,
    sam_mask: np.ndarray,
    image_rgb: np.ndarray,
) -> go.Figure:
    """
    Create an interactive Plotly depth heatmap with container region highlighted.

    Args:
        depth_map  : Metric depth map (H×W, float32, meters)
        sam_mask   : Full SAM binary mask for container outline
        image_rgb  : Original RGB image for background reference

    Returns:
        fig : Plotly figure (heatmap)
    """
    # Normalize depth to [0, 1] for display
    d_min, d_max = depth_map.min(), depth_map.max()
    depth_norm = (depth_map - d_min) / (d_max - d_min + 1e-6)

    # Masked depth (only inside container)
    masked_depth = np.where(sam_mask > 0, depth_norm, np.nan)

    h, w = depth_map.shape
    fig = go.Figure()

    # Background: full image depth (muted)
    fig.add_trace(go.Heatmap(
        z=depth_norm,
        colorscale="Greys",
        showscale=False,
        opacity=0.4,
        name="Full depth",
    ))

    # Foreground: container depth (vivid)
    fig.add_trace(go.Heatmap(
        z=masked_depth,
        colorscale="Inferno",
        showscale=True,
        colorbar=dict(
            title=dict(text="Depth (normalized)", side="right"),
            tickvals=[0, 0.5, 1.0],
            ticktext=[f"{d_min:.2f}m", f"{(d_min+d_max)/2:.2f}m", f"{d_max:.2f}m"],
        ),
        name="Container depth",
    ))

    fig.update_layout(
        title="Depth Map — Container Region Highlighted",
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        yaxis=dict(autorange="reversed", showticklabels=False),
        xaxis=dict(showticklabels=False),
    )

    return fig


# ─── VIZ 3: 3D Point Cloud ───────────────────────────────────────────────────

def _plane_mesh(
    plane_params: np.ndarray,
    center: np.ndarray,
    size: float = 0.15,
    n: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a planar mesh grid for a RANSAC plane, centered at `center`."""
    a, b, c, d = plane_params

    # Build two in-plane tangent vectors
    normal = np.array([a, b, c], dtype=float)
    normal /= np.linalg.norm(normal) + 1e-9

    # Find a non-collinear vector
    aux = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, aux)) > 0.9:
        aux = np.array([0.0, 1.0, 0.0])

    t1 = np.cross(normal, aux)
    t1 /= np.linalg.norm(t1) + 1e-9
    t2 = np.cross(normal, t1)
    t2 /= np.linalg.norm(t2) + 1e-9

    lin = np.linspace(-size, size, n)
    s1, s2 = np.meshgrid(lin, lin)

    X = center[0] + s1 * t1[0] + s2 * t2[0]
    Y = center[1] + s1 * t1[1] + s2 * t2[1]
    Z = center[2] + s1 * t1[2] + s2 * t2[2]

    return X, Y, Z


def create_3d_point_cloud(
    rim_points: np.ndarray,
    base_points: np.ndarray,
    rim_plane: Optional[np.ndarray] = None,
    base_plane: Optional[np.ndarray] = None,
    depth_cm: Optional[float] = None,
) -> go.Figure:
    """
    Create an interactive 3D Plotly point cloud visualization.

    Shows rim points (blue), base points (red), and fitted planes (mesh).
    This is the primary debug tool — wrong plane fits are instantly visible.

    Args:
        rim_points  : (N, 3) rim 3D point cloud
        base_points : (M, 3) base 3D point cloud
        rim_plane   : (4,) plane params [a, b, c, d] or None
        base_plane  : (4,) plane params or None
        depth_cm    : Estimated depth in cm (for annotation)

    Returns:
        fig : Plotly 3D scatter figure
    """
    fig = go.Figure()

    subsample = lambda pts, n=3000: pts[np.random.choice(len(pts), min(n, len(pts)), replace=False)]

    # Rim points — blue
    if len(rim_points) > 0:
        rp = subsample(rim_points)
        fig.add_trace(go.Scatter3d(
            x=rp[:, 0], y=rp[:, 1], z=rp[:, 2],
            mode="markers",
            marker=dict(size=2, color="#3b82f6", opacity=0.7),
            name="Rim points",
        ))

    # Base points — red/orange
    if len(base_points) > 0:
        bp = subsample(base_points)
        fig.add_trace(go.Scatter3d(
            x=bp[:, 0], y=bp[:, 1], z=bp[:, 2],
            mode="markers",
            marker=dict(size=2, color="#ef4444", opacity=0.7),
            name="Base points",
        ))

    # Rim plane — semi-transparent blue mesh
    if rim_plane is not None and len(rim_points) > 0:
        span = float(np.ptp(rim_points, axis=0).max()) * 0.7
        cx = rim_points.mean(axis=0)
        X, Y, Z = _plane_mesh(rim_plane, cx, size=span / 2)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, "rgba(59,130,246,0.2)"], [1, "rgba(59,130,246,0.2)"]],
            showscale=False,
            name="Rim plane",
            opacity=VIZ_PLANE_OPACITY,
        ))

    # Base plane — semi-transparent red mesh
    if base_plane is not None and len(base_points) > 0:
        span = float(np.ptp(base_points, axis=0).max()) * 0.7
        cx = base_points.mean(axis=0)
        X, Y, Z = _plane_mesh(base_plane, cx, size=span / 2)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, "rgba(239,68,68,0.2)"], [1, "rgba(239,68,68,0.2)"]],
            showscale=False,
            name="Base plane",
            opacity=VIZ_PLANE_OPACITY,
        ))

    title = "3D Point Cloud — Rim (blue) | Base (red)"
    if depth_cm is not None:
        title += f" | Depth ≈ {depth_cm:.1f} cm"

    fig.update_layout(
        title=title,
        height=550,
        paper_bgcolor="#0f172a",
        scene=dict(
            bgcolor="#0f172a",
            xaxis=dict(
                title="X (m)", backgroundcolor="#1e293b",
                gridcolor="#334155", color="#94a3b8",
            ),
            yaxis=dict(
                title="Y (m)", backgroundcolor="#1e293b",
                gridcolor="#334155", color="#94a3b8",
            ),
            zaxis=dict(
                title="Z depth (m)", backgroundcolor="#1e293b",
                gridcolor="#334155", color="#94a3b8",
            ),
        ),
        legend=dict(
            font=dict(color="#e2e8f0"),
            bgcolor="rgba(15,23,42,0.8)",
        ),
        font=dict(color="#e2e8f0"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


# ─── VIZ 4: Confidence Breakdown ─────────────────────────────────────────────

def create_confidence_chart(confidence_result: Dict) -> go.Figure:
    """
    Create a horizontal bar chart showing per-component confidence scores.

    Args:
        confidence_result : Dict from confidence.compute_confidence()

    Returns:
        fig : Plotly bar chart
    """
    scores = confidence_result["scores"]
    weights = confidence_result["weights"]
    label = confidence_result["confidence_label"]
    raw = confidence_result["confidence_raw"]

    components = list(scores.keys())
    score_vals = [scores[c] for c in components]
    weight_vals = [weights[c] for c in components]
    weighted_contributions = [s * w for s, w in zip(score_vals, weight_vals)]

    display_names = {
        "plane_fit": "Plane Fit Quality",
        "point_count": "Point Count",
        "bottom_visibility": "Bottom Visibility",
        "parallelism": "Plane Parallelism",
    }

    colors = [
        f"rgba(34,197,94,0.85)" if v >= 0.75
        else f"rgba(245,158,11,0.85)" if v >= 0.50
        else f"rgba(239,68,68,0.85)"
        for v in score_vals
    ]

    fig = go.Figure()

    # Raw scores
    fig.add_trace(go.Bar(
        x=score_vals,
        y=[display_names[c] for c in components],
        orientation="h",
        name="Component Score",
        marker_color=colors,
        text=[f"{v:.0%}" for v in score_vals],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=12),
    ))

    # Weight annotation
    for i, (c, w) in enumerate(zip(components, weight_vals)):
        fig.add_annotation(
            x=1.02, y=display_names[c],
            text=f"w={w:.0%}",
            showarrow=False,
            font=dict(color="#94a3b8", size=11),
            xref="x", yref="y",
        )

    conf_color = get_confidence_color(label)
    emoji = get_confidence_emoji(label)

    fig.update_layout(
        title=f"Confidence Breakdown — {emoji} {label} ({raw:.0%} overall)",
        title_font=dict(color=conf_color, size=15),
        xaxis=dict(
            range=[0, 1.15],
            tickformat=".0%",
            color="#94a3b8",
            gridcolor="#334155",
        ),
        yaxis=dict(color="#e2e8f0"),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        height=300,
        margin=dict(l=0, r=60, t=50, b=30),
        showlegend=False,
    )

    return fig


# ─── Result Card data ────────────────────────────────────────────────────────

def build_result_dict(
    depth_cm: float,
    error_margin_cm: float,
    confidence_result: Dict,
    warnings: List[str],
    fit_metadata: Dict,
) -> Dict:
    """
    Build the final structured result dictionary for display.

    Args:
        depth_cm          : Estimated depth in centimeters
        error_margin_cm   : ± error margin in cm
        confidence_result : From confidence.compute_confidence()
        warnings          : Aggregated warning strings
        fit_metadata      : From geometry_engine

    Returns:
        Structured result dict for Streamlit display
    """
    label = confidence_result["confidence_label"]
    return {
        "depth_cm": round(depth_cm, 1),
        "depth_m": round(depth_cm / 100.0, 3),
        "error_margin_cm": round(error_margin_cm, 1),
        "confidence_label": label,
        "confidence_raw": round(confidence_result["confidence_raw"], 3),
        "confidence_emoji": get_confidence_emoji(label),
        "confidence_color": get_confidence_color(label),
        "warnings": warnings,
        "fit_metadata": fit_metadata,
        "component_scores": confidence_result["scores"],
    }
