"""
app.py — Streamlit UI for Container Depth Estimation
Pure UI layer — all logic lives in models/ and utils/

No business logic here. This file only orchestrates calls
and presents results beautifully.
"""

import streamlit as st
import numpy as np
import cv2
import logging
import traceback
from PIL import Image as PILImage

# ── Configure logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Container Depth Estimator",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inline CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f172a;
    color: #e2e8f0;
  }

  /* Gradient header */
  .hero-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 40%, #1a1035 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(
      circle at 60% 40%,
      rgba(99, 102, 241, 0.08) 0%,
      transparent 60%
    );
    pointer-events: none;
  }
  .hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
  }
  .hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 0.4rem;
    font-weight: 400;
  }

  /* Result card */
  .result-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
  }
  .result-depth {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
  }
  .result-unit {
    font-size: 1.2rem;
    color: #94a3b8;
    font-weight: 500;
    margin-top: 0.2rem;
  }
  .result-error {
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 0.3rem;
  }
  .confidence-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 99px;
    font-weight: 700;
    font-size: 1rem;
    margin-top: 0.8rem;
  }

  /* Warning box */
  .warning-box {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    color: #fbbf24;
  }

  /* Section label */
  .section-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
  }

  /* Pipeline step indicator */
  .step-badge {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 8px;
    padding: 0.25rem 0.6rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #818cf8;
    display: inline-block;
    margin-bottom: 0.5rem;
  }

  /* Tabs styling */
  .stTabs [data-baseweb="tab-list"] {
    background: #1e293b;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #94a3b8;
    padding: 0.5rem 1.2rem;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.2) !important;
    color: #818cf8 !important;
  }

  /* Upload zone */
  [data-testid="stFileUploaderDropzone"] {
    background: rgba(30, 41, 59, 0.8);
    border: 2px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: 12px;
  }

  /* Metric cards */
  .metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }
  .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
  }
  .metric-label {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 500;
    margin-top: 0.2rem;
  }

  /* Streamlit default overrides */
  .stButton>button {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.8rem;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    transition: all 0.2s ease;
  }
  .stButton>button:hover {
    background: linear-gradient(135deg, #6366f1, #818cf8);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.4);
  }
  div[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, #4f46e5, #38bdf8) !important;
  }

  /* Fill level card */
  .fill-card {
    background: linear-gradient(135deg, #0f2027, #1e293b);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin: 1rem 0;
  }
  .fill-card-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 1rem;
  }
  .fill-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.55rem;
  }
  .fill-label {
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 500;
    min-width: 120px;
  }
  .fill-bar-track {
    flex: 1;
    height: 10px;
    background: #1e293b;
    border-radius: 99px;
    margin: 0 0.8rem;
    overflow: hidden;
  }
  .fill-bar-inner {
    height: 100%;
    border-radius: 99px;
  }
  .fill-value {
    font-size: 0.95rem;
    font-weight: 700;
    color: #e2e8f0;
    min-width: 60px;
    text-align: right;
  }
  .fill-pct-badge {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 99px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-top: 0.8rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Model caching (lazy load) ─────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_detector():
    from models.detector import ContainerDetector
    return ContainerDetector()


@st.cache_resource(show_spinner=False)
def get_depth_estimator():
    from models.depth import MetricDepthEstimator
    return MetricDepthEstimator()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(image_bgr: np.ndarray, progress, status):
    """
    Orchestrate the full 8-module pipeline.
    Returns result dict + all intermediate artifacts.
    """
    from utils.preprocessing import preprocess_image, bgr_to_rgb
    from utils.mask_surgery import run_mask_surgery, detect_fill_level
    from utils.geometry import lift_masks_to_3d
    from utils.confidence import compute_confidence
    from utils.viz import draw_mask_overlay, create_depth_heatmap, create_3d_point_cloud, create_confidence_chart, build_result_dict
    from models.geometry_engine import run_geometry_engine

    artifacts = {}

    # ── M1: Preprocessing ─────────────────────────────────────────────────
    status.markdown('<div class="step-badge">⚙️ Step 1 / 7 — Preprocessing</div>', unsafe_allow_html=True)
    progress.progress(0.05)
    processed_bgr, original_bgr, scale = preprocess_image(image_bgr)
    image_rgb = bgr_to_rgb(original_bgr)
    processed_rgb = bgr_to_rgb(processed_bgr)
    h, w = image_rgb.shape[:2]
    artifacts["image_rgb"] = image_rgb
    artifacts["processed_rgb"] = processed_rgb

    # ── M2: Detection ─────────────────────────────────────────────────────
    status.markdown('<div class="step-badge">🔍 Step 2 / 7 — Detecting Container</div>', unsafe_allow_html=True)
    progress.progress(0.15)
    detector = get_detector()
    sam_mask, detect_meta = detector.detect_and_segment(processed_rgb)
    if sam_mask is None or sam_mask.max() == 0:
        raise RuntimeError(
            "No container detected in the image. "
            "Ensure a bowl, cup, bucket, or similar container is clearly visible."
        )
    artifacts["sam_mask"] = sam_mask
    artifacts["detect_meta"] = detect_meta
    logger.info(f"Detection strategy: {detect_meta['strategy']}")

    # ── M4: Metric Depth (before mask surgery, on full image) ─────────────
    status.markdown('<div class="step-badge">🌊 Step 3 / 7 — Estimating Metric Depth</div>', unsafe_allow_html=True)
    progress.progress(0.35)
    depth_estimator = get_depth_estimator()
    depth_map = depth_estimator.estimate_depth(image_rgb)
    artifacts["depth_map"] = depth_map

    # ── M3: Mask Surgery (needs depth for visibility check) ───────────────
    status.markdown('<div class="step-badge">✂️ Step 4 / 7 — Mask Surgery</div>', unsafe_allow_html=True)
    progress.progress(0.50)
    surgery = run_mask_surgery(sam_mask, depth_map)
    rim_mask = surgery["rim_mask"]
    interior_mask = surgery["interior_mask"]
    artifacts["rim_mask"] = rim_mask
    artifacts["interior_mask"] = interior_mask
    artifacts["surgery"] = surgery

    # Apply masks to depth
    rim_depth, base_depth, depth_sane = depth_estimator.apply_masks(depth_map, rim_mask, interior_mask)
    if not depth_sane:
        surgery["warnings"].append(
            "⚠️ Depth sanity failed: base appears closer than rim. "
            "Container may be held upside-down or heavily tilted."
        )
    artifacts["rim_depth"] = rim_depth
    artifacts["base_depth"] = base_depth

    # ── M5: 2D → 3D Lift ─────────────────────────────────────────────────
    status.markdown('<div class="step-badge">🗺️ Step 5 / 7 — Lifting to 3D</div>', unsafe_allow_html=True)
    progress.progress(0.65)
    lift = lift_masks_to_3d(rim_mask, interior_mask, depth_map, h, w)
    rim_pts = lift["rim_points"]
    base_pts = lift["base_points"]
    artifacts["lift"] = lift

    # ── M6: Geometry Engine ───────────────────────────────────────────────
    status.markdown('<div class="step-badge">📐 Step 6 / 7 — RANSAC Geometry</div>', unsafe_allow_html=True)
    progress.progress(0.80)
    geo = run_geometry_engine(
        rim_pts,
        base_pts,
        sam_mask=sam_mask,
        depth_map=depth_map,
        intrinsics=lift["intrinsics"],
    )
    artifacts["geo"] = geo

    if not geo["success"]:
        raise RuntimeError(
            "Geometric depth calculation failed. " +
            " | ".join(geo["warnings"]) or "Unknown error in plane fitting."
        )

    # ── Fill level detection ───────────────────────────────────────────────
    status.markdown('<div class="step-badge">🧪 Step 7 / 8 — Fill Level Detection</div>', unsafe_allow_html=True)
    progress.progress(0.88)
    fill_result = detect_fill_level(
        interior_mask=interior_mask,
        image_bgr=original_bgr,
        full_depth_cm=geo["depth_cm"],
    )
    artifacts["fill_result"] = fill_result
    logger.info(f"Fill: filled={fill_result['filled_depth_cm']} cm, empty={fill_result['empty_depth_cm']} cm")

    # ── M7: Confidence ────────────────────────────────────────────────────
    status.markdown('<div class="step-badge">📊 Step 8 / 8 — Scoring Confidence</div>', unsafe_allow_html=True)
    progress.progress(0.94)
    confidence = compute_confidence(
        rim_inlier_ratio=geo["rim_plane"]["inlier_ratio"],
        base_inlier_ratio=geo["base_plane"]["inlier_ratio"],
        n_rim_points=len(rim_pts),
        n_base_points=len(base_pts),
        visibility_ratio=surgery["visibility_ratio"],
        parallelism_cos=geo["parallelism_cos"],
    )

    # ── M8: Visualizations ────────────────────────────────────────────────
    all_warnings = surgery["warnings"] + geo["warnings"]

    annotated_rgb = draw_mask_overlay(original_bgr, rim_mask, interior_mask)
    artifacts["annotated_rgb"] = annotated_rgb

    depth_heatmap_fig = create_depth_heatmap(depth_map, sam_mask, image_rgb)
    artifacts["depth_heatmap"] = depth_heatmap_fig

    rim_plane_params = geo["rim_plane"]["plane_params"] if geo["rim_plane"] else None
    base_plane_params = geo["base_plane"]["plane_params"] if geo["base_plane"] else None

    cloud_fig = create_3d_point_cloud(
        rim_pts, base_pts,
        rim_plane=rim_plane_params,
        base_plane=base_plane_params,
        depth_cm=geo["depth_cm"],
    )
    artifacts["cloud_fig"] = cloud_fig

    conf_fig = create_confidence_chart(confidence)
    artifacts["conf_fig"] = conf_fig

    result = build_result_dict(
        depth_cm=geo["depth_cm"],
        error_margin_cm=geo["error_margin_cm"],
        confidence_result=confidence,
        warnings=all_warnings,
        fit_metadata={
            "rim_inlier_ratio": geo["rim_plane"]["inlier_ratio"] if geo["rim_plane"] else 0.0,
            "base_inlier_ratio": geo["base_plane"]["inlier_ratio"] if geo["base_plane"] else 0.0,
            "rim_n_inliers": geo["rim_plane"]["n_inliers"] if geo["rim_plane"] else 0,
            "base_n_inliers": geo["base_plane"]["n_inliers"] if geo["base_plane"] else 0,
            "parallelism_cos": geo["parallelism_cos"],
            "strategy": detect_meta["strategy"],
            "scale_factor": scale,
            "method_used": geo.get("method_used", "—"),
        },
    )
    artifacts["result"] = result

    progress.progress(1.0)
    status.markdown("")
    return result, artifacts


# ── Render result ─────────────────────────────────────────────────────────────

def render_result(result: dict, artifacts: dict):
    """Render the full result dashboard."""
    from utils.confidence import get_confidence_color, get_confidence_emoji

    # ── Result Card ───────────────────────────────────────────────────────
    label = result["confidence_label"]
    color = result["confidence_color"]
    emoji = result["confidence_emoji"]
    depth = result["depth_cm"]
    err = result["error_margin_cm"]

    badge_bg = {
        "HIGH": "rgba(34,197,94,0.15)",
        "MEDIUM": "rgba(245,158,11,0.15)",
        "LOW": "rgba(239,68,68,0.15)",
    }.get(label, "rgba(100,116,139,0.1)")

    st.markdown(f"""
    <div class="result-card">
      <div class="section-label">Estimated Interior Depth</div>
      <div class="result-depth">{depth}</div>
      <div class="result-unit">centimeters</div>
      <div class="result-error">± {err} cm margin</div>
      <div class="confidence-badge" style="background:{badge_bg}; color:{color}; border: 1px solid {color}40;">
        {emoji} {label} CONFIDENCE
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards row ──────────────────────────────────────────────────
    fm = result["fit_metadata"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        method_label = fm.get("method_used", "—")
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{result['depth_m']:.3f} m</div>
          <div class="metric-label">Depth in meters</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        rim_ir = fm.get("rim_inlier_ratio", 0)
        ir_color = "#22c55e" if rim_ir >= 0.6 else "#f59e0b" if rim_ir >= 0.3 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:{ir_color}">{rim_ir:.0%}</div>
          <div class="metric-label">Rim plane inlier ratio</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        base_ir = fm.get("base_inlier_ratio", 0)
        ir_color2 = "#22c55e" if base_ir >= 0.6 else "#f59e0b" if base_ir >= 0.3 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:{ir_color2}">{base_ir:.0%}</div>
          <div class="metric-label">Base plane inlier ratio</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        method = fm.get("method_used", "—")
        method_icon = "💯" if "strip" in method else "📍" if "ransac" in method else "📐"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="font-size:1rem">{method_icon} {method}</div>
          <div class="metric-label">Estimation method</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fill Level Card ───────────────────────────────────────────────────
    fill = artifacts.get("fill_result", {})
    full_d = result["depth_cm"]
    filled_d = fill.get("filled_depth_cm")
    empty_d  = fill.get("empty_depth_cm")
    fill_ratio = fill.get("fill_ratio")
    fill_detected = fill.get("fill_detected", False)

    if filled_d is not None and empty_d is not None:
        fill_pct = int(round((fill_ratio or 0) * 100))
        empty_pct = 100 - fill_pct

        # Colour coding
        if fill_pct >= 70:
            fill_color = "#22c55e"   # green — very full
        elif fill_pct >= 35:
            fill_color = "#f59e0b"   # amber — half full
        else:
            fill_color = "#ef4444"   # red — nearly empty

        snr_note = "" if fill_detected else "<br><span style='color:#64748b;font-size:0.78rem;'>⚠️ Fill line confidence low — result approximate</span>"

        st.markdown(f"""
        <div class="fill-card">
          <div class="fill-card-title">📊 Fill Level Analysis</div>

          <div class="fill-row">
            <span class="fill-label">🔵 Full Depth</span>
            <div class="fill-bar-track">
              <div class="fill-bar-inner" style="width:100%; background:linear-gradient(90deg,#38bdf8,#818cf8);"></div>
            </div>
            <span class="fill-value">{full_d} cm</span>
          </div>

          <div class="fill-row">
            <span class="fill-label">🟢 Filled</span>
            <div class="fill-bar-track">
              <div class="fill-bar-inner" style="width:{fill_pct}%; background:{fill_color};"></div>
            </div>
            <span class="fill-value">{filled_d} cm</span>
          </div>

          <div class="fill-row">
            <span class="fill-label">⬜ Remaining</span>
            <div class="fill-bar-track">
              <div class="fill-bar-inner" style="width:{empty_pct}%; background:#334155;"></div>
            </div>
            <span class="fill-value">{empty_d} cm</span>
          </div>

          <div class="fill-pct-badge" style="background:{fill_color}22; color:{fill_color}; border:1px solid {fill_color}55;">
            {fill_pct}% filled · {empty_pct}% empty
          </div>
          {snr_note}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ Fill level could not be determined — interior mask too small or flat container.")

    # ── Warnings ──────────────────────────────────────────────────────────
    if result["warnings"]:
        st.markdown("**Warnings & Notes**")
        for w in result["warnings"]:
            st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Visualization tabs ────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        "🎭 Mask Overlay",
        "🌡️ Depth Heatmap",
        "🧊 3D Point Cloud",
        "📊 Confidence",
    ])

    with t1:
        st.markdown("**Rim region (blue) and base/interior region (red) extracted from SAM2 mask**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(artifacts["image_rgb"], caption="Original Image", use_container_width=True)
        with col_b:
            st.image(artifacts["annotated_rgb"], caption="Mask Overlay — Rim (blue) | Base (red)", use_container_width=True)

    with t2:
        st.plotly_chart(artifacts["depth_heatmap"], use_container_width=True)

    with t3:
        st.markdown(
            "**3D point cloud showing rim (blue) and base (red) regions with fitted planes.** "
            "If the planes look wrong, the depth estimate will be wrong — this is your primary debug view."
        )
        st.plotly_chart(artifacts["cloud_fig"], use_container_width=True)

    with t4:
        st.plotly_chart(artifacts["conf_fig"], use_container_width=True)
        cs = result["component_scores"]
        st.markdown("**Detailed Score Breakdown**")
        col_a, col_b, col_c, col_d = st.columns(4)
        score_items = [
            ("Plane Fit", cs["plane_fit"]),
            ("Point Count", cs["point_count"]),
            ("Visibility", cs["bottom_visibility"]),
            ("Parallelism", cs["parallelism"]),
        ]
        for col, (name, val) in zip([col_a, col_b, col_c, col_d], score_items):
            color_v = "#22c55e" if val >= 0.75 else "#f59e0b" if val >= 0.5 else "#ef4444"
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value" style="color:{color_v}">{val:.0%}</div>
                  <div class="metric-label">{name}</div>
                </div>""", unsafe_allow_html=True)


# ── Sidebar: About ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    **Container Depth Estimation** uses a full
    geometric computer vision pipeline:

    1. **CLAHE + Bilateral Denoise**
    2. **YOLO + SAM2** detection
    3. **Mask Surgery** (rim/base split)
    4. **Depth Anything V2** metric depth
    5. **2D → 3D** back-projection
    6. **RANSAC Plane Fitting**
    7. **Confidence Scoring**

    ---
    **Supported containers:**
    bowl, cup, bucket, pot, vase, bottle

    **Tip:** Use clear, well-lit images.
    Ensure the opening and base are both visible.
    """)
    st.markdown("---")
    st.markdown("**Models**")
    st.caption("• YOLOv8m (COCO)")
    st.caption("• SAM2-hiera-small")
    st.caption("• Depth Anything V2 (Small metric)")


# ── Main layout ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
  <div class="hero-title">📏 Container Depth Estimator</div>
  <div class="hero-sub">
    RANSAC · SAM2 · Depth Anything V2 · 3D Geometry · Confidence Scoring
  </div>
</div>
""", unsafe_allow_html=True)

col_upload, col_result = st.columns([1, 2], gap="large")

with col_upload:
    st.markdown("### Upload Image")
    uploaded = st.file_uploader(
        "Drop a container image (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="main_uploader",
        label_visibility="collapsed",
    )

    if uploaded:
        img_bytes = uploaded.read()
        from utils.preprocessing import load_image_from_bytes
        image_bgr = load_image_from_bytes(img_bytes)
        image_rgb_preview = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image(image_rgb_preview, caption="Uploaded Image", use_container_width=True)

        run_btn = st.button("🚀 Estimate Depth", key="run_btn")

        if run_btn:
            with col_result:
                status_ph = st.empty()
                progress_ph = st.progress(0.0)

                try:
                    with st.spinner("Running pipeline…"):
                        result, artifacts = run_pipeline(image_bgr, progress_ph, status_ph)

                    status_ph.empty()
                    progress_ph.empty()
                    render_result(result, artifacts)

                except RuntimeError as e:
                    status_ph.empty()
                    progress_ph.empty()
                    st.error(f"🚨 **Pipeline Error:** {e}")
                    logger.error(f"Pipeline error: {e}")

                except Exception as e:
                    status_ph.empty()
                    progress_ph.empty()
                    st.error(f"🚨 **Unexpected error:** {e}")
                    logger.error(f"Unexpected error:\n{traceback.format_exc()}")
    else:
        with col_result:
            st.markdown("""
            <div style="
              background: #1e293b;
              border: 1px dashed #334155;
              border-radius: 16px;
              padding: 4rem 2rem;
              text-align: center;
              color: #64748b;
            ">
              <div style="font-size: 3rem; margin-bottom: 1rem;">📷</div>
              <div style="font-size: 1.1rem; font-weight: 600;">Upload an image to begin</div>
              <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                Results will appear here after analysis
              </div>
            </div>
            """, unsafe_allow_html=True)
