"""
Traffic Sign Recognition — Streamlit Dashboard
================================================
Demonstrates three inference pipelines:
  • CNN   — deep learning classifier (.keras)
  • SVM   — classical ML classifier (.pkl)
  • Detection — HSV + contour CV pipeline
"""

from __future__ import annotations

import pathlib
import numpy as np
import streamlit as st

# ── Utility modules ───────────────────────────────────────────────────────────
from utils.preprocessing import (
    load_image_as_array,
    bgr_to_rgb,
    build_cnn_input,
    infer_cnn_input_kind,
    log_cnn_tensor,
    select_sign_roi_bgr,
)
from utils.keras_model_load import load_keras_model_file
from utils.svm_features import extract_features
from utils.detection import detect_signs, annotate_with_labels

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# PATHS — look in /models first, fall back to Outputs directory
# ═════════════════════════════════════════════════════════════════════════════

BASE_DIR    = pathlib.Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "Outputs" / "Classification_and_Evaluation"

SEARCH_DIRS = [MODELS_DIR, OUTPUTS_DIR]


def _find_file(candidates: list[str]) -> pathlib.Path | None:
    """Return the first existing file path from a list of relative names."""
    for search_dir in SEARCH_DIRS:
        for name in candidates:
            p = search_dir / name
            if p.exists():
                return p
    # wildcard search for .keras files
    for search_dir in SEARCH_DIRS:
        if search_dir.exists():
            found = list(search_dir.glob("*.keras"))
            if found:
                return found[0]
    return None


def _find_keras_model() -> pathlib.Path | None:
    for search_dir in SEARCH_DIRS:
        if search_dir.exists():
            found = sorted(search_dir.glob("*.keras"))
            if found:
                return found[0]
    return None


def _find_pkl(candidates: list[str]) -> pathlib.Path | None:
    for search_dir in SEARCH_DIRS:
        for name in candidates:
            p = search_dir / name
            if p.exists():
                return p
    return None


# ═════════════════════════════════════════════════════════════════════════════
# CACHED MODEL LOADERS
# ═════════════════════════════════════════════════════════════════════════════

def load_cnn_model():
    path = _find_keras_model()
    if path is None:
        return None, "CNN model (.keras) not found. Copy it to the /models folder."

    path_str = str(path)
    try:
        model = load_keras_model_file(path_str)
    except Exception as exc:
        return None, f"Failed to load CNN model: {exc}"

    print("\n[CNN] Loaded model from:", path_str, flush=True)
    print("[CNN] model.input_shape:", model.input_shape, flush=True)
    try:
        print("[CNN] model.output_shape:", model.output_shape, flush=True)
    except Exception:
        print("[CNN] model.output_shape: (multi-output or unknown)", flush=True)
    model.summary(print_fn=lambda line: print("  " + line, flush=True))

    return model, path_str


@st.cache_resource(show_spinner="Loading SVM artifacts…")
def load_svm_artifacts():
    import joblib

    svm_path = _find_pkl(["svm_trained_model.pkl", "svm_model.pkl"])
    scaler_path = _find_pkl(["feature_scaler.pkl", "scaler.pkl"])
    encoder_path = _find_pkl(["label_encoder.pkl", "encoder.pkl"])

    errors = []
    svm = scaler = encoder = None

    if svm_path:
        try:
            svm = joblib.load(str(svm_path))
        except Exception as e:
            errors.append(f"SVM: {e}")
    else:
        errors.append("svm_trained_model.pkl not found")

    if scaler_path:
        try:
            scaler = joblib.load(str(scaler_path))
        except Exception as e:
            errors.append(f"Scaler: {e}")
    else:
        errors.append("feature_scaler.pkl not found")

    if encoder_path:
        try:
            encoder = joblib.load(str(encoder_path))
        except Exception as e:
            errors.append(f"Encoder: {e}")
    else:
        errors.append("label_encoder.pkl not found")

    return svm, scaler, encoder, errors


@st.cache_resource(show_spinner="Loading feature scaler…")
def load_feature_scaler():
    """Same artifact as SVM training (required for Conv1D CNN in this project)."""
    import joblib

    path = _find_pkl(["feature_scaler.pkl", "scaler.pkl"])
    if path is None:
        return None
    try:
        return joblib.load(str(path))
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading label encoder…")
def load_label_encoder_only():
    import joblib
    path = _find_pkl(["label_encoder.pkl", "encoder.pkl"])
    if path is None:
        return None
    try:
        return joblib.load(str(path))
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
        /* Main background */
        .stApp { background-color: #0f1117; color: #e0e0e0; }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #161b27 0%, #0d1117 100%);
            border-right: 1px solid #30363d;
        }

        /* Card containers */
        .cv-card {
            background: #161b27;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
        }

        /* Prediction badge */
        .pred-badge {
            display: inline-block;
            background: linear-gradient(135deg, #1f6feb, #388bfd);
            color: white;
            border-radius: 8px;
            padding: 6px 18px;
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-top: 6px;
        }
        .pred-badge.red  { background: linear-gradient(135deg, #da3633, #f85149); }
        .pred-badge.green{ background: linear-gradient(135deg, #238636, #3fb950); }

        /* Section title */
        .section-title {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: #8b949e;
            margin-bottom: 8px;
        }

        /* Confidence bar */
        .conf-bar-wrap {
            background: #21262d;
            border-radius: 6px;
            height: 12px;
            width: 100%;
            margin-top: 8px;
            overflow: hidden;
        }
        .conf-bar-fill {
            height: 100%;
            border-radius: 6px;
            background: linear-gradient(90deg, #1f6feb, #388bfd);
            transition: width 0.4s ease;
        }

        /* Divider */
        hr.cv-hr { border-color: #30363d; margin: 16px 0; }

        /* Image border */
        img { border-radius: 8px; }

        /* Detection chip */
        .det-chip {
            display: inline-block;
            border-radius: 20px;
            padding: 3px 12px;
            font-size: 0.82rem;
            font-weight: 600;
            margin: 3px 3px 3px 0;
        }
        .chip-red  { background: #3d1c1c; color: #f85149; border: 1px solid #6e2424; }
        .chip-blue { background: #0d2044; color: #388bfd; border: 1px solid #1f6feb; }
        .chip-green{ background: #0d2d1e; color: #3fb950; border: 1px solid #238636; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🚦 Traffic Sign CV")
    st.markdown("<hr class='cv-hr'>", unsafe_allow_html=True)

    st.markdown("### Select Pipeline")
    pipeline = st.radio(
        label="pipeline",
        options=[
            "CNN Classification (Deep Learning)",
            "SVM Classification (Classical ML)",
            "Traffic Sign Detection (HSV + contours)",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.checkbox(
        "Use ROI from object_detection.py for CNN/SVM (same as detection_results_eval.csv)",
        value=True,
        key="prefer_sign_crop",
        help="Crop = detect_sign_bbox from src/Detection; if no box, full image (CSV fallback).",
    )

    st.markdown("<hr class='cv-hr'>", unsafe_allow_html=True)

    st.markdown("#### About")
    pipeline_info = {
        "CNN Classification (Deep Learning)": (
            "🧠 **Deep Learning (Conv1D)**\n\n"
            "This checkpoint classifies **2277-D feature vectors** (same as `features.csv`): "
            "HSV histogram + Canny contour area + HOG, then **`StandardScaler`** → tensor "
            "shape **(1, 2277, 1)**. Keep `feature_scaler.pkl` beside the `.keras` file."
        ),
        "SVM Classification (Classical ML)": (
            "📐 **Classical ML**\n\n"
            "Support Vector Machine on **HSV histogram + contour area + HOG** features "
            "(2277-D), scaled with the saved `StandardScaler`."
        ),
        "Traffic Sign Detection (HSV + contours)": (
            "🔍 **Computer Vision**\n\n"
            "HSV colour masking → Gaussian blur → Morphological ops → "
            "Contour detection → Bounding box selection."
        ),
    }
    st.info(pipeline_info[pipeline])

    st.markdown("<hr class='cv-hr'>", unsafe_allow_html=True)
    with st.expander("Developer"):
        st.caption("Logs appear in the terminal where you ran `streamlit run`.")
        st.checkbox(
            "CNN debug (input shape, min/max, extra traces)",
            value=False,
            key="cnn_debug_logs",
        )

    st.markdown("<hr class='cv-hr'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.75rem;color:#8b949e;text-align:center'>"
        "Graduation Project Demo · Computer Vision<br>"
        "© 2026</p>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════

pipeline_icons = {
    "CNN Classification (Deep Learning)": "🧠",
    "SVM Classification (Classical ML)": "📐",
    "Traffic Sign Detection (HSV + contours)": "🔍",
}
st.markdown(
    f"<h1 style='margin-bottom:2px'>{pipeline_icons[pipeline]} {pipeline}</h1>",
    unsafe_allow_html=True,
)

tag_colors = {
    "CNN Classification (Deep Learning)": "#1f6feb",
    "SVM Classification (Classical ML)": "#388bfd",
    "Traffic Sign Detection (HSV + contours)": "#3fb950",
}
st.markdown(
    f"<span style='background:{tag_colors[pipeline]};color:white;padding:3px 12px;"
    f"border-radius:20px;font-size:0.8rem;font-weight:600'>Active Pipeline</span>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# IMAGE UPLOAD
# ═════════════════════════════════════════════════════════════════════════════

uploaded = st.file_uploader(
    "Upload a traffic sign image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="The image will be processed by the selected pipeline.",
)

if uploaded is None:
    st.markdown(
        "<div class='cv-card' style='text-align:center;padding:48px;'>"
        "<span style='font-size:3rem'>📷</span><br>"
        "<p style='color:#8b949e;margin-top:12px'>Upload an image to begin analysis</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()


# ─── Load image ───────────────────────────────────────────────────────────────
uploaded.seek(0)
image_bgr = load_image_as_array(uploaded)

if image_bgr is None:
    st.error("Could not decode the uploaded image. Please try a different file.")
    st.stop()

image_rgb = bgr_to_rgb(image_bgr)

use_ml_crop = bool(st.session_state.get("prefer_sign_crop", True))
ml_bgr, ml_input_note = select_sign_roi_bgr(image_bgr, use_detection=use_ml_crop)

# ═════════════════════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

col_left, col_right = st.columns([1, 1], gap="large")


# ── Left: Input image ─────────────────────────────────────────────────────────
with col_left:
    st.markdown("<div class='section-title'>Input Image</div>", unsafe_allow_html=True)
    st.image(image_rgb, use_container_width=True)

    h_orig, w_orig = image_bgr.shape[:2]
    st.markdown(
        f"<p style='color:#8b949e;font-size:0.8rem;margin-top:4px'>"
        f"📐 {w_orig} × {h_orig} px &nbsp;|&nbsp; 🎨 RGB</p>",
        unsafe_allow_html=True,
    )
    if use_ml_crop:
        st.caption(f"CNN/SVM input: {ml_input_note}")


# ═════════════════════════════════════════════════════════════════════════════
# ─── PIPELINE EXECUTION ──────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 1. CNN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if pipeline == "CNN Classification (Deep Learning)":

    model, model_info = load_cnn_model()
    encoder = load_label_encoder_only()
    feature_scaler = load_feature_scaler()
    debug_cnn = bool(st.session_state.get("cnn_debug_logs", False))

    with col_right:
        st.markdown("<div class='section-title'>CNN Prediction</div>", unsafe_allow_html=True)

        if model is None:
            st.error(f"⚠️ {model_info}")
            st.info(
                "Make sure the `.keras` file is placed in the `/models` folder "
                "or in `Outputs/Classification_and_Evaluation/`."
            )
            st.stop()

        # ── Pre-process & predict ─────────────────────────────────────────────
        with st.spinner("Running CNN inference…"):
            x, cnn_kind, prep_err = build_cnn_input(
                model, ml_bgr, feature_scaler, debug=debug_cnn
            )
            if prep_err or x is None:
                st.error(f"⚠️ {prep_err or 'Could not build model input.'}")
                st.stop()
            preds = model.predict(x, verbose=0)[0]  # logits or softmax: (n_classes,)

        if debug_cnn:
            log_cnn_tensor("CNN raw output (pre-argmax)", preds.reshape(1, -1))
            print(
                f"[CNN DEBUG] softmax sum ≈ {float(np.sum(preds)):.6f} "
                f"(expect ~1.0 if final activation is softmax)",
                flush=True,
            )
            if encoder is not None and hasattr(encoder, "classes_"):
                print(f"[CNN DEBUG] LabelEncoder.classes_[:12] = {encoder.classes_[:12]}", flush=True)

        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])

        # ── Decode label ──────────────────────────────────────────────────────
        if encoder is not None:
            try:
                label = str(encoder.inverse_transform([top_idx])[0])
            except Exception:
                label = f"Class {top_idx}"
        else:
            label = f"Class {top_idx}"

        # ── Display result ────────────────────────────────────────────────────
        badge_cls = "green" if confidence >= 0.7 else ("pred-badge" if confidence >= 0.4 else "red")
        st.markdown(
            f"<div class='cv-card'>"
            f"<p style='margin:0 0 4px;color:#8b949e;font-size:0.85rem'>Predicted Class</p>"
            f"<span class='pred-badge {badge_cls}'>{label}</span>"
            f"<hr class='cv-hr'>"
            f"<p style='margin:0 0 4px;color:#8b949e;font-size:0.85rem'>Confidence</p>"
            f"<p style='font-size:1.5rem;font-weight:700;margin:0'>{confidence * 100:.1f}%</p>"
            f"<div class='conf-bar-wrap'>"
            f"  <div class='conf-bar-fill' style='width:{confidence * 100:.1f}%'></div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Top-3 predictions ─────────────────────────────────────────────────
        top3_idx = np.argsort(preds)[::-1][:3]
        st.markdown("**Top-3 Predictions**")
        for rank, idx in enumerate(top3_idx, 1):
            prob = float(preds[idx])
            if encoder is not None:
                try:
                    lbl = str(encoder.inverse_transform([idx])[0])
                except Exception:
                    lbl = f"Class {idx}"
            else:
                lbl = f"Class {idx}"
            bar_html = (
                f"<div style='margin-bottom:8px'>"
                f"  <div style='display:flex;justify-content:space-between;"
                f"font-size:0.85rem;margin-bottom:3px'>"
                f"    <span>#{rank} {lbl}</span>"
                f"    <span style='color:#8b949e'>{prob * 100:.1f}%</span>"
                f"  </div>"
                f"  <div class='conf-bar-wrap'>"
                f"    <div class='conf-bar-fill' style='width:{prob * 100:.1f}%'></div>"
                f"  </div>"
                f"</div>"
            )
            st.markdown(bar_html, unsafe_allow_html=True)

        st.progress(float(min(max(confidence, 0.0), 1.0)))

        if cnn_kind == "conv1d_features":
            input_desc = "2277-D scaled features → Conv1D"
        else:
            try:
                _sh = model.input_shape
                input_desc = f"{int(_sh[2])}×{int(_sh[1])} RGB (normalized)"
            except Exception:
                input_desc = "image tensor"

        st.markdown(
            f"<p style='color:#8b949e;font-size:0.75rem;margin-top:8px'>"
            f"Model: <code>{pathlib.Path(str(model_info)).name}</code> &nbsp;|&nbsp; "
            f"Input: {input_desc}</p>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. SVM PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif pipeline == "SVM Classification (Classical ML)":

    svm, scaler, encoder, errors = load_svm_artifacts()

    with col_right:
        st.markdown("<div class='section-title'>SVM Prediction</div>", unsafe_allow_html=True)

        if errors:
            for err in errors:
                st.warning(f"⚠️ {err}")

        if svm is None:
            st.error("SVM model could not be loaded. Check the /models folder.")
            st.stop()

        with st.spinner("Extracting notebook features (HSV+HOG+…) & running SVM…"):
            features = extract_features(ml_bgr)            # (1, n_feat)
            if scaler is not None:
                try:
                    features = scaler.transform(features)
                except Exception as e:
                    st.warning(f"Scaler transform failed ({e}). Using raw features.")

            prediction = svm.predict(features)
            pred_label = str(prediction[0])

            # Try to decode via label encoder
            if encoder is not None:
                try:
                    pred_label = str(encoder.inverse_transform(prediction)[0])
                except Exception:
                    # encoder may already store string labels
                    pass

            # Confidence — decision_function gives margin scores
            confidence_text = "N/A"
            conf_val = 0.0
            try:
                if hasattr(svm, "predict_proba"):
                    proba = svm.predict_proba(features)[0]
                    conf_val = float(np.max(proba))
                    confidence_text = f"{conf_val * 100:.1f}%"
                elif hasattr(svm, "decision_function"):
                    decision = svm.decision_function(features)[0]
                    # Normalise via softmax for display purposes
                    if np.isscalar(decision):
                        conf_val = float(1 / (1 + np.exp(-abs(decision))))
                    else:
                        exp_d = np.exp(decision - np.max(decision))
                        proba = exp_d / exp_d.sum()
                        conf_val = float(np.max(proba))
                    confidence_text = f"{conf_val * 100:.1f}%"
            except Exception:
                pass

        st.markdown(
            f"<div class='cv-card'>"
            f"<p style='margin:0 0 4px;color:#8b949e;font-size:0.85rem'>Predicted Class</p>"
            f"<span class='pred-badge'>{pred_label}</span>"
            f"<hr class='cv-hr'>"
            f"<p style='margin:0 0 4px;color:#8b949e;font-size:0.85rem'>Confidence Score</p>"
            f"<p style='font-size:1.5rem;font-weight:700;margin:0'>{confidence_text}</p>"
            + (
                f"<div class='conf-bar-wrap'>"
                f"<div class='conf-bar-fill' style='width:{conf_val * 100:.1f}%'></div>"
                f"</div>"
                if conf_val > 0 else ""
            )
            + f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Pipeline Details**")
        st.markdown(
            "<div class='cv-card'>"
            "<p style='margin:0;font-size:0.85rem;line-height:1.9'>"
            "1️⃣ &nbsp;Resize image to <b>64 × 64</b><br>"
            "2️⃣ &nbsp;<b>HSV 3D histogram</b> (8×8×8 bins) + "
            "<b>Canny contour area</b> + <b>HOG</b> (9 ori, 8×8 cells, 2×2 blocks, L2-Hys)<br>"
            "3️⃣ &nbsp;Vector length <b>2277</b> → <b>StandardScaler</b><br>"
            "4️⃣ &nbsp;SVM <b>predict</b> → <code>label_encoder</code> decode"
            "</p></div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. DETECTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif pipeline == "Traffic Sign Detection (HSV + contours)":

    # Optionally load CNN for classifying detected crops
    model, model_info = load_cnn_model()
    encoder = load_label_encoder_only()
    feature_scaler = load_feature_scaler()
    debug_cnn = bool(st.session_state.get("cnn_debug_logs", False))

    cnn_kind = infer_cnn_input_kind(model) if model is not None else None
    skip_cnn_classify = (
        model is not None
        and cnn_kind == "conv1d_features"
        and feature_scaler is None
    )

    with st.spinner("Running detection pipeline…"):
        annotated_bgr, detections = detect_signs(image_bgr)

        if model is not None and not skip_cnn_classify:
            for det in detections:
                batch, _, err = build_cnn_input(
                    model, det.crop, feature_scaler, debug=debug_cnn
                )
                if err or batch is None:
                    continue
                preds = model.predict(batch, verbose=0)[0]
                top_idx = int(np.argmax(preds))
                det.confidence = float(preds[top_idx])
                if encoder is not None:
                    try:
                        det.label = str(encoder.inverse_transform([top_idx])[0])
                    except Exception:
                        det.label = f"Class {top_idx}"
                else:
                    det.label = f"Class {top_idx}"

            annotated_bgr = annotate_with_labels(annotated_bgr, detections)

    if skip_cnn_classify:
        st.warning(
            "CNN loaded but `feature_scaler.pkl` is missing — "
            "skipping per-crop classification for Conv1D models."
        )

    annotated_rgb = bgr_to_rgb(annotated_bgr)

    with col_right:
        st.markdown("<div class='section-title'>Detection Result</div>", unsafe_allow_html=True)
        st.image(annotated_rgb, use_container_width=True)

        n = len(detections)
        status_color = "#3fb950" if n > 0 else "#f85149"
        st.markdown(
            f"<div class='cv-card' style='text-align:center'>"
            f"<p style='font-size:2rem;font-weight:700;margin:0;color:{status_color}'>{n}</p>"
            f"<p style='color:#8b949e;margin:4px 0 0'>Sign{'s' if n != 1 else ''} Detected</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if detections:
            st.markdown("**Detected Regions**")
            for i, det in enumerate(detections, 1):
                chip_cls = "chip-red" if det.color == "red" else "chip-blue"
                label_str = f"&nbsp;→&nbsp;<b>{det.label}</b>" if det.label else ""
                conf_str  = (
                    f"&nbsp;<span style='color:#8b949e'>({det.confidence * 100:.0f}%)</span>"
                    if det.confidence > 0 else ""
                )
                area_str = f"Area: {det.area:,} px²"
                x, y, w, h = det.bbox
                st.markdown(
                    f"<div class='cv-card' style='padding:12px 16px;margin-bottom:8px'>"
                    f"  <span class='det-chip {chip_cls}'>{det.color.upper()}</span>"
                    f"  <strong>Sign #{i}</strong>{label_str}{conf_str}"
                    f"  <p style='font-size:0.78rem;color:#8b949e;margin:6px 0 0'>"
                    f"  📍 ({x}, {y}) &nbsp;|&nbsp; 📐 {w}×{h} &nbsp;|&nbsp; {area_str}"
                    f"  </p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # ── Show crops ────────────────────────────────────────────────────
            st.markdown("**Detected Crops**")
            crop_cols = st.columns(min(len(detections), 4))
            for col, det in zip(crop_cols, detections):
                with col:
                    crop_rgb = bgr_to_rgb(det.crop)
                    st.image(crop_rgb, use_container_width=True)
                    caption = det.label if det.label else det.color.upper()
                    st.markdown(
                        f"<p style='text-align:center;font-size:0.78rem;"
                        f"color:#8b949e;margin-top:4px'>{caption}</p>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info(
                "No traffic signs detected. Try an image with a clear red or blue sign "
                "under good lighting conditions."
            )

        # ── Pipeline steps summary ────────────────────────────────────────────
        with st.expander("Detection Pipeline Details"):
            st.markdown(
                "Parameters match **`src/Detection/object_detection.py`** (CSV used by "
                "`FeatureExtraction.ipynb`). Streamlit adds **NMS** only to dedupe boxes.\n\n"
                "| Step | Operation | Parameters |\n"
                "|------|-----------|------------|\n"
                "| 1 | BGR→HSV | OpenCV |\n"
                "| 2 | Red mask | "
                "H∈[0,10]∪[170,180], S≥70, V≥50 |\n"
                "| 3 | Blue mask | H∈[90,140], S≥60, V≥40 |\n"
                "| 4 | Mask denoise | GaussianBlur 5×5 on **binary mask** |\n"
                "| 5 | Morphology | Open 3×3, Close 5×5 (on mask) |\n"
                "| 6 | Contours | RETR_EXTERNAL |\n"
                "| 7 | Geometry | convex hull → `boundingRect`, area≥300, aspect 0.6–1.8, "
                "box area ≤ 90% image |\n"
                "| 8 | Score | `rect_area × fill_ratio`; **NMS** IoU=0.35, max 8 boxes |\n"
                + ("| 9 | CNN on crop | optional per ROI |" if model else "")
            )


# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("<br><hr class='cv-hr'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#8b949e;font-size:0.78rem'>"
    "Traffic Sign Recognition · Computer Visio Project · "
    "Built with Streamlit · TensorFlow · scikit-learn · OpenCV"
    "</p>",
    unsafe_allow_html=True,
)
