"""
Image preprocessing for CNN and helper routines shared with ML pipelines.

This project's Keras model is a Conv1D classifier on **scaled 2277-D feature vectors**
(shape (batch, 2277, 1)), not raw images. Image-based CNNs remain supported when the
model reports a 4D input (N, H, W, C).
"""

from __future__ import annotations

import sys
from typing import Any, Literal, Optional

import cv2
import numpy as np

from utils.svm_features import extract_training_style_features

InputKind = Literal["image_nhwc", "conv1d_features"]

# Must match FeatureExtraction.ipynb / features.csv numeric columns
EXPECTED_FEATURE_DIM = 2277


def load_image_as_array(uploaded_file) -> np.ndarray | None:
    """Convert a Streamlit uploaded file to a BGR NumPy array (OpenCV format)."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) image to RGB (display-friendly)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def select_sign_roi_bgr(
    image_bgr: np.ndarray,
    use_detection: bool = True,
) -> tuple[np.ndarray, str]:
    """
    Aligns CNN/SVM input with ``FeatureExtraction.ipynb``:

    * CSV rows from ``src/Detection/object_detection.py`` supply ``x,y,w,h``;
      ``get_roi`` crops that rectangle (no margin).
    * If the same detector finds no sign here, use the full image (CSV fallback).
    """
    if not use_detection or image_bgr is None or image_bgr.size == 0:
        return image_bgr, "Full frame (CSV-style ROI disabled in sidebar)"

    from utils.detection import roi_for_feature_extraction_notebook

    return roi_for_feature_extraction_notebook(image_bgr)


def infer_cnn_input_kind(model: Any) -> InputKind:
    """
    Distinguish image CNN (N,H,W,C) from Conv1D-on-features (N, steps, channels).

    Note: Keras may report ``(None, 2277, None)`` for the channel dimension; that must
    still be treated as Conv1D, not as an image model.
    """
    try:
        shape = model.input_shape
    except Exception:
        return "image_nhwc"

    if shape is None or len(shape) < 3:
        return "image_nhwc"

    if len(shape) == 4:
        return "image_nhwc"

    if len(shape) == 3:
        length, channels = shape[1], shape[2]
        if length is None:
            return "image_nhwc"
        n_steps = int(length)
        # Conv1D on long feature sequences (this project: 2277); not e.g. (batch, 64, 64)
        if n_steps <= 32:
            return "image_nhwc"
        if channels is None or int(channels) in (1, 2, 3, 4):
            return "conv1d_features"

    return "image_nhwc"


def log_cnn_tensor(tag: str, batch: np.ndarray, file=sys.stdout) -> None:
    """Debug helper: tensor shape/dtype/range (print to terminal when running Streamlit)."""
    print(f"\n[CNN DEBUG] {tag}", file=file, flush=True)
    print(f"  shape : {batch.shape}", file=file, flush=True)
    print(f"  dtype : {batch.dtype}", file=file, flush=True)
    print(
        f"  min/max : {float(np.min(batch)):.6g} / {float(np.max(batch)):.6g}",
        file=file,
        flush=True,
    )
    if np.isnan(batch).any():
        print("  WARNING: input contains NaN", file=file, flush=True)


def preprocess_for_cnn(
    image_bgr: np.ndarray, target_size: tuple[int, int]
) -> np.ndarray:
    """
    Resize and normalize an image for a **spatial** CNN.

    Returns:
        Float32 array of shape (1, H, W, C), RGB, values in [0, 1].
    """
    h, w = target_size[0], target_size[1]
    resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def preprocess_for_conv1d_cnn(
    features_2d: np.ndarray,
) -> np.ndarray:
    """
    Reshape scaled feature rows for Conv1D: (1, n_feat) → (1, n_feat, 1), float32.
    """
    if features_2d.ndim == 1:
        features_2d = features_2d.reshape(1, -1)
    out = np.asarray(features_2d, dtype=np.float32).reshape(
        features_2d.shape[0], features_2d.shape[1], 1
    )
    return out


def preprocess_for_svm(image_bgr: np.ndarray, target_size: tuple = (64, 64)) -> np.ndarray:
    """Resize an image (legacy helper); full SVM features use ``utils.svm_features``."""
    h, w = target_size
    resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return resized


def _expected_conv1d_steps(model: Any) -> Optional[int]:
    try:
        sh = model.input_shape
        if sh is None or len(sh) != 3 or sh[1] is None:
            return None
        return int(sh[1])
    except Exception:
        return None


def build_cnn_input(
    model: Any,
    image_bgr: np.ndarray,
    scaler: Optional[Any] = None,
    *,
    debug: bool = False,
) -> tuple[Optional[np.ndarray], InputKind, Optional[str]]:
    """
    Build model input: **Conv1D feature pipeline** (this project) or spatial CNN.

    For Conv1D: raw image → same 2277-D features as training → ``scaler.transform``
    → ``float32`` tensor ``(1, N, 1)``.
    """
    kind = infer_cnn_input_kind(model)

    if kind == "conv1d_features":
        if scaler is None:
            return (
                None,
                kind,
                "This Conv1D CNN expects scaled feature vectors. "
                "Add `feature_scaler.pkl` next to the model (same scaler as training).",
            )

        expected_steps = _expected_conv1d_steps(model)
        n_scaler = getattr(scaler, "n_features_in_", None)
        if (
            expected_steps is not None
            and n_scaler is not None
            and int(n_scaler) != expected_steps
        ):
            return (
                None,
                kind,
                f"Scaler/model mismatch: StandardScaler expects {n_scaler} features, "
                f"but model input length is {expected_steps}. "
                "Use the `feature_scaler.pkl` saved with this `.keras` file.",
            )

        feats = extract_training_style_features(image_bgr)
        if feats.shape[1] != EXPECTED_FEATURE_DIM:
            return (
                None,
                kind,
                f"Feature extraction length {feats.shape[1]} != expected {EXPECTED_FEATURE_DIM}.",
            )

        try:
            feats_scaled = scaler.transform(feats)
        except Exception as exc:
            return None, kind, f"Scaler transform failed: {exc}"

        if expected_steps is not None and feats_scaled.shape[1] != expected_steps:
            return (
                None,
                kind,
                f"After scaling, feature dim is {feats_scaled.shape[1]} but model expects {expected_steps}.",
            )

        batch = preprocess_for_conv1d_cnn(feats_scaled)
        if debug:
            log_cnn_tensor("Conv1D model input (scaled, reshaped)", batch)
        return batch, kind, None

    try:
        input_shape = model.input_shape
        target_h, target_w = int(input_shape[1]), int(input_shape[2])
    except Exception:
        target_h, target_w = 64, 64

    batch = preprocess_for_cnn(image_bgr, (target_h, target_w))
    if debug:
        log_cnn_tensor("Image CNN input (normalized RGB)", batch)
    return batch, kind, None
