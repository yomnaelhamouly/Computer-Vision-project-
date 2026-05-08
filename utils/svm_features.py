"""
Feature extraction for SVM and (Conv1D) CNN classification.

Matches ``Notebooks/FeatureExtraction.ipynb``:
  • Resize to 64×64
  • 3D HSV histogram (8×8×8 bins) → 512 dims
  • Contour-area sum from Canny edges → 1 dim
  • HOG (9 ori, 8×8 cells, 2×2 blocks, L2-Hys) → 1764 dims
Total: 2277 features, then StandardScaler in training.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog

SVM_INPUT_SIZE = (64, 64)

# Identical to FeatureExtraction.ipynb 
HOG_PARAMS_NOTEBOOK = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=False,
    feature_vector=True,
)


def extract_training_style_features(image_bgr: np.ndarray) -> np.ndarray:
    """
    Full 2277-D vector as in features.csv / scaler / Conv1D CNN training.

    Args:
        image_bgr: Full image or crop in OpenCV BGR format.

    Returns:
        Array of shape (1, 2277), float64.
    """
    h, w = SVM_INPUT_SIZE
    roi = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256],
    )
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges255 = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(
        edges255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    area = float(sum(cv2.contourArea(c) for c in contours))

    hog_feat = hog(gray, **HOG_PARAMS_NOTEBOOK)

    vec = np.hstack([hsv_hist, np.array([area], dtype=np.float64), hog_feat])
    return vec.reshape(1, -1)


def extract_features(image_bgr: np.ndarray) -> np.ndarray:
    """Primary entry point for SVM / shared ML preprocessing."""
    return extract_training_style_features(image_bgr)
