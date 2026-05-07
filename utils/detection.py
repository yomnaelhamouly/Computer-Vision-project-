"""
Traffic sign detection — parameters and mask pipeline copied from
``src/Detection/object_detection.py`` (used to build ``detection_results_eval.csv``).

Streamer-facing helpers add multi-candidate scoring + greedy NMS to cut duplicate
boxes only; HSV ranges, morphology on the mask, convex hull, aspect and area
rules are unchanged from the project script.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── From object_detection.py (verbatim intent) ──────────────────────────────
MIN_AREA = 300

RED_RANGES = [
    ((0, 70, 50), (10, 255, 255)),
    ((170, 70, 50), (180, 255, 255)),
]

BLUE_RANGE = ((90, 60, 40), (140, 255, 255))

ASPECT_MIN = 0.6
ASPECT_MAX = 1.8
MAX_IMAGE_COVER = 0.9

# Multi-box UI: suppress overlaps (same mask often fragments); does not change mask.
NMS_IOU = 0.35
MAX_DETECTIONS = 8


@dataclass
class Detection:
    bbox: tuple
    color: str
    crop: np.ndarray
    confidence: float = 0.0
    label: str = ""
    area: int = 0
    score: float = 0.0
    contour: Optional[np.ndarray] = field(default=None, repr=False)


def build_color_mask(hsv: np.ndarray) -> np.ndarray:
    """Same steps as ``object_detection.build_color_mask`` (mask-only blur + morph)."""
    red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in RED_RANGES:
        red_mask |= cv2.inRange(
            hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)
        )

    blue_mask = cv2.inRange(
        hsv,
        np.array(BLUE_RANGE[0], dtype=np.uint8),
        np.array(BLUE_RANGE[1], dtype=np.uint8),
    )

    combined = cv2.bitwise_or(red_mask, blue_mask)
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    combined = cv2.GaussianBlur(combined, (5, 5), 0)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel5)
    return combined


def _dominant_color(crop_hsv: np.ndarray) -> str:
    red_mask = np.zeros(crop_hsv.shape[:2], dtype=np.uint8)
    for lower, upper in RED_RANGES:
        red_mask |= cv2.inRange(
            crop_hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)
        )
    blue_mask = cv2.inRange(
        crop_hsv,
        np.array(BLUE_RANGE[0], dtype=np.uint8),
        np.array(BLUE_RANGE[1], dtype=np.uint8),
    )
    red_px = cv2.countNonZero(red_mask)
    blue_px = cv2.countNonZero(blue_mask)
    return "blue" if blue_px > red_px else "red"


def detect_sign_bbox(image_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], np.ndarray]:
    """
    Same behaviour as ``object_detection.detect_sign_bbox``: single best box by
    ``rect_area * fill_ratio``, or ``None`` if nothing passes filters.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = build_color_mask(hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_h, image_w = image_bgr.shape[:2]
    image_area = image_h * image_w
    best_box = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        hull = cv2.convexHull(cnt)
        x, y, w, h = cv2.boundingRect(hull)
        rect_area = w * h
        if rect_area <= 0:
            continue

        fill_ratio = area / rect_area
        aspect_ratio = w / float(h)

        if rect_area > image_area * MAX_IMAGE_COVER:
            continue
        if aspect_ratio < ASPECT_MIN or aspect_ratio > ASPECT_MAX:
            continue

        score = rect_area * fill_ratio
        if score > best_score:
            best_score = score
            best_box = (x, y, w, h)

    return best_box, mask


def roi_for_feature_extraction_notebook(
    image_bgr: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """
    Matches ``FeatureExtraction.get_roi`` when CSV rows come from
    ``object_detection.process_split``:

    * If ``detect_sign_bbox`` finds a box → crop ``image[y:y+h, x:x+w]`` (no margin).
    * If not → full image (same as CSV row ``0, 0, image_w, image_h``).
    """
    box, _ = detect_sign_bbox(image_bgr)
    ih, iw = image_bgr.shape[:2]
    if box is None:
        return image_bgr, "Full image (no detection — CSV-style fallback)"

    x, y, w, h = box
    roi = image_bgr[y : y + h, x : x + w]
    if roi.size == 0:
        return image_bgr, "Full image (empty crop fallback)"
    return roi, "ROI from detector (src/Detection/object_detection.py)"


def _iou_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter + 1e-6
    return inter / union


def _nms_xywh(
    boxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    iou_thresh: float,
) -> List[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [
            j
            for j in order
            if _iou_xywh(boxes[i], boxes[j]) < iou_thresh
        ]
    return keep


def detect_signs(image_bgr: np.ndarray) -> tuple[np.ndarray, List[Detection]]:
    """
    Build mask exactly like ``object_detection``, then collect **all** qualifying
    hull boxes (same thresholds), sort by ``rect_area * fill_ratio``, NMS, draw top
    ``MAX_DETECTIONS``.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = build_color_mask(hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ih, iw = image_bgr.shape[:2]
    image_area = ih * iw

    raw: List[Tuple[float, int, int, int, int, np.ndarray, int]] = []

    for cnt in contours:
        area = int(cv2.contourArea(cnt))
        if area < MIN_AREA:
            continue

        hull = cv2.convexHull(cnt)
        x, y, w, h = cv2.boundingRect(hull)
        rect_area = w * h
        if rect_area <= 0:
            continue

        fill_ratio = float(area) / rect_area
        aspect_ratio = w / float(h)

        if rect_area > image_area * MAX_IMAGE_COVER:
            continue
        if aspect_ratio < ASPECT_MIN or aspect_ratio > ASPECT_MAX:
            continue

        score = rect_area * fill_ratio
        raw.append((score, x, y, w, h, hull, area))

    raw.sort(key=lambda t: t[0], reverse=True)
    if not raw:
        return image_bgr.copy(), []

    boxes = [(t[1], t[2], t[3], t[4]) for t in raw]
    scores = [t[0] for t in raw]
    keep_idx = _nms_xywh(boxes, scores, NMS_IOU)
    kept = [raw[i] for i in keep_idx[:MAX_DETECTIONS]]

    annotated = image_bgr.copy()
    detections: List[Detection] = []

    max_score = kept[0][0] if kept else 1.0
    for score, x, y, w, h, hull, area in kept:
        crop_bgr = image_bgr[y : y + h, x : x + w]
        if crop_bgr.size == 0:
            continue

        crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        color = _dominant_color(crop_hsv)

        det = Detection(
            bbox=(x, y, w, h),
            color=color,
            crop=crop_bgr,
            area=area,
            contour=hull,
            score=score,
            confidence=min(1.0, score / max_score),
        )
        detections.append(det)

        box_color = (0, 0, 220) if color == "red" else (220, 80, 0)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 3)
        cv2.drawContours(annotated, [hull], -1, box_color, 2)

    return annotated, detections


def annotate_with_labels(
    annotated: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
    output = annotated.copy()
    for det in detections:
        if not det.label:
            continue
        x, y, w, h = det.bbox
        text = f"{det.label}"
        if det.confidence > 0:
            text += f"  {det.confidence * 100:.1f}%"
        color = (0, 0, 220) if det.color == "red" else (220, 80, 0)
        cv2.putText(
            output,
            text,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return output
