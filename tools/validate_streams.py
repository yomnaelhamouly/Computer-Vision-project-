"""
Quick import / smoke checks for utils used by the Streamlit app.
Run from project root:  python tools/validate_streams.py
"""

from __future__ import annotations


def main() -> None:
    import numpy as np

    from utils.detection import detect_sign_bbox, detect_signs, roi_for_feature_extraction_notebook
    from utils.svm_features import extract_training_style_features

    img = np.zeros((120, 160, 3), dtype=np.uint8)
    box, mask = detect_sign_bbox(img)
    assert mask.shape == (120, 160)
    assert box is None or len(box) == 4

    ann, dets = detect_signs(img)
    assert ann.shape == img.shape
    assert isinstance(dets, list)

    roi, note = roi_for_feature_extraction_notebook(img)
    assert roi.shape == img.shape
    assert "Full image" in note or "ROI" in note

    feats = extract_training_style_features(roi)
    assert feats.shape == (1, 2277)

    print("OK: detection + feature extraction shapes match notebooks project.")


if __name__ == "__main__":
    main()
