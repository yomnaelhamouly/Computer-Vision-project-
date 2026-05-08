# ============================================================
# IoU EVALUATION SCRIPT
# ============================================================
# This script evaluates traffic sign detection performance
# using:
#
#       Intersection over Union (IoU)
#
# IoU measures how much the predicted bounding box overlaps
# with the ground truth bounding box.
#
# Formula:
#
#            Intersection Area
# IoU = -------------------------------
#        Union Area
#
# Higher IoU = better detection quality
#
# Typical interpretation:
#   IoU = 1.0   -> Perfect overlap
#   IoU > 0.5   -> Good detection
#   IoU < 0.3   -> Weak detection
#
# Inputs:
#   1. detection_results_eval.csv
#      -> predicted bounding boxes
#
#   2. pseudo_ground_truth.csv
#      -> pseudo ground truth boxes
#
# Output:
#   Average IoU score across all images
# ============================================================


# ============================================================
# IMPORT LIBRARY
# ============================================================

import pandas as pd     # Used for CSV loading and processing


# ============================================================
# FILE PATHS
# ============================================================

# CSV containing predicted bounding boxes
pred_path = "detections_v2/detection_results_eval.csv"

# CSV containing pseudo ground truth boxes
gt_path = "detections_v2/pseudo_ground_truth.csv"


# ============================================================
# LOAD CSV FILES
# ============================================================

# Read prediction results
pred_df = pd.read_csv(pred_path)

# Read pseudo ground truth annotations
gt_df = pd.read_csv(gt_path)


# ============================================================
# IoU FUNCTION
# ============================================================

def calculate_iou(boxA, boxB):

    """
    Calculates Intersection over Union (IoU)
    between two bounding boxes.

    Each box format:
        (x, y, w, h)

    where:
        x = left coordinate
        y = top coordinate
        w = width
        h = height
    """


    # ========================================================
    # INTERSECTION COORDINATES
    # ========================================================

    # Left boundary of overlap
    xA = max(boxA[0], boxB[0])

    # Top boundary of overlap
    yA = max(boxA[1], boxB[1])

    # Right boundary of overlap
    xB = min(boxA[0] + boxA[2],
             boxB[0] + boxB[2])

    # Bottom boundary of overlap
    yB = min(boxA[1] + boxA[3],
             boxB[1] + boxB[3])


    # ========================================================
    # INTERSECTION WIDTH & HEIGHT
    # ========================================================

    # Prevent negative width
    inter_w = max(0, xB - xA)

    # Prevent negative height
    inter_h = max(0, yB - yA)


    # ========================================================
    # INTERSECTION AREA
    # ========================================================

    inter_area = inter_w * inter_h


    # ========================================================
    # AREA OF EACH BOX
    # ========================================================

    # Predicted box area
    areaA = boxA[2] * boxA[3]

    # Ground truth box area
    areaB = boxB[2] * boxB[3]


    # ========================================================
    # UNION AREA
    # ========================================================

    # Union =
    # Area A + Area B - Intersection
    union = areaA + areaB - inter_area


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    # Prevent division by zero
    if union == 0:
        return 0


    # ========================================================
    # FINAL IoU
    # ========================================================

    return inter_area / union


# ============================================================
# EVALUATE ALL IMAGES
# ============================================================

# Store IoU scores for all images
ious = []


# Loop through all rows
for i in range(len(pred_df)):

    # ========================================================
    # GET CURRENT ROWS
    # ========================================================

    pred_row = pred_df.iloc[i]
    gt_row = gt_df.iloc[i]


    # ========================================================
    # CREATE BOX TUPLES
    # ========================================================

    # Predicted bounding box
    pred_box = (
        pred_row["x"],
        pred_row["y"],
        pred_row["w"],
        pred_row["h"]
    )

    # Ground truth bounding box
    gt_box = (
        gt_row["x"],
        gt_row["y"],
        gt_row["w"],
        gt_row["h"]
    )


    # ========================================================
    # COMPUTE IoU
    # ========================================================

    iou = calculate_iou(pred_box, gt_box)

    # Store result
    ious.append(iou)


# ============================================================
# COMPUTE FINAL AVERAGE IoU
# ============================================================

average_iou = sum(ious) / len(ious)


# ============================================================
# PRINT RESULT
# ============================================================

print("Average IoU:", average_iou)


# ============================================================
# NOTES
# ============================================================
#
# This evaluation uses pseudo ground truth boxes,
# not manually annotated boxes.
#
# Therefore:
#   - IoU values may be lower than real detection datasets
#   - Results represent approximate localization quality
# ============================================================