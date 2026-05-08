# ============================================================
# TRAFFIC SIGN DETECTION PIPELINE
# ============================================================
# This script detects traffic signs using traditional
# computer vision techniques based on:
#   1. Color segmentation in HSV color space
#   2. Morphological filtering
#   3. Contour detection
#   4. Geometric filtering
#
# The detected bounding boxes are saved into:
#   - Annotated output images
#   - Binary masks
#   - CSV file for evaluation
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os                  # File and folder operations
import csv                 # CSV file handling
import cv2                 # OpenCV image processing
import numpy as np         # Numerical operations


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing processed images
INPUT_DIR = "processed_dataset"

# Folder where detection results will be saved
OUTPUT_DIR = "detections_v2"

# CSV filename for storing predicted bounding boxes
CSV_NAME = "detection_results_eval.csv"

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Minimum contour area threshold
# Small contours below this value are ignored as noise
MIN_AREA = 300


# ============================================================
# HSV COLOR RANGES
# ============================================================

# Red color appears in TWO ranges in HSV:
# around 0° and around 180°
# Therefore we need two separate ranges.

RED_RANGES = [
    ((0, 70, 50), (10, 255, 255)),      # Lower red range
    ((170, 70, 50), (180, 255, 255))    # Upper red range
]

# HSV range for blue traffic signs
BLUE_RANGE = ((90, 60, 40), (140, 255, 255))


# ============================================================
# CHECK IF FILE IS IMAGE
# ============================================================

def is_image_file(filename):
    """
    Returns True if the file extension is an image format.
    """
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTENSIONS


# ============================================================
# CREATE DIRECTORY IF IT DOES NOT EXIST
# ============================================================

def ensure_dir(path):
    """
    Creates folder safely if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


# ============================================================
# BUILD COLOR MASK
# ============================================================

def build_color_mask(hsv):

    """
    Creates binary mask containing:
        - Red traffic signs
        - Blue traffic signs

    Steps:
        1. Detect red regions
        2. Detect blue regions
        3. Combine masks
        4. Apply blur
        5. Apply morphology operations
    """

    # Empty mask for red colors
    red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # Detect red regions using both HSV ranges
    for lower, upper in RED_RANGES:

        red_mask |= cv2.inRange(
            hsv,
            np.array(lower, dtype=np.uint8),
            np.array(upper, dtype=np.uint8)
        )

    # Detect blue regions
    blue_mask = cv2.inRange(
        hsv,
        np.array(BLUE_RANGE[0], dtype=np.uint8),
        np.array(BLUE_RANGE[1], dtype=np.uint8)
    )

    # Combine red + blue masks
    combined = cv2.bitwise_or(red_mask, blue_mask)


    # ========================================================
    # MORPHOLOGICAL FILTERING
    # ========================================================

    # Small kernels used for morphology operations
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)

    # Gaussian blur smooths noisy regions
    combined = cv2.GaussianBlur(combined, (5, 5), 0)

    # MORPH_OPEN:
    # Removes tiny isolated noise pixels
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        kernel3
    )

    # MORPH_CLOSE:
    # Fills holes and connects fragmented regions
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        kernel5
    )

    return combined


# ============================================================
# DETECT TRAFFIC SIGN BOUNDING BOX
# ============================================================

def detect_sign_bbox(image):

    """
    Detects the best traffic sign candidate in the image.

    Pipeline:
        1. Convert image to HSV
        2. Build color mask
        3. Find contours
        4. Filter contours using geometry
        5. Select best contour
    """

    # Convert image from BGR → HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Build binary mask for red/blue signs
    mask = build_color_mask(hsv)

    # Find external contours only
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Get image dimensions
    image_h, image_w = image.shape[:2]

    # Total image area
    image_area = image_h * image_w

    # Best detection initialization
    best_box = None
    best_score = -1


    # ========================================================
    # PROCESS EACH CONTOUR
    # ========================================================

    for contour in contours:

        # Contour area
        area = cv2.contourArea(contour)

        # Ignore very small contours (noise)
        if area < MIN_AREA:
            continue


        # ====================================================
        # CONVEX HULL
        # ====================================================

        # Smooth contour shape
        contour = cv2.convexHull(contour)


        # ====================================================
        # BOUNDING RECTANGLE
        # ====================================================

        # Extract bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Rectangle area
        rect_area = w * h

        # Skip invalid boxes
        if rect_area <= 0:
            continue


        # ====================================================
        # GEOMETRIC FEATURES
        # ====================================================

        # Fill ratio:
        # Measures how well contour fills rectangle
        fill_ratio = area / rect_area

        # Aspect ratio:
        # Helps reject unrealistic shapes
        aspect_ratio = w / float(h)


        # ====================================================
        # GEOMETRIC FILTERING
        # ====================================================

        # Ignore giant detections covering almost entire image
        if rect_area > image_area * 0.9:
            continue

        # Ignore unrealistic shapes
        # Traffic signs are usually near square/circle
        if aspect_ratio < 0.6 or aspect_ratio > 1.8:
            continue


        # ====================================================
        # DETECTION SCORE
        # ====================================================

        # Good detection should:
        #   - be large
        #   - tightly filled
        score = rect_area * fill_ratio

        # Keep best candidate
        if score > best_score:
            best_score = score
            best_box = (x, y, w, h)

    # Return best detection + binary mask
    return best_box, mask


# ============================================================
# DRAW BOUNDING BOX
# ============================================================

def draw_box(image, box):

    """
    Draws green rectangle around detected sign.
    """

    output = image.copy()

    if box is not None:

        x, y, w, h = box

        # Green rectangle
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    return output


# ============================================================
# GET ALL IMAGES FROM SPLIT
# ============================================================

def get_all_images(split_dir):

    """
    Recursively scans all folders and collects image paths.
    """

    image_paths = []

    for root, _, files in os.walk(split_dir):

        for file in files:

            if is_image_file(file):

                image_paths.append(os.path.join(root, file))

    return sorted(image_paths)


# ============================================================
# PROCESS TRAIN / VAL / TEST SPLIT
# ============================================================

def process_split(split_name, writer):

    """
    Processes one dataset split:
        - detect signs
        - save masks
        - save annotated images
        - save CSV detections
    """

    input_split_dir = os.path.join(INPUT_DIR, split_name)

    output_split_dir = os.path.join(OUTPUT_DIR, split_name)

    mask_split_dir = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_masks"
    )

    # Create output folders
    ensure_dir(output_split_dir)
    ensure_dir(mask_split_dir)

    # Skip if folder missing
    if not os.path.isdir(input_split_dir):

        print(f"Skipped missing folder: {input_split_dir}")
        return

    # Collect all image paths
    image_paths = get_all_images(input_split_dir)


    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for input_path in image_paths:

        # Read image
        image = cv2.imread(input_path)

        # Skip unreadable images
        if image is None:

            print(f"Failed to read: {input_path}")
            continue


        # ====================================================
        # CREATE RELATIVE PATHS
        # ====================================================

        relative_path = os.path.relpath(
            input_path,
            input_split_dir
        )

        relative_dir = os.path.dirname(relative_path)

        filename = os.path.basename(relative_path)


        # ====================================================
        # CREATE OUTPUT FOLDERS
        # ====================================================

        save_image_dir = os.path.join(
            output_split_dir,
            relative_dir
        )

        save_mask_dir = os.path.join(
            mask_split_dir,
            relative_dir
        )

        ensure_dir(save_image_dir)
        ensure_dir(save_mask_dir)


        # ====================================================
        # DETECT TRAFFIC SIGN
        # ====================================================

        box, mask = detect_sign_bbox(image)

        # Draw green bounding box
        detected_image = draw_box(image, box)


        # ====================================================
        # SAVE OUTPUT FILES
        # ====================================================

        output_image_path = os.path.join(
            save_image_dir,
            filename
        )

        output_mask_path = os.path.join(
            save_mask_dir,
            filename
        )

        # Save detected image
        cv2.imwrite(output_image_path, detected_image)

        # Save binary mask
        cv2.imwrite(output_mask_path, mask)


        # ====================================================
        # SAVE CSV RESULT
        # ====================================================

        csv_image_path = f"{split_name}/{relative_path.replace(os.sep, '/')}"


        # If sign detected
        if box is not None:

            x, y, w, h = box

            writer.writerow([
                csv_image_path,
                x,
                y,
                w,
                h
            ])

            print(
                f"Detected: {csv_image_path} "
                f"-> x={x}, y={y}, w={w}, h={h}"
            )

        # If detection failed
        else:

            image_h, image_w = image.shape[:2]

            # Fallback = full image
            writer.writerow([
                csv_image_path,
                0,
                0,
                image_w,
                image_h
            ])

            print(f"No detection: {csv_image_path}")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    """
    Main execution function.
    """

    # Create output directory
    ensure_dir(OUTPUT_DIR)

    # Full CSV path
    csv_path = os.path.join(
        OUTPUT_DIR,
        CSV_NAME
    )

    # Open CSV file for writing
    with open(
        csv_path,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        # CSV header
        writer.writerow([
            "image_path",
            "x",
            "y",
            "w",
            "h"
        ])


        # ====================================================
        # PROCESS ALL DATA SPLITS
        # ====================================================

        for split_name in ["train", "val", "test"]:

            process_split(split_name, writer)


    # ========================================================
    # FINISHED
    # ========================================================

    print("Done")

    print(f"Results saved in: {OUTPUT_DIR}")

    print(f"CSV saved in: {csv_path}")


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()