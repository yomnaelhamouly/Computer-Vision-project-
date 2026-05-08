# ============================================================
# PSEUDO GROUND TRUTH GENERATION
# ============================================================
# This script creates artificial (pseudo) bounding boxes
# for traffic sign images.
#
# Since the dataset does not contain real object detection
# annotations, we generate approximate bounding boxes
# automatically.
#
# The generated boxes are saved into:
#     pseudo_ground_truth.csv
#
# Each row contains:
#     image_path, x, y, w, h
#
# where:
#     x, y = top-left corner
#     w, h = width and height of bounding box
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os          # File and folder operations
import csv         # CSV file handling
import cv2         # OpenCV image reading


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing processed dataset
INPUT_DIR = "processed_dataset"

# Folder where output CSV will be saved
OUTPUT_DIR = "detections_v2"

# Output CSV filename
CSV_NAME = "pseudo_ground_truth.csv"

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# CHECK IF FILE IS IMAGE
# ============================================================

def is_image_file(filename):

    """
    Returns True if file extension belongs to an image.
    """

    return os.path.splitext(filename.lower())[1] in IMAGE_EXTENSIONS


# ============================================================
# CREATE DIRECTORY IF NOT EXISTS
# ============================================================

def ensure_dir(path):

    """
    Safely creates folder if it does not exist.
    """

    os.makedirs(path, exist_ok=True)


# ============================================================
# GET ALL IMAGES FROM SPLIT
# ============================================================

def get_all_images(split_dir):

    """
    Recursively scans all folders and collects image paths.

    This allows processing images inside:
        train/
        val/
        test/

    including all class subfolders.
    """

    image_paths = []

    # Walk through all folders recursively
    for root, _, files in os.walk(split_dir):

        for file in files:

            # Keep only image files
            if is_image_file(file):

                image_paths.append(
                    os.path.join(root, file)
                )

    # Sort paths for consistency
    return sorted(image_paths)


# ============================================================
# CREATE PSEUDO BOUNDING BOX
# ============================================================

def create_pseudo_box(image_w, image_h):

    """
    Creates approximate bounding box for traffic sign.

    Assumption:
        Traffic sign occupies central area of image.

    Bounding box:
        - starts at 15% from top-left
        - covers 70% of image width
        - covers 70% of image height

    This is called:
        Pseudo Ground Truth

    because the boxes are estimated automatically
    rather than manually annotated.
    """

    # Top-left x coordinate
    x = int(image_w * 0.15)

    # Top-left y coordinate
    y = int(image_h * 0.15)

    # Bounding box width
    w = int(image_w * 0.70)

    # Bounding box height
    h = int(image_h * 0.70)

    return x, y, w, h


# ============================================================
# PROCESS DATA SPLIT
# ============================================================

def process_split(split_name, writer):

    """
    Processes one dataset split:
        - train
        - val
        - test

    Steps:
        1. Read images
        2. Generate pseudo bounding boxes
        3. Save coordinates into CSV
    """

    # Full path of split folder
    input_split_dir = os.path.join(
        INPUT_DIR,
        split_name
    )


    # ========================================================
    # CHECK IF SPLIT EXISTS
    # ========================================================

    if not os.path.isdir(input_split_dir):

        print(f"Skipped missing folder: {input_split_dir}")

        return


    # ========================================================
    # GET IMAGE PATHS
    # ========================================================

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
        # GET IMAGE SIZE
        # ====================================================

        image_h, image_w = image.shape[:2]


        # ====================================================
        # GENERATE PSEUDO BOX
        # ====================================================

        x, y, w, h = create_pseudo_box(
            image_w,
            image_h
        )


        # ====================================================
        # CREATE RELATIVE PATH
        # ====================================================

        relative_path = os.path.relpath(
            input_path,
            input_split_dir
        )

        # Convert Windows "\" to "/"
        csv_image_path = (
            f"{split_name}/"
            f"{relative_path.replace(os.sep, '/')}"
        )


        # ====================================================
        # SAVE CSV ROW
        # ====================================================

        writer.writerow([
            csv_image_path,
            x,
            y,
            w,
            h
        ])


        # Print result for monitoring
        print(
            f"Pseudo GT: {csv_image_path} "
            f"-> x={x}, y={y}, w={w}, h={h}"
        )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    """
    Main execution function.
    """

    # Create output folder
    ensure_dir(OUTPUT_DIR)

    # Full CSV output path
    csv_path = os.path.join(
        OUTPUT_DIR,
        CSV_NAME
    )


    # ========================================================
    # OPEN CSV FILE
    # ========================================================

    with open(
        csv_path,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as file:

        # CSV writer object
        writer = csv.writer(file)


        # ====================================================
        # CSV HEADER
        # ====================================================

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

    print(
        f"Pseudo Ground Truth saved in: {csv_path}"
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

# Runs only when executing script directly
if __name__ == "__main__":

    main()