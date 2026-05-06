# 🚦 Traffic Sign Detection & Recognition

> A full end-to-end Computer Vision pipeline for detecting, segmenting, and classifying traffic signs using the **GTSRB** dataset — from raw image collection through model evaluation.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Pipeline Architecture](#-pipeline-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Project Team](#-project-team)

---

## 🔍 Project Overview

This project implements a comprehensive Computer Vision pipeline for **traffic sign detection and recognition**. Using the German Traffic Sign Recognition Benchmark (**GTSRB**) dataset — which contains **43 distinct traffic sign categories** — the system progresses through every stage of a production-ready vision pipeline:

1. Raw data collection and rigorous cleaning
2. Image preprocessing and augmentation
3. Color-based object detection with bounding-box generation and segmentation masks
4. Handcrafted feature engineering (HOG + color histograms)
5. Classification with both classical ML (SVM) and deep learning (CNN)
6. Quantitative evaluation with standard metrics and visual plots

The goal is to build a robust, reproducible system capable of accurately identifying traffic signs under varied real-world conditions such as different lighting, viewing angles, and scales.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧹 **Data Cleaning** | Removes corrupted, duplicate, and unreadable images; verifies class completeness |
| ⚖️ **Dataset Balancing** | Creates a perfectly balanced subset of 200 images per class across all 43 categories |
| ✂️ **Train/Val/Test Split** | Stratified 70 / 15 / 15 split ensuring no class bias |
| 🖼️ **Preprocessing** | Resizing (64×64), Gaussian blur, CLAHE contrast enhancement, BGR→RGB conversion |
| 🔄 **Data Augmentation** | Random horizontal flip, rotation (±15°), and brightness variation on training images |
| 🟥 **Object Detection** | HSV color-space masking for red and blue signs; contour-based bounding-box extraction |
| 🎭 **Segmentation & Masks** | Binary segmentation masks saved alongside annotated images for every split |
| 📐 **IoU Evaluation** | Intersection over Union computed against pseudo ground-truth bounding boxes |
| 🔬 **Feature Extraction** | HOG descriptors and color histogram feature vectors via scikit-image |
| 🤖 **Classification** | Dual-model comparison: Support Vector Machine (SVM) vs. Convolutional Neural Network (CNN) |
| 📊 **Evaluation Metrics** | Accuracy, Precision, Recall, confusion matrix, and result plots |

---

## 🏗️ Pipeline Architecture

```
Raw GTSRB Dataset
       │
       ▼
┌─────────────────┐
│  Data Cleaning  │  ← corruption removal, deduplication, class balancing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │  ← resize, blur, CLAHE, augmentation
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Object Detection   │  ← HSV masking, bounding boxes, segmentation masks, IoU
└────────┬────────────┘
         │
         ▼
┌────────────────────┐
│ Feature Extraction │  ← HOG descriptors, color features, feature vectors
└────────┬───────────┘
         │
         ▼
┌──────────────────────────┐
│ Classification (SVM/CNN) │  ← model training, prediction generation
└────────┬─────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Evaluation & Visualization  │  ← Accuracy, Precision, Recall, confusion matrix
└─────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Core Language
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)

### Computer Vision & Image Processing
| Library | Usage |
|---|---|
| **OpenCV** (`cv2`) | Image I/O, resizing, color-space conversion (BGR/RGB/HSV/LAB), Gaussian blur, CLAHE, morphological ops, contour detection, bounding-box drawing |
| **Pillow** (`PIL`) | Image validation and corruption detection |
| **scikit-image** | HOG (Histogram of Oriented Gradients) feature extraction |

### Data Science & Machine Learning
| Library | Usage |
|---|---|
| **NumPy** | Array manipulation, mask operations, numerical computation |
| **Pandas** | Dataset metadata management, CSV I/O, result logging |
| **scikit-learn** | SVM classifier (`SVC`), `LabelEncoder`, `StandardScaler`, evaluation metrics |
| **TensorFlow / Keras** | CNN model architecture, GPU-accelerated training (T4 on Google Colab) |

### Visualization
| Library | Usage |
|---|---|
| **Matplotlib** | Image display, class distribution plots, result visualization |
| **Seaborn** | Confusion matrix heatmaps, metric comparison plots |

### Utilities
| Library | Usage |
|---|---|
| **tqdm** | Progress bars for preprocessing loops |
| **hashlib** | MD5-based duplicate image detection |
| **shutil / os** | File system operations, dataset restructuring |

---

## 📁 Project Structure

```
Computer-Vision-project-/
│
├── Data/
│   ├── Raw/
│   │   ├── GTSRB/               # Original GTSRB dataset (Train / Test / Meta)
│   │   ├── balanced_dataset/    # 200 images per class (43 classes)
│   │   └── dataset/             # Stratified split (train / val / test)
│   └── Processed/               # Preprocessed images ready for modeling
│
├── Notebooks/
│   ├── Classification_and_Evaluation_Metrics.ipynb   # SVM vs CNN training & metrics
│   └── FeatureExtraction.ipynb                       # HOG & color feature engineering
│
├── Outputs/
│   ├── Classification_and_Evaluation/
│   │   ├── feature_scaler.pkl   # Fitted StandardScaler
│   │   ├── label_encoder.pkl    # Fitted LabelEncoder
│   │   └── predictions.csv      # Model predictions output
│   ├── DataCleaning/
│   │   ├── dataset_metadata.csv # Full dataset path-label manifest
│   │   └── labels.csv           # Class label reference
│   └── Detection/
│       ├── detection_results_eval.csv   # Predicted bounding boxes
│       └── pseudo_ground_truth.csv      # Ground-truth boxes for IoU computation
│
├── Reports/
│   └── datasetReport.txt        # Full dataset summary and quality report
│
├── src/
│   ├── DatasetCleaning/
│   │   └── datasetCleaning.py   # Cleaning, balancing & splitting pipeline
│   ├── DatasetPreprocess/
│   │   └── preprocess.py        # Preprocessing & augmentation pipeline
│   └── Detection/
│       ├── object_detection.py  # HSV-based detection, mask generation
│       ├── pseudo_ground_truth.py
│       └── evaluation.py        # IoU calculation
│
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Name** | GTSRB (German Traffic Sign Recognition Benchmark) |
| **Total Images** | 8,600 |
| **Classes** | 43 traffic sign categories |
| **Images per Class** | 200 (perfectly balanced) |
| **Image Size** | 64 × 64 pixels |
| **Color Format** | RGB (3 channels) |
| **Training Set** | 6,020 images — 70% |
| **Validation Set** | 1,290 images — 15% |
| **Test Set** | 1,290 images — 15% |

> The dataset covers variations in lighting conditions, viewing angles, and distances to improve model generalization to real-world driving scenarios.

---

## 👥 Project Team

| # | Member | Role & Contribution |
|---|---|---|
| 1 | 👩‍💼 **Salma** | **Project Manager** — Organized the team, designed the full pipeline architecture, and coordinated all stages of delivery |
| 2 | 👩‍🔬 **Yomna** | **Data Collection & Splitting** — Sourced the GTSRB dataset, performed data cleaning, and executed the train/val/test split |
| 3 | 👨‍💻 **Abdelrahman** | **Preprocessing** — Applied image preprocessing techniques including resizing, normalization strategy, and noise reduction (Gaussian blur + CLAHE) |
| 4 | 👨‍🔭 **Mohamed Adel** | **Object Detection & Segmentation** — Implemented HSV-based object detection, generated bounding boxes, produced segmentation masks, and computed IoU scores |
| 5 | 👨‍🔬 **Abdelrahman** | **Feature Engineering** — Extracted color histograms and HOG shape descriptors; assembled feature vectors for downstream classification |
| 6 | 👩‍🤖 **Rawan** | **Model Training & Classification** — Trained SVM and CNN classifiers on extracted features and generated model predictions |
| 7 | 📈 **Mohammed Saied** | **Model Evaluation** — Computed Accuracy, Precision, and Recall metrics; plotted confusion matrices and performance visualizations |
| 8 | 📝 **Malak** | **Documentation** — Responsible for project documentation, reporting, and maintaining the dataset quality report |

---

<div align="center">

**Built with ❤️ by the Computer Vision Team**

</div>
