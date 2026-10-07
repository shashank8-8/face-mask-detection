# Face Mask Detection using Transfer Learning

## Project Description

A real-time face mask detection system using transfer learning with **MobileNetV2**, **TensorFlow/Keras**, and **OpenCV**.

---

## Team Members

1. Sharanaprasad
2. Sumedh
3. Shashank
4. Vishnu

---

## Technologies

- Python 3.13
- TensorFlow & Keras
- MobileNetV2
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Project Pipeline

Dataset (`FMD_DATASET`)
→ Inspection & Cleaning
→ Balanced 80/20 Preprocessing
→ MobileNetV2 Transfer Learning
→ Binary Classification (Mask / No Mask)
→ Real-Time OpenCV Webcam & Web Dashboard

---

## 1. Dataset Inspection Summary

- **Source Archive**: `FMD_DATASET.zip` (14,536 total images)
- **Original Classes & Hierarchy**:
  - `incorrect_mask`: 5,000 images (`mc`: 2,500, `mmc`: 2,500)
  - `with_mask`: 4,789 images (`simple`: 4,000, `complex`: 789)
  - `without_mask`: 4,747 images (`simple`: 4,000, `complex`: 747)
- **Image Formats**: All `.jpg` extension (Internal bitstreams: 12,889 JPEG, 1,647 PNG)
- **Corrupted / Unreadable Images**: 0
- **Non-Image Files**: 0
- **Image Resolutions**:
  - Minimum: 26 × 37 px
  - Maximum: 4912 × 5412 px
  - Average: 634 × 640 px
  - Most common resolution: 1024 × 1024 px (7,593 images)

---

## 2. Directory Structure

The prepared dataset is organized in `processed_dataset/` as follows:

```
processed_dataset/
├── train/
│   ├── mask/       (3,797 images)
│   └── no_mask/    (3,797 images)
└── validation/
    ├── mask/       (950 images)
    └── no_mask/    (950 images)
```

- **Target Classes**: Binary classification (`mask` and `no_mask`).
- **3rd Class Handling**: `incorrect_mask` is excluded as agreed to maintain strict binary ground truth.
- **Split Ratio**: 80% Training / 20% Validation.
- **Balance**: Exactly balanced across both classes (4,747 images each; 3,797 train + 950 val).
- **Original Image Preservation**: Original images are never altered or permanently resized.

---

## 3. Scripts

- **[inspect_dataset.py](file:///c:/Users/SHASHANK%20RN/PB%20mask%20detector/PB_Project/inspect_dataset.py)**: Scans the raw dataset archive, detects file types, checks for corrupt images, and reports resolutions and class distribution.
- **[preprocess_dataset.py](file:///c:/Users/SHASHANK%20RN/PB%20mask%20detector/PB_Project/preprocess_dataset.py)**: Filters classes, validates images, balances sample counts, splits into 80/20 train/validation sets, and copies images into the directory structure.
- **[train.py](file:///c:/Users/SHASHANK%20RN/PB%20mask%20detector/PB_Project/train.py)**: MobileNetV2 transfer learning training pipeline.

---

## 4. Model Training & Evaluation (`train.py`)

### Architecture
- **Base Model**: MobileNetV2 pre-trained on ImageNet (weights frozen).
- **Data Augmentation**: Random horizontal flip, random rotation (15%), random zoom (10%).
- **Head**: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(2, Softmax).
- **Loss & Optimizer**: Categorical Crossentropy, Adam optimizer ($lr = 10^{-4}$).
- **Callbacks**:
  - `ModelCheckpoint`: Automatically saves the best model weights on validation accuracy to `model/mask_detector.keras`.
  - `EarlyStopping`: Prevents overfitting by monitoring `val_loss` with patience of 3 epochs.

### How to Run

```bash
# Default (10 epochs, batch size 32)
python train.py

# Custom epochs or batch size
python train.py --epochs 15 --batch-size 32
```

### Outputs
- **Model File**: `model/mask_detector.keras`
- **Evaluation Reports**:
  - `reports/training_curves.png`: Training vs. validation accuracy and loss graphs
  - `reports/confusion_matrix.png`: Confusion matrix plot
  - `reports/classification_report.txt`: Precision, Recall, and F1-score breakdown

---

## 5. Real-Time & Image Inference Scripts

Once `train.py` finishes saving `model/mask_detector.keras`, you can immediately test detection using either:

### 1. Real-Time Webcam Detection
```bash
python detect_mask_video.py
```
- Automatically accesses your webcam (`index 0`).
- Detects faces in real-time using OpenCV.
- Displays green boxes (`Mask`) and red boxes (`No Mask`) with confidence percentages.
- Press **'q'** or **ESC** to stop.

### 2. Static Image Testing
```bash
python detect_mask_image.py --image path/to/sample.jpg --output result.jpg --show
```
- Annotates faces with bounding boxes and saves the result to `result.jpg`.
