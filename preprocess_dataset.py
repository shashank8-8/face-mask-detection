"""
Dataset Preprocessing Script for Face Mask Detection
=====================================================
This script prepares a clean, balanced, 2-class dataset for training a
Face Mask vs No Mask classification model using MobileNetV2 and TensorFlow.

Requirements satisfied:
1. Two target classes: 'mask' and 'no_mask'
2. Excludes the 3rd class ('incorrect_mask')
3. Verifies image integrity and skips corrupted / unreadable images
4. Preserves the original Kaggle archive without altering original images
5. Generates the standard structure:
     processed_dataset/
       train/
         mask/
         no_mask/
       validation/
         mask/
         no_mask/
6. 80% training / 20% validation split
7. Balances class sample counts
8. Generates a summary report of final counts
"""

import os
import io
import shutil
import random
import zipfile
from PIL import Image

# ----------------- Configuration -----------------
SOURCE_ZIP_PATH = os.path.expanduser(r"~\Downloads\FMD_DATASET.zip")
OUTPUT_DIR = "processed_dataset"
RANDOM_SEED = 42
TRAIN_RATIO = 0.80  # 80% train, 20% validation
# -------------------------------------------------


def prepare_directories(base_dir):
    """Creates the destination directory structure."""
    splits = ["train", "validation"]
    classes = ["mask", "no_mask"]

    if os.path.exists(base_dir):
        print(f"Cleaning existing output directory '{base_dir}'...")
        shutil.rmtree(base_dir)

    for split in splits:
        for cls in classes:
            dir_path = os.path.join(base_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)
    print(f"Created folder hierarchy under '{base_dir}/'")


def is_image_valid(data):
    """
    Validates whether the raw byte stream represents a valid, readable image.
    Returns True if valid, False if corrupted.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            img.load()
        return True
    except Exception:
        return False


def preprocess():
    print("=" * 60)
    print("FACE MASK DATASET PREPROCESSING")
    print("=" * 60)

    if not os.path.exists(SOURCE_ZIP_PATH):
        raise FileNotFoundError(f"Source dataset zip not found at: {SOURCE_ZIP_PATH}")

    random.seed(RANDOM_SEED)

    prepare_directories(OUTPUT_DIR)

    original_total_count = 0
    invalid_count = 0

    # Buckets for valid images
    mask_items = []     # from with_mask
    no_mask_items = []  # from without_mask

    print(f"\nReading archive: {SOURCE_ZIP_PATH}")
    with zipfile.ZipFile(SOURCE_ZIP_PATH, 'r') as z:
        infolist = z.infolist()
        total_in_zip = len(infolist)

        print(f"Scanning and validating {total_in_zip} entries...")
        for idx, info in enumerate(infolist):
            if info.is_dir():
                continue

            original_total_count += 1
            filename = info.filename
            parts = filename.split('/')
            category = parts[0]

            # We only keep 'with_mask' and 'without_mask'
            if category not in ['with_mask', 'without_mask']:
                continue

            raw_bytes = z.read(info)

            # Check for corruption
            if not is_image_valid(raw_bytes):
                invalid_count += 1
                continue

            item = {
                "name": info.filename,
                "bytes": raw_bytes,
                "category": category
            }

            if category == 'with_mask':
                mask_items.append(item)
            elif category == 'without_mask':
                no_mask_items.append(item)

            if (idx + 1) % 2500 == 0:
                print(f"  Scanned {idx + 1}/{total_in_zip} entries...")

    print(f"\nInitial valid images found:")
    print(f"  - with_mask: {len(mask_items)}")
    print(f"  - without_mask: {len(no_mask_items)}")

    # Balance the two classes
    balanced_count = min(len(mask_items), len(no_mask_items))
    print(f"\nBalancing classes to {balanced_count} images per class...")

    random.shuffle(mask_items)
    random.shuffle(no_mask_items)

    final_mask = mask_items[:balanced_count]
    final_no_mask = no_mask_items[:balanced_count]

    # Calculate train/validation split indices
    train_count_per_class = int(balanced_count * TRAIN_RATIO)
    val_count_per_class = balanced_count - train_count_per_class

    mask_train = final_mask[:train_count_per_class]
    mask_val = final_mask[train_count_per_class:]

    no_mask_train = final_no_mask[:train_count_per_class]
    no_mask_val = final_no_mask[train_count_per_class:]

    splits_data = [
        ("train", "mask", mask_train),
        ("train", "no_mask", no_mask_train),
        ("validation", "mask", mask_val),
        ("validation", "no_mask", no_mask_val)
    ]

    print("\nWriting images to destination folders...")
    for split_name, class_name, items in splits_data:
        dest_folder = os.path.join(OUTPUT_DIR, split_name, class_name)
        for i, item in enumerate(items, 1):
            out_filename = f"{class_name}_{i:05d}.jpg"
            out_path = os.path.join(dest_folder, out_filename)
            with open(out_path, 'wb') as f:
                f.write(item["bytes"])

    total_training = len(mask_train) + len(no_mask_train)
    total_validation = len(mask_val) + len(no_mask_val)
    total_final_mask = len(final_mask)
    total_final_no_mask = len(final_no_mask)

    # Final Report
    print("\n" + "=" * 60)
    print("FINAL PREPROCESSING REPORT")
    print("=" * 60)
    print(f"* Original image count        : {original_total_count}")
    print(f"* Invalid/corrupted image count: {invalid_count}")
    print(f"* Final mask count            : {total_final_mask}")
    print(f"* Final no_mask count         : {total_final_no_mask}")
    print(f"* Training count              : {total_training} ({len(mask_train)} mask, {len(no_mask_train)} no_mask)")
    print(f"* Validation count            : {total_validation} ({len(mask_val)} mask, {len(no_mask_val)} no_mask)")
    print("=" * 60)
    print(f"Processed dataset successfully saved to: '{os.path.abspath(OUTPUT_DIR)}'\n")


if __name__ == '__main__':
    preprocess()
