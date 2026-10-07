"""
Face Mask Detection on Static Images
====================================
This script takes an image file path as input, detects human faces,
and predicts whether each person is wearing a face mask.
The result is saved to disk and optionally displayed.

Usage:
  python detect_mask_image.py --image path/to/sample.jpg
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "model/mask_detector.keras"
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
INPUT_IMAGE_SIZE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(description="Run mask detection on an image file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--output", type=str, default="output_detected.jpg", help="Path to save annotated image")
    parser.add_argument("--show", action="store_true", help="Display result in a GUI window")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model '{MODEL_PATH}' not found. Please complete training first.")
        return

    print(f"Loading face detector and model...")
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image at '{args.image}'.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    print(f"Detected {len(faces)} face(s) in image.")

    for (x, y, w, h) in faces:
        face_roi = image[y:y + h, x:x + w]
        if face_roi.size == 0:
            continue

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, INPUT_IMAGE_SIZE)
        face_array = np.expand_dims(face_resized, axis=0)

        pred = model.predict(face_array, verbose=0)[0]
        mask_prob, no_mask_prob = pred

        if mask_prob > no_mask_prob:
            label = f"Mask: {mask_prob * 100:.1f}%"
            color = (0, 255, 0)
        else:
            label = f"No Mask: {no_mask_prob * 100:.1f}%"
            color = (0, 0, 255)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label_y = y - 10 if y - 10 > 10 else y + 20
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(image, (x, label_y - text_h - 4), (x + text_w, label_y + baseline), color, cv2.FILLED)
        cv2.putText(image, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imwrite(args.output, image)
    print(f"Annotated result saved to: {args.output}")

    if args.show:
        cv2.imshow("Detection Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
