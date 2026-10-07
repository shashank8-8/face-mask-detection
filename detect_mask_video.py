"""
Real-Time Face Mask Detection via Webcam
=========================================
This script uses OpenCV to capture video frames, detects human faces,
and classifies each detected face as 'Mask' or 'No Mask' using the
trained MobileNetV2 model.

Controls:
- Press 'q' or 'ESC' to exit the video window.
"""

import os
import cv2
import numpy as np
import tensorflow as tf

# ----------------- Configuration -----------------
MODEL_PATH = "model/mask_detector.keras"
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
CONFIDENCE_THRESHOLD = 0.5
INPUT_IMAGE_SIZE = (224, 224)
# -------------------------------------------------


def load_model_safely(model_path):
    """Loads the trained Keras model, checking for file existence."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please ensure training has completed and the model file exists."
        )
    print(f"Loading trained model from '{model_path}'...")
    return tf.keras.models.load_model(model_path)


def detect_and_predict_mask(frame, face_cascade, model):
    """
    Detects faces in the frame and predicts whether a mask is worn.
    Returns:
      locs: list of (x, y, w, h) bounding boxes
      preds: list of [prob_mask, prob_no_mask]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    locs = []
    preds = []

    for (x, y, w, h) in faces:
        # Extract face ROI (Region of Interest)
        face_roi = frame[y:y + h, x:x + w]
        if face_roi.size == 0:
            continue

        # Convert BGR (OpenCV) to RGB (TensorFlow model expectation)
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, INPUT_IMAGE_SIZE)
        face_array = np.expand_dims(face_resized, axis=0)

        # Predict
        prediction = model.predict(face_array, verbose=0)[0]
        
        locs.append((x, y, w, h))
        preds.append(prediction)

    return locs, preds


def main():
    print("=" * 60)
    print("REAL-TIME FACE MASK DETECTION")
    print("=" * 60)

    # 1. Load Face Cascade
    if not os.path.exists(CASCADE_PATH):
        raise FileNotFoundError(f"Haar cascade XML not found at {CASCADE_PATH}")
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    print(f"Face detector loaded: {os.path.basename(CASCADE_PATH)}")

    # 2. Load Mask Classifier
    model = load_model_safely(MODEL_PATH)
    print("Model loaded successfully!")

    # 3. Open Video Stream
    print("\nStarting video stream (Webcam index 0)...")
    print("Press 'q' or 'ESC' to exit.\n")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam (index 0). If you have multiple cameras, try index 1.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam. Exiting...")
            break

        # Detect faces and predict masks
        locs, preds = detect_and_predict_mask(frame, face_cascade, model)

        for (box, pred) in zip(locs, preds):
            (x, y, w, h) = box
            mask_prob, no_mask_prob = pred

            # Determine class label and bounding box color
            if mask_prob > no_mask_prob:
                label = f"Mask: {mask_prob * 100:.1f}%"
                color = (0, 255, 0)      # Green for Mask
            else:
                label = f"No Mask: {no_mask_prob * 100:.1f}%"
                color = (0, 0, 255)      # Red for No Mask

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw filled banner for text readability
            label_y = y - 10 if y - 10 > 10 else y + 20
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x, label_y - text_h - 4), (x + text_w, label_y + baseline), color, cv2.FILLED)
            cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("Face Mask Detector - Press 'q' to Quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video stream terminated.")


if __name__ == "__main__":
    main()
