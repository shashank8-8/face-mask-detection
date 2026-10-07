"""
Flask Web Application for Face Mask Detection
==============================================
Provides a rich browser dashboard with:
- Drag-and-drop static image classification
- Live browser webcam streaming detection
- Evaluation metrics and training charts display
- 1-click sample image testing
"""

import os
import time
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------- Configuration ----------------
MODEL_PATH = "model/mask_detector.keras"
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
REPORTS_DIR = "reports"
INPUT_SIZE = (224, 224)
# -----------------------------------------------

# Global holders for model & cascade
MODEL = None
FACE_CASCADE = None


def init_detector():
    global MODEL, FACE_CASCADE
    if MODEL is None and os.path.exists(MODEL_PATH):
        print(f"Loading Keras model from '{MODEL_PATH}'...")
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        # Warmup model
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        MODEL.predict(dummy, verbose=0)
        print("Model loaded and warmed up.")
    if FACE_CASCADE is None and os.path.exists(CASCADE_PATH):
        print("Loading Haar cascade detector...")
        FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)


def process_cv2_image(cv_image):
    """Detects faces and predicts mask status on a BGR image."""
    init_detector()
    if cv_image is None or MODEL is None or FACE_CASCADE is None:
        return cv_image, []

    h_img, w_img = cv_image.shape[:2]
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(30, 30)
    )

    # If no face is detected by Haar cascade, treat the full image as a single portrait ROI
    if len(faces) == 0:
        faces = np.array([[0, 0, w_img, h_img]])

    detections = []
    annotated = cv_image.copy()

    for (x, y, w, h) in faces:
        face_roi = cv_image[y:y + h, x:x + w]
        if face_roi.size == 0:
            continue

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, INPUT_SIZE)
        face_array = np.expand_dims(face_resized, axis=0)

        pred = MODEL.predict(face_array, verbose=0)[0]
        mask_prob = float(pred[0])
        no_mask_prob = float(pred[1])

        is_mask = mask_prob > no_mask_prob
        confidence = mask_prob if is_mask else no_mask_prob
        label = "Mask" if is_mask else "No Mask"
        color = (0, 230, 115) if is_mask else (30, 40, 235)  # Neon Green vs Vibrant Red

        detections.append({
            "box": [int(x), int(y), int(w), int(h)],
            "label": label,
            "confidence": round(confidence * 100, 2),
            "mask_prob": round(mask_prob * 100, 2),
            "no_mask_prob": round(no_mask_prob * 100, 2)
        })

        # Draw box and styled banner
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        label_text = f"{label}: {confidence * 100:.1f}%"
        label_y = y - 12 if y - 12 > 12 else y + 25
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (x, label_y - th - 6), (x + tw + 6, label_y + baseline + 2), color, cv2.FILLED)
        cv2.putText(annotated, label_text, (x + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return annotated, detections


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory(REPORTS_DIR, filename)


@app.route("/predict_image", methods=["POST"])
def predict_image():
    t_start = time.time()
    cv_img = None

    # Case 1: Sample image requested
    sample_type = request.form.get("sample")
    if sample_type:
        if sample_type == "mask":
            path = "processed_dataset/validation/mask/mask_00001.jpg"
        else:
            path = "processed_dataset/validation/no_mask/no_mask_00001.jpg"
        if os.path.exists(path):
            cv_img = cv2.imread(path)

    # Case 2: Uploaded file
    if cv_img is None and "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if cv_img is None:
        return jsonify({"error": "No valid image provided"}), 400

    annotated, detections = process_cv2_image(cv_img)

    # Encode annotated image to Base64
    _, buffer = cv2.imencode(".jpg", annotated)
    b64_img = base64.b64encode(buffer).decode("utf-8")
    infer_time = round((time.time() - t_start) * 1000, 1)

    return jsonify({
        "success": True,
        "inference_time_ms": infer_time,
        "total_faces": len(detections),
        "detections": detections,
        "image_data": f"data:image/jpeg;base64,{b64_img}"
    })


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """Receives a webcam video frame as base64 and returns detections."""
    t_start = time.time()
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No frame data"}), 400

    header, encoded = data["image"].split(",", 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Frame decode failed"}), 400

    _, detections = process_cv2_image(frame)
    infer_time = round((time.time() - t_start) * 1000, 1)

    return jsonify({
        "success": True,
        "inference_time_ms": infer_time,
        "detections": detections
    })


@app.route("/api/stats")
def get_stats():
    return jsonify({
        "model_name": "MobileNetV2 (Transfer Learning)",
        "accuracy": "98.74%",
        "val_samples": 1900,
        "train_samples": 7594,
        "classes": ["mask", "no_mask"],
        "tflite_size_mb": "2.55 MB (Quantized)",
        "keras_size_mb": "11.07 MB"
    })


if __name__ == "__main__":
    init_detector()
    print("\nStarting Face Mask Detection Web Dashboard on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
