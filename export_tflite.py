"""
Export Trained Model to TensorFlow Lite (TFLite)
================================================
This script converts the trained MobileNetV2 Keras model
('model/mask_detector.keras') into an optimized TFLite model
('model/mask_detector.tflite') suitable for low-latency edge devices
(such as Raspberry Pi, Android, iOS, and embedded microcontrollers).
"""

import os
import tensorflow as tf

MODEL_PATH = "model/mask_detector.keras"
TFLITE_OUTPUT_PATH = "model/mask_detector.tflite"
TFLITE_QUANT_PATH = "model/mask_detector_quant.tflite"


def export_tflite():
    print("=" * 60)
    print("EXPORTING TO TENSORFLOW LITE (TFLite)")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model '{MODEL_PATH}' not found!")

    print(f"Loading Keras model from '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 1. Standard FP32 TFLite conversion
    print("Converting model to standard TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    original_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    tflite_size_mb = os.path.getsize(TFLITE_OUTPUT_PATH) / (1024 * 1024)

    print(f"Standard TFLite model saved to: '{TFLITE_OUTPUT_PATH}'")
    print(f"  - Original Keras model size : {original_size_mb:.2f} MB")
    print(f"  - TFLite model size         : {tflite_size_mb:.2f} MB")

    # 2. Dynamic range quantized TFLite conversion (further reduces size for IoT)
    print("\nOptimizing with dynamic range quantization...")
    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter_quant.convert()

    with open(TFLITE_QUANT_PATH, "wb") as f:
        f.write(tflite_quant_model)

    quant_size_mb = os.path.getsize(TFLITE_QUANT_PATH) / (1024 * 1024)
    print(f"Quantized TFLite model saved to: '{TFLITE_QUANT_PATH}'")
    print(f"  - Quantized model size      : {quant_size_mb:.2f} MB")
    print(f"  - Compression ratio         : {original_size_mb / quant_size_mb:.1f}x smaller!")

    print("\n" + "=" * 60)
    print("TFLITE EXPORT COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    export_tflite()
