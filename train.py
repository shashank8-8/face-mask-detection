"""
Face Mask Detection - MobileNetV2 Training Script
==================================================
This script trains a high-accuracy, lightweight binary classifier
(Mask vs. No Mask) using MobileNetV2 transfer learning on the
preprocessed dataset.

Features:
- Transfer learning with ImageNet pre-trained MobileNetV2
- Data augmentation (flipping, slight rotation, zoom)
- Modern tf.data input pipeline with prefetching
- Callbacks: ModelCheckpoint (saves best weights) and EarlyStopping
- Comprehensive evaluation:
    * Accuracy and Loss curves plot
    * Confusion Matrix plot
    * Precision, Recall, and F1-score classification report
- Saves the trained model to 'model/mask_detector.keras'
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for Face Mask Detection")
    parser.add_argument("--data-dir", type=str, default="processed_dataset",
                        help="Path to preprocessed dataset directory")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training and validation (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate (default: 0.0001)")
    parser.add_argument("--output-dir", type=str, default="model",
                        help="Directory to save the trained model")
    parser.add_argument("--reports-dir", type=str, default="reports",
                        help="Directory to save plots and evaluation metrics")
    return parser.parse_args()


def load_datasets(data_dir, batch_size, img_size=(224, 224)):
    """Loads train and validation datasets from disk."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Dataset directories not found. Make sure '{data_dir}/train' "
            f"and '{data_dir}/validation' exist. Run preprocess_dataset.py first."
        )

    print("\n--- Loading Dataset ---")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="categorical",
        class_names=["mask", "no_mask"],
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode="categorical",
        class_names=["mask", "no_mask"],
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    )

    print(f"Classes: {train_ds.class_names}")

    # Optimize data pipeline with AUTOTUNE prefetching
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, ["mask", "no_mask"]


def build_model(img_size=(224, 224, 3)):
    """Constructs the MobileNetV2 transfer learning model."""
    print("\n--- Building MobileNetV2 Model ---")

    # Data augmentation block applied directly inside the model
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    # Load pre-trained MobileNetV2 without top fully-connected layers
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=img_size
    )
    # Freeze base model weights so we only train the new custom head
    base_model.trainable = False

    # Define model inputs and architecture
    inputs = tf.keras.Input(shape=img_size)
    x = data_augmentation(inputs)
    x = preprocess_input(x)  # Scales pixel values between [-1, 1] as MobileNetV2 expects
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation="softmax", name="classification_output")(x)

    model = tf.keras.Model(inputs, outputs, name="FaceMask_MobileNetV2")
    return model


def plot_metrics(history, save_dir):
    """Plots and saves accuracy and loss curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#2563eb", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#16a34a", linewidth=2, linestyle="--")
    plt.title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#dc2626", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#d97706", linewidth=2, linestyle="--")
    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    chart_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Saved training curves to: {chart_path}")


def evaluate_model(model, val_ds, class_names, save_dir):
    """Evaluates the model on validation data and prints classification report & confusion matrix."""
    print("\n--- Evaluating Model on Validation Set ---")
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # Save classification report to text file
    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12, fontweight="bold")

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")


def main():
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    # 1. Load data
    train_ds, val_ds, class_names = load_datasets(args.data_dir, args.batch_size)

    # 2. Build model
    model = build_model()
    model.summary()

    # 3. Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Callbacks
    best_model_path = os.path.join(args.output_dir, "mask_detector.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 5. Train
    print(f"\n--- Starting Training for {args.epochs} Epochs ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # 6. Save final model
    print(f"\nSaving final model to: {best_model_path}")
    model.save(best_model_path)

    # 7. Plots & Evaluation
    plot_metrics(history, args.reports_dir)
    evaluate_model(model, val_ds, class_names, args.reports_dir)

    print("\n" + "=" * 60)
    print("TRAINING & EVALUATION COMPLETE!")
    print(f"Model saved to: {best_model_path}")
    print(f"Reports saved to: {os.path.abspath(args.reports_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
