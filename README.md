# Face Mask Detection using Transfer Learning

A face mask classification project using a pretrained MobileNetV2 model.
Includes training, validation plots, an OpenCV webcam demo, and the
MaskLab browser website.

## Live Demo

https://masklab-live.vishnuyadav-venkates.chatgpt.site

Wait for “Model ready”, start the camera and allow camera access.
You can also upload an image and save a prediction screenshot.

## Technologies

- Python 3.11
- TensorFlow/Keras
- MobileNetV2 pretrained on ImageNet
- OpenCV
- NumPy and Matplotlib
- HTML, CSS, JavaScript and TensorFlow.js
- BlazeFace for face detection in the browser

## Dataset

Source: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

- With mask: 3,725 images
- Without mask: 3,828 images
- Total: 7,553 images
- Training/validation split: 80% / 20%

Download and extract the images into:

```text
data/
├── with_mask/
└── without_mask/
```

Dataset images are excluded from Git.

## Model

- Input: 160 × 160 RGB images
- Pixel normalization included in the Keras model
- Frozen MobileNetV2 feature extractor
- Global average pooling, dropout and one sigmoid output
- Output probability represents “No Mask”

Reported best validation accuracy: **98.94%**.

This is validation accuracy on dataset images, not a measurement of
live webcam accuracy.

## Installation

From the repository root, using Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train the Model

```powershell
.\.venv\Scripts\python.exe src\train.py
```

Outputs include:

- `models/mask_detector.keras`
- `models/class_names.json`
- `results/accuracy.png`
- `results/loss.png`
- `results/history.json`
- `results/validation_metrics.json`

## OpenCV Webcam Demo

```powershell
.\.venv\Scripts\python.exe src\webcam.py
```

Click the webcam window before using these keys:

- **S**: Save a screenshot
- **Q**: Quit and release the camera

## Run the Website Locally

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 --directory web
```

Open http://localhost:8000 in Chrome or Edge.

The browser uses the converted TensorFlow.js classifier and BlazeFace.
Camera images are processed on your device.

## Training Plots

![Training and validation accuracy](results/accuracy.png)
![Training and validation loss](results/loss.png)

## Limitations

- Lighting, face angle and occlusion can affect predictions.
- Face detection may miss covered faces.
- OpenCV and the browser use different face detectors.
- Confidence is a model estimate and does not guarantee correctness.
- This project is an educational demonstration.
