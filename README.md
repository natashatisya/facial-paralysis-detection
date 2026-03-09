# Automated Facial Paralysis Detection System

A deep learning-based system that accurately identifies normal 
and paralyzed facial expressions, either in real-time via 
webcam or through uploaded images.

## About
This system was developed to assist in automated facial 
paralysis detection using deep learning. It focuses on 
optimizing the InceptionResNetV2 model to achieve precise 
detection of normal and paralyzed facial expressions.

## Objectives
- Detect and distinguish between normal and paralyzed facial expressions
- Fine-tune deep learning models for high accuracy
- Support both real-time webcam detection and image upload

## Model Used
- InceptionResNetV2 *(primary model)*
- DenseNet
- VGG19

## Download Pre-trained Models
Models are too large for GitHub. Download here:

> [Download All Models (Google Drive)](https://drive.google.com/drive/folders/1fiSqhKOw-MDmXY9ue5RQChcCaHEEZg44?usp=drive_link)

After downloading, place the model files inside the `Model/` folder:
```
Model/
├── inception_resnetv2_model.h5
├── densenet_model.h5
└── vgg19_model.h5
```

## How to Run

### 1. Clone the repository
```
git clone https://github.com/natashatisya/facial-paralysis-detection.git
cd facial-paralysis-detection
```

### 2. Install dependencies
```
pip install flask tensorflow keras opencv-python numpy flask-socketio
```

### 3. Download and place model files
Download from Google Drive link above and place in `Model/` folder.

### 4. Run the app
```
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```

## Features
- **Real-time detection** via webcam
- **Image upload** for facial paralysis prediction
- Displays result as **Normal Face** or **Stroke Face** with confidence score

## Built With
- Python
- Flask
- TensorFlow / Keras
- OpenCV
- InceptionResNetV2
