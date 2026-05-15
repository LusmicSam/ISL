# Indian Sign Language (ISL) Detection 🤟

This project aims to detect and recognize Indian Sign Language (ISL) gestures in real-time or from images using the Mediapipe library and a custom-trained Keras model.

![All gestures covered by project](images/allGestures.png)

## 🌟 Features
- **Real-time Detection:** Detects ISL gestures using the webcam.
- **Image Upload:** Upload an image to identify the hand signs.
- **Streamlit Web App:** Easily deployable via Streamlit Community Cloud with a clean and interactive interface.
- **High Accuracy:** Utilizes Mediapipe for robust hand landmark detection and a Neural Network (Keras) for gesture classification.

## 🚀 Live Demo
*(Insert your Streamlit app link here when deployed)*

## 🛠️ Requirements & Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/LusmicSam/ISL.git
   cd ISL
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### 1. Web App (Streamlit)
To run the Streamlit web interface locally:
```bash
streamlit run app.py
```
This will open a local web server (usually at `http://localhost:8501`) where you can:
- **Take a picture** with your camera to detect signs.
- **Upload an image** to process existing photos.

### 2. Local Real-Time Script
To run the standard OpenCV webcam script (opens a separate window):
```bash
python isl_detection.py
```
*Press `ESC` to exit the webcam window.*

## 🧠 How it works

1. **Hand Tracking:** The program uses the Mediapipe library to detect 21 3D landmarks on the hand and fingers.
2. **Preprocessing:** Coordinates are normalized and made relative to the wrist to ensure distance invariance.
3. **Classification:** These processed landmarks are fed into a Keras model (`model.h5`) trained on an [ISL Kaggle dataset](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl).
4. **Prediction:** The model outputs the predicted gesture (A-Z, 1-9), which is then drawn onto the image or live feed.

![Process image](images/process.png)

## 📁 File Structure

- `app.py`: Streamlit web application.
- `isl_detection.py`: Python script for real-time OpenCV webcam detection.
- `dataset_keypoint_generation.py`: Script to convert the ISL dataset images into 42 (x,y) landmark coordinates.
- `keypoint.csv`: Extracted 42 landmarks for all images in the dataset.
- `ISL_classifier.ipynb`: Jupyter notebook used to train the Keras classification model.
- `model.h5`: The trained Keras gesture classifier model.
- `requirements.txt`: Python dependencies required to run the project.

## 📸 Examples
![example image 1](images/example1.png)
![example image 2](images/example2.png)

## 🔮 Future Improvements
- Expand the dataset to include more examples and variations.
- Support dynamic gestures (words/phrases instead of just static letters).
- Text-to-speech functionality to narrate the detected signs.

---
**Author:** Shivam
