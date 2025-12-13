# 🩺 Skin Cancer Detection using Deep Learning

A computer vision project that detects skin cancer from dermoscopic images using deep learning. The system leverages CNNs and transfer learning, with Grad-CAM visualizations for explainability and a TFLite model for lightweight deployment.

---

## 📂 Dataset Used

Melanoma Cancer Dataset:
https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset
   
---

## 🎯 Key Features

- Skin lesion classification using deep learning
- Custom CNN and MobileNetV2 (transfer learning)
- Grad-CAM heatmaps for model interpretability
- Confidence score (%) for predictions
- TensorFlow Lite (TFLite) model for optimized inference

---

## 🧠 Models Used

- Custom CNN
- MobileNetV2 (Transfer Learning)
- TensorFlow Lite (TFLite) for optimized inference

---

## 📊 Explainability with Grad-CAM

**Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize the regions of the image that influence the model’s decision, making the predictions more interpretable and trustworthy.
Grad-CAM is also used with MobileNetV2 and not the CNN.**
---

## ▶️ Pre-trained Models

1. skin_cancer_cnn.h5
2. skin_cancer_mobilenetv2.h5
3. skin_cancer_model.tflite

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---
