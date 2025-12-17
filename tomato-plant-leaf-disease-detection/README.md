# 🍅 Tomato Plant Leaf Disease Detection

A **research-oriented college project** developed as part of the *Research* subject, focused on detecting diseases in tomato plant leaves using **machine learning and deep learning techniques**. The system compares multiple models and provides predictions through a **Flask-based web application**.

---

## 📌 Project Overview

Tomato crops are highly susceptible to various leaf diseases that significantly affect yield and quality. This project aims to automatically identify tomato leaf diseases from images using computer vision and ML/DL models.

*The application allows users to upload a leaf image, select a trained model, and receive:*

* Predicted disease class
* Confidence score (%)
* Model used for prediction

---

## 🎯 Objectives

* Detect tomato leaf diseases using image-based classification
* Compare traditional ML and deep learning approaches
* Deploy trained models using a user-friendly web interface
* Serve as an academic **research and final-year–level project**

---

## 🧠 Models Implemented

| Model                   | Description                                              |
| ----------------------- | -------------------------------------------------------- |
| **CNN**                 | Custom Convolutional Neural Network trained from scratch |
| **ResNet50**            | Transfer learning using pretrained ResNet50              |
| **MobileNetV2 (V2Net)** | Lightweight transfer learning model                      |
| **KNN**                 | Classical ML model using extracted image features        |

---

## 🦠 Disease Classes

* Bacterial Spot
* Early Blight
* Late Blight
* Leaf Mold
* Septoria Leaf Spot
* Spider Mites
* Target Spot
* Yellow Leaf Curl Virus
* Mosaic Virus
* Healthy

---

## ⚙️ Workflow

1. **Data Preprocessing**

   * Image resizing & normalization
   * Data augmentation
   * Train / validation / test split

2. **Model Training**

   * CNN trained from scratch
   * Transfer learning with ResNet50 & MobileNetV2
   * Feature-based KNN classifier

3. **Evaluation**

   * Accuracy, loss curves
   * Classification report
   * Confusion matrix

4. **Deployment**

   * Flask-based web application
   * Model selection at runtime
   * Confidence score display

---

## 🌐 Web Application Features

* Image upload support
* Model selection (CNN / ResNet / MobileNet / KNN)
* Disease prediction with confidence
* Clean Bootstrap-based UI

---

## 🛠 Tech Stack

**Languages & Frameworks**

* Python
* Flask

**Deep Learning & ML**

* TensorFlow / Keras
* Scikit-learn

**Image Processing**

* OpenCV
* PIL

**Frontend**

* HTML
* Bootstrap 5

---

## 📊 Results

* **Custom CNN achieved the highest test accuracy** among all implemented models, demonstrating strong performance when trained specifically on the tomato leaf dataset.
* **MobileNetV2 (V2Net)** achieved high test accuracy (~87%), offering a good balance between performance and computational efficiency.
* **ResNet50** achieved lower test accuracy (~49%), indicating that deeper pretrained architectures may not always generalize well on limited agricultural datasets without extensive fine-tuning.
* **KNN** produced reasonable results as a classical machine learning baseline but was outperformed by deep learning models.

---

## 🎓 Academic Context

* **Project Type:** College Research Project
* **Subject:** Research
* **Domain:** Computer Vision, Machine Learning, Deep Learning
* **Use Case:** Smart Agriculture / Plant Disease Detection

---
