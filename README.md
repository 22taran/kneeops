# KneeOps: MRI Knee Injury Classification Web Application

## Overview
KneeOps is a web-based application for classifying knee MRI images to detect injuries using deep learning. It features a React frontend for user interaction and a Python backend (Flask) for model inference. The backend uses a ResNet18 convolutional neural network, trained on labeled MRI data, to provide accurate predictions.

---

## Table of Contents
- [Model Choice and Rationale](#model-choice-and-rationale)
- [Data Preprocessing](#data-preprocessing)
- [Training Pipeline](#training-pipeline)
- [Backend (Flask API)](#backend-flask-api)
- [Frontend (React)](#frontend-react)
- [End-to-End Flow](#end-to-end-flow)
- [Why This Approach is Better](#why-this-approach-is-better)
- [Deployment & Maintenance](#deployment--maintenance)
- [References](#references)
- [Future Improvements](#future-improvements)

---

## Model Choice and Rationale

### Model Used
- **ResNet18** (from the `timm` library)
  - **Input Channels:** 1 (grayscale MRI images)
  - **Number of Classes:** Matches the number of unique injury types in your dataset (as determined by the label encoder during training)

### Why ResNet18?
- **Proven Performance:** Widely used in medical imaging for strong generalization.
- **Residual Connections:** Prevent vanishing gradients, enabling deeper networks and better feature extraction.
- **Lightweight:** Efficient for real-time inference in web apps.
- **Transfer Learning:** Pretrained weights can be leveraged, but here the model is trained from scratch for single-channel input.

### Why Not a Custom CNN?
- **Generalization:** Custom CNNs may underperform on complex medical images.
- **Community Support:** ResNet is well-tested and widely supported.

---

## Data Preprocessing
- **Resize:** All images resized to (224, 224) pixels.
- **Grayscale Conversion:** MRI images are single-channel.
- **Normalization:** Mean = `[0.485]`, Std = `[0.229]` (chosen for MRI intensity distribution).

---

## Training Pipeline
- **Label Encoding:** Class names encoded using a `LabelEncoder` and saved to `classes.json` or `classes.npy` for consistent mapping.
- **Model Saving:** Trained model’s state dict saved as `best_model.pt` for deployment.

---

## Backend (Flask API)
- **Model Loading:**
  - Loads ResNet18 with correct architecture (`in_chans=1`, `num_classes=...`).
  - Loads trained weights from `best_model.pt`.
  - Loads class names from `classes.json` for correct label mapping.
- **Image Preprocessing:**
  - Uploaded images are resized, converted to grayscale, and normalized as per training.
- **Prediction Endpoint:**
  - Accepts image uploads and returns predicted class and confidence for each image.
- **Error Handling:**
  - Handles model loading errors, file format issues, and provides informative error messages.

---

## Frontend (React)
- **File Upload:**
  - Users can upload one or more MRI images.
  - Only the first image’s analysis is shown in the chat interface.
- **Analysis Display:**
  - Left sidebar shows overall summary: total images, average confidence, healthy/injured status.
  - Chat interface displays analysis for the first image only.
- **User Experience:**
  - Simple, intuitive interface for clinicians or users.
  - Immediate feedback after upload.

---

## End-to-End Flow
1. User uploads MRI images via the web interface.
2. Frontend sends images to the backend API.
3. Backend preprocesses images and runs inference using ResNet18.
4. Predictions (class and confidence) are returned to the frontend.
5. Frontend displays the results:
   - Chat shows the first image’s analysis.
   - Sidebar shows overall summary.

---

## Why This Approach is Better
- **Accuracy:** ResNet18 is well-suited for medical images, outperforming simpler CNNs.
- **Consistency:** Preprocessing and label mapping are matched between training and inference.
- **Scalability:** Modular design allows for easy upgrades (e.g., deeper ResNets).
- **User-Centric:** Frontend is designed for clarity and actionable insights.

---

## Deployment & Maintenance
- **Model and class names must be kept in sync between training and backend deployment.**
- **Backend should be restarted after updating the model or class mapping files.**
- **Frontend can be extended for more features (batch analysis, report downloads, etc.).**

---

## References
- [He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.](https://arxiv.org/abs/1512.03385)
- [timm: PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [ResNet in Medical Imaging: A Review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7282275/)

---

## Future Improvements
- **Model Upgrades:** Experiment with deeper ResNets or domain-specific architectures.
- **Explainability:** Add Grad-CAM or saliency maps for interpretability.
- **Multi-class/Multi-label Support:** Detect multiple injuries per image.
- **Integration:** Connect with hospital PACS systems for seamless workflow. 