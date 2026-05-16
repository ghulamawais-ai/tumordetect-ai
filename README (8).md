# 🧠 TumorDetect AI

<div align="center">

![TumorDetect AI](https://img.shields.io/badge/TumorDetect-AI-blue?style=for-the-badge&logo=brain)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### 🚀 [**Live Demo — Try it Now**](https://huggingface.co/spaces/Awais138/tumordetect-ai)

An AI-powered Brain Tumor Detection System that analyzes MRI scans using Deep Learning, Explainable AI, and automated PDF report generation.

</div>

---

## 📌 Overview

**TumorDetect AI** is a web-based medical imaging application that allows users to upload brain MRI scans and receive:

- Automatic tumor detection and classification
- Visual tumor segmentation with bounding box
- Grad-CAM explainability heatmaps
- Detailed AI-generated PDF diagnostic report

> ⚠️ **Disclaimer:** This tool is for educational and research purposes only. It is NOT a substitute for professional medical diagnosis.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🔬 MRI Upload | Accepts PNG/JPG brain MRI images |
| 🎨 Preprocessing | RGB + HSV + LAB color fusion for enhanced image quality |
| 🧩 Segmentation | Dense Attention U-Net for precise tumor region detection |
| 🏷️ Classification | ResNet50 model classifying 4 tumor types |
| 🔥 Explainability | Grad-CAM heatmaps showing where AI focuses |
| 📄 PDF Report | Automated diagnostic report with patient details |
| ⚡ Real-time | Fast inference with GPU/CPU support |

---

## 🤖 AI Models

### 1. Classification Model — ResNet50
- **Architecture:** ResNet50 (fine-tuned)
- **Task:** 4-class brain tumor classification
- **Classes:**
  - ✅ No Tumor
  - 🔴 Glioma
  - 🟠 Meningioma
  - 🟡 Pituitary Tumor
- **Input:** 224×224 RGB MRI image
- **Output:** Tumor type + confidence score

### 2. Segmentation Model — Dense Attention U-Net
- **Architecture:** U-Net with Dense blocks + Attention gates
- **Task:** Pixel-wise tumor region segmentation
- **Input:** Brain MRI image
- **Output:** Binary tumor mask + bounding box + size measurements (cm)

### 3. Explainability — Grad-CAM
- **Method:** Gradient-weighted Class Activation Mapping
- **Purpose:** Visualizes which regions of the MRI influenced the AI decision
- **Output:** Color heatmap overlaid on original MRI

> 📦 Models are hosted on Hugging Face Spaces due to file size constraints.

---

## 🛠️ Tech Stack

**Backend**
- Python 3.10
- Flask (Web Framework)
- PyTorch (Deep Learning)
- OpenCV (Image Processing)
- ReportLab (PDF Generation)
- NumPy

**Frontend**
- HTML5, CSS3, JavaScript
- Responsive UI with animations

**Deployment**
- Hugging Face Spaces (Docker)

---

## 📁 Project Structure

```
tumordetect-ai/
│
├── app.py                  # Main Flask application
├── classification.py       # ResNet50 tumor classifier
├── segmentation.py         # Dense Attention U-Net segmentation
├── explainable_ai.py       # Grad-CAM implementation
├── preprocessing.py        # Image enhancement pipeline
├── xai_text.py             # AI-generated text explanations
├── report_generator.py     # PDF report generation
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
│
├── models/
│   └── weights/
│       ├── best_resnet50_brain_tumor.pth
│       └── best_dense_attention_unet.pth
│
├── templates/
│   └── index.html          # Frontend UI
│
└── static/                 # CSS, JS, Images
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/ghulamawais-ai/tumordetect-ai.git
cd tumordetect-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add model weights to models/weights/ folder

# 4. Run the app
python app.py
```

Then open `http://localhost:10000` in your browser.

---

## 📊 How It Works

```
MRI Upload
    ↓
Image Preprocessing (RGB + HSV + LAB fusion)
    ↓
MRI Validation (Is it a brain scan?)
    ↓
Tumor Segmentation (Dense Attention U-Net)
    ↓
Tumor Classification (ResNet50)
    ↓
Grad-CAM Explainability
    ↓
PDF Report Generation
    ↓
Results Displayed
```

---

## 📷 Screenshots

> Live demo available at: [https://huggingface.co/spaces/Awais138/tumordetect-ai](https://huggingface.co/spaces/Awais138/tumordetect-ai)

---

## 👨‍💻 Developer

**Ghulam Awais**
Department of Artificial Intelligence
The Islamia University of Bahawalpur — 2026

[![GitHub](https://img.shields.io/badge/GitHub-ghulamawais--ai-black?style=flat&logo=github)](https://github.com/ghulamawais-ai)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>⭐ If you found this useful, please star the repo!</strong>
</div>
