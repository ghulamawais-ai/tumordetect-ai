---
title: TumorDetect AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 10000
pinned: true
license: mit
---

# TumorDetect AI

**Developed by:** Ghulam Awais **Supervised by:** Sir Mohsin Abbas

## Live Demo

https://huggingface.co/spaces/Awais138/tumordetect-ai

## Overview

An AI-powered Brain Tumor Detection System that analyzes brain MRI scans using Deep Learning, Explainable AI (Grad-CAM), and automated PDF report generation.

## Features

- 🔬 Brain MRI upload and analysis
- 🎨 RGB + HSV + LAB color fusion preprocessing
- 🧩 Dense Attention U-Net tumor segmentation
- 🏷️ ResNet50 4-class tumor classification
- 🔥 Grad-CAM explainable AI heatmaps
- 📄 Automated PDF diagnostic report generation

## AI Models

- **Classification:** ResNet50 — detects Glioma, Meningioma, Pituitary Tumor, No Tumor
- **Segmentation:** Dense Attention U-Net — pixel-wise tumor region detection
- **Explainability:** Grad-CAM — heatmap visualization of AI decision regions

> 📦 Models are hosted on Hugging Face Spaces due to size constraints.

## Tech Stack

- Python, Flask, PyTorch, OpenCV, ReportLab

## License

MIT
