import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# ✅ MUST MATCH TRAINING FOLDER NAMES EXACTLY
# -------------------------------------------------
CLASS_NAMES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]

# -------------------------------------------------
# LOAD CLASSIFIER
# -------------------------------------------------
def load_classifier(weights_path, num_classes=4):
    """
    Loads ResNet50 classifier with correct head.
    STRICT weight loading to avoid silent bugs.
    """

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)  # 🔥 STRICT LOAD (NO silent failure)

    model.eval()
    model.to(DEVICE)

    print("✅ Classifier loaded with trained weights")
    return model

# -------------------------------------------------
# IMAGE CLASSIFICATION
# -------------------------------------------------
def classify_image(model, bgr_img, input_size=224):
    """
    Returns:
    - tumor_type (string)
    - confidence (float)
    - full probability vector
    """

    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    x = transform(rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

# -------------------------------------------------
# MRI VALIDATION (BASIC, SAFE)
# -------------------------------------------------
def is_brain_mri(bgr_img):
    """
    Simple MRI check:
    MRI → low saturation, grayscale dominant
    """

    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    sat_mean = hsv[..., 1].mean()
    sat_std = hsv[..., 1].std()

    return sat_mean < 30 and sat_std < 20
