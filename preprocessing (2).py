import cv2
import numpy as np

# -----------------------------------
# CLAHE Enhancement
# -----------------------------------
def apply_clahe(channel):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(channel)


# -----------------------------------
# DEPTH MAP (SIS-style)
# -----------------------------------
def generate_depth_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gradient (edges)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)

    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)

    # Laplacian (fine details)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)

    # Combine both → depth-like map
    depth = cv2.addWeighted(
        gradient.astype(np.uint8), 0.6,
        laplacian.astype(np.uint8), 0.4,
        0
    )

    return depth


# -----------------------------------
# MAIN FUNCTION (IMPORTANT)
# -----------------------------------
def adaptive_channel_selection(img):
    """
    This function will be used in your app.py
    DO NOT change function name
    """

    img = cv2.resize(img, (256, 256))

    # ---------------- RGB ----------------
    r, g, b = cv2.split(img)

    r = apply_clahe(r)
    g = apply_clahe(g)
    b = apply_clahe(b)

    rgb = cv2.merge([r, g, b])

    # ---------------- HSV ----------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = apply_clahe(v)

    hsv = cv2.merge([h, s, v])
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ---------------- LAB ----------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b2 = cv2.split(lab)

    l = apply_clahe(l)

    lab = cv2.merge([l, a, b2])
    lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ---------------- DEPTH MAP ----------------
    depth = generate_depth_map(img)
    depth_3ch = cv2.merge([depth, depth, depth])

    # ---------------- FINAL FUSION ----------------
    fused = cv2.addWeighted(rgb, 0.3, hsv, 0.3, 0)
    fused = cv2.addWeighted(fused, 1.0, lab, 0.2, 0)
    fused = cv2.addWeighted(fused, 1.0, depth_3ch, 0.2, 0)

    return fused


# -----------------------------------
# OPTIONAL (if your old code uses this)
# -----------------------------------
def preprocess_image(img):
    """
    Some files like classification.py may call preprocess_image()
    So we map it to adaptive_channel_selection
    """
    return adaptive_channel_selection(img)