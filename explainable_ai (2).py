import torch
import numpy as np
import cv2

# -------------------------------------------------
# Grad-CAM for ResNet (Clean & Correct)
# -------------------------------------------------
def gradcam_for_resnet(model, bgr_img):
    """
    Generates Grad-CAM heatmap highlighting
    regions influencing the AI prediction.

    NOTE:
    - This is NOT tumor segmentation
    - This is an explainability map
    """

    model.eval()
    device = next(model.parameters()).device

    # Target layer for ResNet
    target_layer = model.layer4

    activations = []
    gradients = []

    # ---------- Hooks ----------
    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # ---------- Preprocess (match training) ----------
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0)
    img = img.to(device)

    # ---------- Forward ----------
    output = model(img)
    class_idx = output.argmax(dim=1).item()

    # ---------- Backward ----------
    model.zero_grad()
    output[0, class_idx].backward()

    # ---------- Grad-CAM ----------
    act = activations[0]      # (1, C, H, W)
    grad = gradients[0]       # (1, C, H, W)

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()
    cam = torch.relu(cam)

    cam = cam.detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # Resize to original image size
    cam = cv2.resize(cam, (bgr_img.shape[1], bgr_img.shape[0]))

    # Remove hooks
    h1.remove()
    h2.remove()

    return cam


# -------------------------------------------------
# Overlay Grad-CAM (Clean Visualization)
# -------------------------------------------------
def overlay_cam(bgr_img, cam, alpha=0.35):
    """
    Overlays Grad-CAM heatmap on image.

    Color meaning:
    - Blue  : low influence
    - Red   : high influence
    """

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        bgr_img, 1 - alpha,
        heatmap, alpha,
        0
    )

    return overlay


# -------------------------------------------------
# CAM Analysis (for text explanation)
# -------------------------------------------------
def analyze_cam(cam):
    """
    Extracts simple statistics for dynamic text explanation
    """

    return {
        "mean_activation": float(np.mean(cam)),
        "max_activation": float(np.max(cam)),
        "activated_area": float(np.sum(cam > 0.6) / cam.size)
    }
