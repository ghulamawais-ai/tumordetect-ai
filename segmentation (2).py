import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# -------------------------------------------------
# Convolution Block
# -------------------------------------------------
class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.conv(x)


# -------------------------------------------------
# Dense Block
# -------------------------------------------------
class DenseBlock(nn.Module):

    def __init__(self,in_c,growth):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c,growth,3,padding=1)
        self.conv2 = nn.Conv2d(in_c+growth,growth,3,padding=1)

    def forward(self,x):

        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(torch.cat([x,x1],dim=1)))

        return torch.cat([x,x1,x2],dim=1)
# -------------------------------------------------
# Attention Gate
# -------------------------------------------------
class AttentionBlock(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# -------------------------------------------------
# Dense Attention U-Net
# -------------------------------------------------
class DenseAttentionUNet(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):

        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv1 = ConvBlock(in_ch, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)

        self.dense = DenseBlock(512, 128)  # 896 channels

        self.up4 = nn.ConvTranspose2d(768,256,2,stride=2)
        self.att4 = AttentionBlock(256, 512, 128)
        self.conv_up4 = ConvBlock(768, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionBlock(128, 256, 64)
        self.conv_up3 = ConvBlock(384, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att2 = AttentionBlock(64, 128, 32)
        self.conv_up2 = ConvBlock(192, 64)

        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):

        d1 = self.conv1(x)
        d2 = self.conv2(self.pool(d1))
        d3 = self.conv3(self.pool(d2))
        d4 = self.conv4(self.pool(d3))

        bridge = self.dense(self.pool(d4))

        u1 = self.up4(bridge)
        d4 = self.att4(u1, d4)
        c1 = self.conv_up4(torch.cat([u1, d4], dim=1))

        u2 = self.up3(c1)
        d3 = self.att3(u2, d3)
        c2 = self.conv_up3(torch.cat([u2, d3], dim=1))

        u3 = self.up2(c2)
        d2 = self.att2(u3, d2)
        c3 = self.conv_up2(torch.cat([u3, d2], dim=1))

        return self.out(c3)


# -------------------------------------------------
# Load Model
# -------------------------------------------------
def load_segmentation_model(weights_path):

    model = DenseAttentionUNet(in_ch=3, out_ch=1)

    state_dict = torch.load(weights_path, map_location=DEVICE)

    model.load_state_dict(state_dict)

    model.eval().to(DEVICE)

    print("✅ Dense Attention U-Net loaded successfully")

    return model


# -------------------------------------------------
# Tumor Segmentation
# -------------------------------------------------
def segment_tumor(model, bgr_img, input_size=256, thresh=0.03):

    H, W = bgr_img.shape[:2]

    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(rgb, (input_size, input_size))
    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2,0,1))

    x = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        logits = model(x)

        prob = torch.sigmoid(logits)

        prob = F.interpolate(
            prob,
            size=(H,W),
            mode="bilinear",
            align_corners=False
        )

        prob = prob.squeeze().cpu().numpy()

    mask = (prob >= thresh).astype(np.uint8) * 255
    print("DEBUG: mask pixel sum =", np.sum(mask))

    # Clean noise but DO NOT remove tumor
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# -------------------------------------------------
# Overlay Mask
# -------------------------------------------------
def overlay_mask(bgr_img, mask):

    output = bgr_img.copy()

    output[mask > 0] = (0,0,255)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(output, contours, -1, (0,0,200), 2)

    return output


# -------------------------------------------------
# Tumor Measurements
# -------------------------------------------------
def compute_tumor_stats(mask, pixel_spacing=0.05):

    coords = np.column_stack(np.where(mask > 0))

    if len(coords) == 0:
        return {
            "location": None,
            "width_px": 0,
            "height_px": 0,
            "area_px": 0,
            "width_cm": 0,
            "height_cm": 0,
            "area_cm2": 0
        }

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    width_px = x_max - x_min
    height_px = y_max - y_min
    area_px = int(np.sum(mask > 0))

    width_cm = width_px * pixel_spacing
    height_cm = height_px * pixel_spacing
    area_cm2 = area_px * (pixel_spacing ** 2)

    return {
        "location": (int(x_min), int(y_min), int(x_max), int(y_max)),
        "width_px": int(width_px),
        "height_px": int(height_px),
        "area_px": area_px,
        "width_cm": round(width_cm,2),
        "height_cm": round(height_cm,2),
        "area_cm2": round(area_cm2,2)
    }


# -------------------------------------------------
# Draw Bounding Box
# -------------------------------------------------
def draw_bounding_box(image, stats, color=(0,0,255), thickness=2):

    if not stats or not stats.get("location"):
        return image

    x1,y1,x2,y2 = stats["location"]

    boxed = image.copy()

    cv2.rectangle(boxed,(x1,y1),(x2,y2),color,thickness)

    return boxed