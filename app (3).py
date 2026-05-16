from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os, cv2, time, traceback
import torch
import numpy as np

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

from preprocessing import adaptive_channel_selection
from classification import load_classifier, classify_image, is_brain_mri
from explainable_ai import gradcam_for_resnet, overlay_cam, analyze_cam
from segmentation import (
    load_segmentation_model,
    segment_tumor,
    overlay_mask,
    compute_tumor_stats,
    draw_bounding_box
)
from xai_text import generate_dynamic_explanation
from report_generator import generate_pdf


# -------------------------------------------------
# Flask Setup
# -------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Max upload size (10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


# -------------------------------------------------
# Load Models
# -------------------------------------------------
print("🔄 Loading AI models...")

try:
    CLS_MODEL = load_classifier("models/weights/best_resnet50_brain_tumor.pth")
    print("✅ Classification model loaded")
except Exception as e:
    CLS_MODEL = None
    print("❌ Classifier failed:", e)

try:
    SEG_MODEL = load_segmentation_model("models/weights/best_dense_attention_unet.pth")
    print("✅ Segmentation model loaded")
except Exception as e:
    SEG_MODEL = None
    print("❌ Segmentation failed:", e)

print("🚀 TumorDetect AI initialized")


# -------------------------------------------------
# Risk Level Helper
# -------------------------------------------------
def compute_risk(tumor_type, confidence):

    if tumor_type.lower() in ["no tumor", "no_tumor"]:
        return "No Risk"

    if confidence >= 0.85:
        return "High Risk"
    elif confidence >= 0.65:
        return "Medium Risk"
    else:
        return "Low Risk"


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/reports/<path:filename>")
def download_report(filename):
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)


# -------------------------------------------------
# Upload + AI Processing
# -------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload_file():

    if CLS_MODEL is None:
        return jsonify({"error": "Classifier unavailable"}), 503

    try:

        # ---------- Patient Info ----------
        patient_id = request.form.get("patient_id", "Unknown")
        patient_name = request.form.get("patient_name", "Unknown")
        age = request.form.get("age", "N/A")
        gender = request.form.get("gender", "N/A")
        mri_date = request.form.get("mri_date", "N/A")

        # ---------- File ----------
        file = request.files.get("file")

        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            return jsonify({"error": "Only PNG/JPG images allowed"}), 400

        # ---------- Save Image ----------
        ts = int(time.time())
        safe_name = file.filename.replace(" ", "_")

        base_name = f"{ts}_{safe_name}"
        orig_path = os.path.join(UPLOAD_FOLDER, base_name)

        file.save(orig_path)

        image = cv2.imread(orig_path)

        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        # ---------- Preprocessing ----------
        enhanced = adaptive_channel_selection(image)

        enhanced_name = f"enhanced_{base_name}"
        enhanced_path = os.path.join(UPLOAD_FOLDER, enhanced_name)

        cv2.imwrite(enhanced_path, enhanced)

        # ---------- MRI Validation ----------
        if not is_brain_mri(image):

            return jsonify({
                "tumor_type": "Not an MRI Scan",
                "confidence": "N/A",
                "risk_level": "N/A",
                "original_image": url_for("uploaded_file", filename=base_name),
                "enhanced_image": url_for("uploaded_file", filename=enhanced_name),
                "segmentation_image": "",
                "gradcam_image": "",
                "report_pdf": ""
            })

        # -------------------------------------------------
        # Segmentation
        # -------------------------------------------------
        tumor_stats = {}
        seg_overlay_path = ""

        if SEG_MODEL is not None:

            mask = segment_tumor(SEG_MODEL, image)

            if mask is not None and np.sum(mask) > 0:

                tumor_stats = compute_tumor_stats(mask)

                seg_overlay = overlay_mask(image, mask)
                seg_overlay = draw_bounding_box(seg_overlay, tumor_stats)

                seg_name = f"seg_{base_name}"
                seg_overlay_path = os.path.join(UPLOAD_FOLDER, seg_name)

                cv2.imwrite(seg_overlay_path, seg_overlay)

        # -------------------------------------------------
        # Classification
        # -------------------------------------------------
        tumor_type, confidence, _ = classify_image(CLS_MODEL, image)

        confidence_percent = f"{confidence*100:.2f}%"
        risk_level = compute_risk(tumor_type, confidence)

        # -------------------------------------------------
        # Grad-CAM Explainability
        # -------------------------------------------------
        gradcam_path = ""
        cam_info = {}
        details = {}

        if tumor_type.lower() not in ["no tumor", "no_tumor"]:

            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate GradCAM
            cam = gradcam_for_resnet(CLS_MODEL, rgb_img)

            # Normalize CAM
            cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            # If segmentation exists, combine with mask
            if mask is not None and np.sum(mask) > 0:
    
               mask_norm = mask / 255.0
               # Blend GradCAM with tumor mask
               cam_refined = cam_norm + 0.5 * mask_norm
               cam_refined = np.clip(cam_refined, 0, 1)

               cam_overlay = overlay_cam(image, cam_refined)

            else:
               cam_overlay = overlay_cam(image, cam_norm)


            gradcam_name = f"gradcam_{base_name}"
            gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_name)

            cv2.imwrite(gradcam_path, cam_overlay)

            cam_info = analyze_cam(cam)

            details = generate_dynamic_explanation(tumor_type, cam_info)

        else:
            cam_info = {"mean_activation": 0.0, "max_activation": 0.0, "activated_area": 0.0}
            details = generate_dynamic_explanation(tumor_type, cam_info)

        # -------------------------------------------------
        # Generate PDF Report
        # -------------------------------------------------
        report_filename = f"report_{patient_id}_{ts}.pdf"
        report_path = os.path.join(REPORT_FOLDER, report_filename)

        patient_data = {

            "Patient ID": patient_id,
            "Patient Name": patient_name,
            "Age": age,
            "Gender": gender,
            "MRI Date": mri_date,

            "Tumor Type": tumor_type,
            "Risk Level": risk_level,
            "Model Confidence": confidence_percent,

            "Tumor Location": str(tumor_stats.get("location","N/A")),
            "Tumor Width": f"{tumor_stats.get('width_cm',0)} cm",
            "Tumor Height": f"{tumor_stats.get('height_cm',0)} cm",
            "Tumor Area": f"{tumor_stats.get('area_cm2',0)} cm²",

            "Description": details.get("desc","N/A"),
            "Possible Cause": details.get("cause","N/A"),
            "Treatment": details.get("treat","N/A")
        }

        image_paths = {
            "Original MRI": orig_path,
            "Enhanced MRI": enhanced_path
        }

        if seg_overlay_path and os.path.exists(seg_overlay_path):
            image_paths["Tumor Segmentation"] = seg_overlay_path

        if gradcam_path and os.path.exists(gradcam_path):
            image_paths["GradCAM"] = gradcam_path

        generate_pdf(
            report_path, patient_data, image_paths,
            cam_mean=cam_info.get("mean_activation"),
            cam_peak=cam_info.get("max_activation"),
            cam_area=cam_info.get("activated_area", 0) * 100 if cam_info else None
        )

        # -------------------------------------------------
        # API Response
        # -------------------------------------------------
        return jsonify({

            "tumor_type": tumor_type,
            "confidence": confidence_percent,
            "risk_level": risk_level,

            "original_image": url_for("uploaded_file", filename=base_name),
            "enhanced_image": url_for("uploaded_file", filename=enhanced_name),

            "segmentation_image": (
                url_for("uploaded_file", filename=f"seg_{base_name}")
                if seg_overlay_path else ""
            ),

            "gradcam_image": (
                url_for("uploaded_file", filename=f"gradcam_{base_name}")
                if gradcam_path else ""
            ),

            "tumor_location": tumor_stats.get("location"),
            "tumor_width_cm": tumor_stats.get("width_cm"),
            "tumor_height_cm": tumor_stats.get("height_cm"),
            "tumor_area_cm2": tumor_stats.get("area_cm2"),

            "tumor_description": details.get("desc"),
            "tumor_cause": details.get("cause"),
            "tumor_treatment": details.get("treat"),

            "tumor_description": details.get("desc"),
            "tumor_cause": details.get("cause"),
            "tumor_treatment": details.get("treat"),

            "cam_mean_activation": round(cam_info.get("mean_activation", 0), 3) if cam_info else None,
            "cam_max_activation":  round(cam_info.get("max_activation",  0), 3) if cam_info else None,
            "cam_activated_area":  round(cam_info.get("activated_area",  0) * 100, 1) if cam_info else None,

            "report_pdf": url_for("download_report", filename=report_filename)

        })

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


# -------------------------------------------------
# Run Server
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
