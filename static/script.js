// TumorDetect AI - Final Stable Script (WITH SEGMENTATION + MEASUREMENTS)

// -------------------------------------------------
// ELEMENT REFERENCES
// -------------------------------------------------
const uploadBox = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const errorText = document.getElementById("errorText");

const loadingSection = document.getElementById("loadingSection");
const resultsSection = document.getElementById("resultsSection");
const processingText = document.getElementById("processingText");

const origImg = document.getElementById("origImg");
const enhancedImg = document.getElementById("enhancedImg");
const gradcamImg = document.getElementById("gradcamImg");
const segImg = document.getElementById("segImg");

const tumorType = document.getElementById("tumorType");
const confidence = document.getElementById("confidence");
const tumorDesc = document.getElementById("tumorDesc");
const tumorCause = document.getElementById("tumorCause");
const tumorTreat = document.getElementById("tumorTreat");
const confidenceBarInner = document.getElementById("confidenceBarInner");

const tumorLocation = document.getElementById("tumorLocation");
const tumorWidth = document.getElementById("tumorWidth");
const tumorHeight = document.getElementById("tumorHeight");
const tumorArea = document.getElementById("tumorArea");

const pdfBtn = document.getElementById("downloadPdfBtn");

const camMean = document.getElementById("camMean");
const camPeak = document.getElementById("camPeak");
const camArea = document.getElementById("camArea");

// Hide PDF button initially
if (pdfBtn) pdfBtn.style.display = "none";

// -------------------------------------------------
// FILE INPUT
// -------------------------------------------------
if (fileInput) {
  fileInput.addEventListener("change", e => {
    if (e.target.files && e.target.files[0]) {
      handleUpload(e.target.files[0]);
    }
  });
}

// -------------------------------------------------
// DRAG & DROP
// -------------------------------------------------
if (uploadBox) {
  ["dragover", "dragenter"].forEach(evt => {
    uploadBox.addEventListener(evt, e => {
      e.preventDefault();
      uploadBox.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(evt => {
    uploadBox.addEventListener(evt, e => {
      e.preventDefault();
      uploadBox.classList.remove("dragover");
    });
  });

  uploadBox.addEventListener("drop", e => {
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  });
}

// -------------------------------------------------
// UPLOAD LOGIC
// -------------------------------------------------
async function handleUpload(file) {
  resetDisplay();
  showLoading(true);

  try {
    const steps = [
      "Uploading MRI scan...",
      "Enhancing contrast...",
      "Segmenting tumor region...",
      "Classifying tumor...",
      "Generating explainable AI...",
      "Generating medical report..."
    ];

    for (const step of steps) {
      processingText.textContent = step;
      await new Promise(r => setTimeout(r, 350));
    }

    const formData = new FormData();
    formData.append("file", file);

    // Patient details
    formData.append("patient_id", document.getElementById("patientId").value);
    formData.append("patient_name", document.getElementById("patientName").value);
    formData.append("age", document.getElementById("age").value);
    formData.append("gender", document.getElementById("gender").value);
    formData.append("mri_date", document.getElementById("mriDate").value);

    const res = await fetch("/upload", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Server error");

    displayResults(data, file);

  } catch (err) {
    showError(err.message);
  } finally {
    loadingSection.classList.add("hidden");
  }
}

// -------------------------------------------------
// DISPLAY RESULTS
// -------------------------------------------------
function displayResults(data, file) {
  uploadBox.classList.add("hidden");
  resultsSection.classList.remove("hidden");
  setTimeout(() => resultsSection.classList.add("visible"), 50);

  // Images
  if (origImg)
    origImg.src = data.original_image || URL.createObjectURL(file);

  if (enhancedImg)
    enhancedImg.src = data.enhanced_image
      ? data.enhanced_image + "?t=" + Date.now()
      : "";

  // Segmentation image
  if (segImg) {
    if (data.segmentation_image) {
      segImg.style.display = "block";
      segImg.src = data.segmentation_image + "?t=" + Date.now();
    } else {
      segImg.style.display = "none";
    }
  }

  // Grad-CAM
  if (
    gradcamImg &&
    data.gradcam_image &&
    data.tumor_type &&
    !data.tumor_type.toLowerCase().includes("no") &&
    !data.tumor_type.toLowerCase().includes("not")
  ) {
    gradcamImg.style.display = "block";
    gradcamImg.src = data.gradcam_image + "?t=" + Date.now();
  } else if (gradcamImg) {
    gradcamImg.style.display = "none";
  }

  // Text results
tumorType.textContent = data.tumor_type || "-";
confidence.textContent = data.confidence || "-";
tumorDesc.textContent = data.tumor_description || "N/A";
tumorCause.textContent = data.tumor_cause || "N/A";
tumorTreat.textContent = data.tumor_treatment || "N/A";

// CAM stats
if (camMean) camMean.textContent = data.cam_mean_activation != null ? data.cam_mean_activation.toFixed(3) : "N/A";
if (camPeak) camPeak.textContent = data.cam_max_activation  != null ? data.cam_max_activation.toFixed(3)  : "N/A";
if (camArea) camArea.textContent = data.cam_activated_area  != null ? data.cam_activated_area.toFixed(1) + "%" : "N/A";

// -------------------------------------------------
// Tumor measurements (NEW)
// -------------------------------------------------
// Tumor measurements
const loc = data.tumor_location;
document.getElementById("tumorLocation").textContent =
  loc ? `(${loc[0]}, ${loc[1]}) to (${loc[2]}, ${loc[3]})` : "-";

document.getElementById("tumorWidth").textContent =
  data.tumor_width_cm ? data.tumor_width_cm.toFixed(2) : "-";

document.getElementById("tumorHeight").textContent =
  data.tumor_height_cm ? data.tumor_height_cm.toFixed(2) : "-";

document.getElementById("tumorArea").textContent =
  data.tumor_area_cm2 ? data.tumor_area_cm2.toFixed(2) : "-";

  // Confidence bar
  const pct = data.confidence
    ? parseFloat(data.confidence.replace("%", ""))
    : 0;

  if (confidenceBarInner) {
    confidenceBarInner.style.width = "0%";
    confidenceBarInner.textContent = "";

    setTimeout(() => {
      confidenceBarInner.style.width = pct + "%";
      confidenceBarInner.textContent = Math.round(pct) + "%";
    }, 200);
  }

  // PDF button
  if (pdfBtn && data.report_pdf) {
    pdfBtn.href = data.report_pdf + "?t=" + Date.now();
    pdfBtn.style.display = "inline-block";
  } else if (pdfBtn) {
    pdfBtn.style.display = "none";
  }

  resultsSection.scrollIntoView({ behavior: "smooth" });
}

// -------------------------------------------------
// UTILITIES
// -------------------------------------------------
function showLoading(show) {
  loadingSection.classList.toggle("hidden", !show);
  uploadBox.classList.toggle("hidden", show);
}

function showError(msg) {
  errorText.textContent = msg;
  uploadBox.classList.remove("hidden");
  loadingSection.classList.add("hidden");
  resultsSection.classList.add("hidden");
}

function resetDisplay() {
  errorText.textContent = "";

  if (origImg) origImg.src = "";
  if (enhancedImg) enhancedImg.src = "";
  if (gradcamImg) gradcamImg.src = "";
  if (segImg) segImg.src = "";

  if (tumorLocation) tumorLocation.textContent = "-";
  if (tumorWidth) tumorWidth.textContent = "-";
  if (tumorHeight) tumorHeight.textContent = "-";
  if (tumorArea) tumorArea.textContent = "-";

  if (pdfBtn) pdfBtn.style.display = "none";

  if (camMean) camMean.textContent = "—";
  if (camPeak) camPeak.textContent = "—";
  if (camArea) camArea.textContent = "—";

  resultsSection.classList.add("hidden");
  resultsSection.classList.remove("visible");

  if (confidenceBarInner) {
    confidenceBarInner.style.width = "0";
    confidenceBarInner.textContent = "";
  }
}

function resetUpload() {
  if (fileInput) fileInput.value = "";
  resetDisplay();
  uploadBox.classList.remove("hidden");
}

// -------------------------------------------------
// SCROLL REVEAL
// -------------------------------------------------
const revealElements = document.querySelectorAll(".reveal");

function revealOnScroll() {
  const windowHeight = window.innerHeight;
  revealElements.forEach(el => {
    if (el.getBoundingClientRect().top < windowHeight - 100) {
      el.classList.add("active");
    }
  });
}

window.addEventListener("scroll", revealOnScroll);
revealOnScroll();