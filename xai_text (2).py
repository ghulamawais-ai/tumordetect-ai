def generate_dynamic_explanation(tumor_type, cam_info):
    """
    Generates detailed, image-specific clinical explanation
    using Grad-CAM activation statistics from ResNet50.

    cam_info keys:
        mean_activation  — average activation strength across the map
        max_activation   — peak activation value
        activated_area   — fraction of image with activation > 0.6
    """

    area   = cam_info.get("activated_area",  0)
    strength = cam_info.get("mean_activation", 0)
    peak   = cam_info.get("max_activation",  0)

    # ── Risk qualifier helper ──────────────────────────────────────
    def risk_qualifier(strength, area):
        if strength > 0.45 and area > 0.25:
            return "high-intensity, widespread"
        elif strength > 0.35 or area > 0.20:
            return "moderate-to-high intensity, diffuse"
        else:
            return "low-to-moderate intensity, localized"

    # ──────────────────────────────────────────────────────────────
    # GLIOMA
    # ──────────────────────────────────────────────────────────────
    if tumor_type in ("Glioma", "glioma_tumor"):

        if strength > 0.4:
            return {
                "desc": (
                    f"Grad-CAM reveals {risk_qualifier(strength, area)} activation "
                    f"(mean={strength:.2f}, peak={peak:.2f}) spread across a "
                    f"{area*100:.1f}% region of the brain parenchyma. "
                    "This diffuse, high-strength pattern is characteristic of infiltrative "
                    "glioma — the tumour cells invade surrounding healthy tissue rather than "
                    "forming a well-defined capsule, making clean surgical margins difficult. "
                    "The ResNet50 model weighted these broadly activated regions most heavily "
                    "in its classification decision."
                ),
                "cause": (
                    "High-grade gliomas (Grade III–IV, including Glioblastoma Multiforme) "
                    "arise from glial support cells — astrocytes, oligodendrocytes, or ependymal cells. "
                    "Risk factors include prior ionising radiation exposure, rare hereditary syndromes "
                    "(Li-Fraumeni, Turcot), and IDH1/IDH2 gene mutations that disrupt normal cell-cycle "
                    "regulation. The broad Grad-CAM spread suggests active angiogenesis and rapid "
                    "cellular proliferation consistent with high-grade pathology."
                ),
                "treat": (
                    "Standard of care: maximal safe surgical resection (craniotomy with awake mapping "
                    "if eloquent cortex is involved), followed by concurrent Temozolomide (TMZ) "
                    "chemotherapy and fractionated radiotherapy (60 Gy / 30 fractions — Stupp protocol). "
                    "Adjuvant TMZ cycles (6 cycles, 5/28-day schedule) are continued post-radiation. "
                    "MGMT promoter methylation status guides chemotherapy benefit prediction. "
                    "Tumour-Treating Fields (TTF / Optune) may be added for GBM. "
                    "Bevacizumab is used for recurrence. Regular MRI follow-up every 2–3 months is essential."
                )
            }
        else:
            return {
                "desc": (
                    f"Grad-CAM shows a {risk_qualifier(strength, area)} activation pattern "
                    f"(mean={strength:.2f}, peak={peak:.2f}) covering {area*100:.1f}% of the scan. "
                    "The relatively contained activation zone suggests a lower-grade or well-demarcated "
                    "glioma where tumour cells are less aggressively infiltrating adjacent tissue. "
                    "The model's attention is concentrated on a more focal cortical/subcortical region, "
                    "which correlates with slower growth kinetics typically seen in Grade I–II gliomas."
                ),
                "cause": (
                    "Low-grade gliomas (LGG, WHO Grade I–II) most commonly carry IDH mutations "
                    "(IDH1 R132H being the most frequent), 1p/19q co-deletion in oligodendrogliomas, "
                    "or ATRX loss in astrocytomas. These molecular markers indicate better prognosis. "
                    "LGGs tend to progress slowly over years but carry a risk of malignant transformation "
                    "to Grade III–IV if untreated or inadequately monitored."
                ),
                "treat": (
                    "Management depends on patient age, tumour location, and molecular profile. "
                    "Options include active surveillance with serial MRI (every 3–6 months) for "
                    "asymptomatic small LGGs, surgical resection to maximise extent of resection (EOR), "
                    "and post-operative radiotherapy (50.4 Gy) with adjuvant PCV or TMZ chemotherapy "
                    "for high-risk features (age > 40, incomplete resection, large tumour size). "
                    "Molecular testing (IDH, 1p/19q, MGMT) is essential for treatment planning."
                )
            }

    # ──────────────────────────────────────────────────────────────
    # MENINGIOMA
    # ──────────────────────────────────────────────────────────────
    elif tumor_type in ("Meningioma", "meningioma_tumor"):

        if area > 0.20:
            return {
                "desc": (
                    f"Grad-CAM identifies a {risk_qualifier(strength, area)} activation zone "
                    f"(mean={strength:.2f}, peak={peak:.2f}) along the brain surface, covering "
                    f"{area*100:.1f}% of the scan. The broad dural-based activation and relatively "
                    "high coverage percentage suggest a larger meningioma with possible involvement "
                    "of adjacent venous sinuses or bony structures. Meningiomas exhibit a sharp "
                    "interface with brain tissue (extra-axial location), which the model correctly "
                    "identifies through high peripheral activation."
                ),
                "cause": (
                    "Meningiomas arise from arachnoid cap cells of the meninges. Risk factors include "
                    "prior cranial irradiation (even low-dose, e.g., for scalp ringworm decades prior), "
                    "female sex (progesterone receptors drive ~66% of cases), NF2 gene mutations, "
                    "and long-term mobile phone radiation (controversial, under investigation). "
                    "Most are benign WHO Grade I; Grade II (atypical) and Grade III (anaplastic) "
                    "variants carry higher recurrence risk and require more aggressive management."
                ),
                "treat": (
                    "For symptomatic or growing meningiomas: complete surgical resection (Simpson "
                    "Grade I–II) is the gold standard, achieving cure in most Grade I cases. "
                    "Stereotactic Radiosurgery (SRS — Gamma Knife, CyberKnife) is preferred for "
                    "tumours ≤3 cm near eloquent structures or skull base. For larger tumours or "
                    "Grade II–III: surgery + adjuvant fractionated radiotherapy (54–60 Gy). "
                    "Recurrence surveillance with MRI at 3, 6, and 12 months then annually is standard. "
                    "Hormone therapy is generally not recommended despite receptor positivity."
                )
            }
        else:
            return {
                "desc": (
                    f"Grad-CAM highlights a well-defined, focal activation near the meningeal surface "
                    f"(mean={strength:.2f}, peak={peak:.2f}, {area*100:.1f}% coverage). "
                    "This tight, peripherally-bounded activation pattern is the hallmark of a small "
                    "extra-axial meningioma — a slow-growing tumour arising from the dural lining. "
                    "The model distinguishes this from intra-axial tumours by correctly focusing on "
                    "surface-level features rather than deep parenchymal changes."
                ),
                "cause": (
                    "Small meningiomas are frequently discovered incidentally on scans performed for "
                    "unrelated reasons. The underlying biology involves NF2/merlin pathway dysfunction, "
                    "TRAF7/AKT1/SMO gene mutations (in skull base variants), and hormonal influences. "
                    "They grow at an average rate of ~1–2 mm per year and may remain stable for decades. "
                    "Spontaneous calcification is common in long-standing lesions."
                ),
                "treat": (
                    "Small asymptomatic meningiomas: watchful waiting with annual MRI is preferred. "
                    "Intervention is triggered by growth > 3 mm/year, new neurological symptoms, "
                    "oedema, or patient preference. Surgical resection is curative for convexity "
                    "meningiomas. SRS (Gamma Knife, 12–14 Gy single fraction) achieves 90–95% "
                    "5-year control rates for tumours < 3 cm. Mifepristone (anti-progestogen) has "
                    "shown limited benefit in trials and is not standard therapy."
                )
            }

    # ──────────────────────────────────────────────────────────────
    # PITUITARY
    # ──────────────────────────────────────────────────────────────
    elif tumor_type in ("Pituitary", "pituitary_tumor"):

        if area < 0.15:
            return {
                "desc": (
                    f"Grad-CAM reveals a tightly localised activation cluster "
                    f"(mean={strength:.2f}, peak={peak:.2f}, {area*100:.1f}% coverage) "
                    "precisely centred in the sellar/parasellar region. This highly focused, "
                    "small-area activation is consistent with a microadenoma (< 10 mm), where "
                    "the abnormality is confined to the pituitary gland itself without significant "
                    "suprasellar extension. The ResNet50 model correctly isolated this subtle, "
                    "small-volume lesion through deep feature extraction on contrast-enhanced T1 patterns."
                ),
                "cause": (
                    "Pituitary microadenomas are among the most common intracranial tumours "
                    "(autopsy prevalence ~10–20%). They arise from anterior pituitary cells and are "
                    "classified by hormonal activity: Prolactinoma (most common, 40%), "
                    "GH-secreting (Acromegaly), ACTH-secreting (Cushing's disease), "
                    "TSH/FSH/LH-secreting (rare), and non-functioning. MEN1 syndrome (multiple "
                    "endocrine neoplasia type 1) is a hereditary risk factor. Most are sporadic."
                ),
                "treat": (
                    "Prolactinomas: first-line dopamine agonist therapy (Cabergoline 0.5–2 mg/week "
                    "or Bromocriptine). Normalises prolactin in 80–90% of cases with tumour shrinkage. "
                    "GH/ACTH-secreting adenomas: transsphenoidal surgery (TSS) via endoscopic "
                    "endonasal approach is first-line, achieving remission in 70–90% of microadenomas. "
                    "Adjuvant somatostatin analogues (Octreotide, Lanreotide) or Pegvisomant for "
                    "GH excess. Radiosurgery (SRS) for post-surgical residual disease. "
                    "Non-functioning adenomas < 1 cm: observe with annual MRI and endocrine evaluation."
                )
            }
        else:
            return {
                "desc": (
                    f"Grad-CAM shows a broader activation region "
                    f"(mean={strength:.2f}, peak={peak:.2f}, {area*100:.1f}% coverage) "
                    "extending beyond the sella, suggesting a macroadenoma (≥ 10 mm) with possible "
                    "suprasellar extension or cavernous sinus involvement. The wider spread of the "
                    "model's attention captures not just the tumour mass but also the compressive "
                    "effect on adjacent structures such as the optic chiasm and hypothalamus."
                ),
                "cause": (
                    "Pituitary macroadenomas often present late due to slow growth. Mass effect "
                    "symptoms include bitemporal hemianopsia (optic chiasm compression), "
                    "headache (dural stretch), and hypopituitarism (compression of normal gland). "
                    "Hormonal hypersecretion (if functional) causes Acromegaly, Cushing's, or "
                    "Hyperprolactinaemia. Cavernous sinus invasion occurs in ~25–30% of cases "
                    "and limits complete surgical resection."
                ),
                "treat": (
                    "Macroadenomas with visual compromise: urgent transsphenoidal surgery (TSS) "
                    "to decompress the optic apparatus. Endoscopic endonasal TSS is preferred over "
                    "craniotomy for most sellar/parasellar lesions. Post-operative MRI at 3 months "
                    "assesses residual disease. Residual or recurrent disease: fractionated "
                    "stereotactic radiotherapy (FSRT, 45–54 Gy) or SRS (Gamma Knife 12–15 Gy for "
                    "< 3 cm lesions > 3 mm from optic chiasm). Long-term pituitary hormone "
                    "replacement therapy (HRT) is often required for post-surgical hypopituitarism. "
                    "Lifelong endocrine and ophthalmological follow-up is essential."
                )
            }

    # ──────────────────────────────────────────────────────────────
    # NO TUMOR
    # ──────────────────────────────────────────────────────────────
    elif tumor_type in ("No Tumor", "no_tumor"):
        return {
            "desc": (
                f"Grad-CAM activation is uniformly low across the entire scan "
                f"(mean={strength:.2f}, peak={peak:.2f}, {area*100:.1f}% high-activation coverage), "
                "with no concentrated region suggesting abnormal tissue. The absence of focal "
                "hot-spots confirms that ResNet50 found no features consistent with glioma, "
                "meningioma, or pituitary adenoma in this MRI. The brain parenchyma, ventricles, "
                "and meningeal surfaces appear within normal AI-assessed limits."
            ),
            "cause": (
                "Normal brain MRI. No structural abnormality detected by the classification model. "
                "The scan may have been ordered for headache, neurological symptoms, or screening — "
                "any ongoing symptoms should be correlated clinically since AI screening does not "
                "replace comprehensive neuroradiological assessment and may not detect early or subtle lesions."
            ),
            "treat": (
                "No tumour-directed treatment required. If symptomatic, clinical evaluation and "
                "specialist neurology review are recommended to investigate non-structural causes "
                "(migraine, vascular, metabolic, or functional disorders). Routine follow-up interval "
                "should be guided by the referring clinician based on presenting symptoms."
            )
        }

    # ──────────────────────────────────────────────────────────────
    # FALLBACK
    # ──────────────────────────────────────────────────────────────
    return {
        "desc": (
            f"Grad-CAM analysis completed (mean activation={strength:.2f}, "
            f"peak={peak:.2f}, activated area={area*100:.1f}%). "
            "Insufficient pattern clarity for detailed automated explanation. "
            "Manual radiological review is strongly recommended."
        ),
        "cause": "Activation pattern does not match known tumour archetypes with sufficient confidence.",
        "treat": "Refer to a neuro-oncology or neuroradiology specialist for comprehensive evaluation."
    }