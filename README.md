<img src="FOR README.png" alt="MediLens Banner" style="width:100%; max-width:1000px;">

# MediLense: AI-Powered Prescription Analysis

MediLense is an AI-based system that converts handwritten or printed medical prescriptions
into structured, clinically meaningful JSON, and surfaces patient-facing medication insights.
It combines an OCR-free vision model (Donut) with an LLM (Phi-4 Unsloth) to normalize and
structure prescription text.


---

## Features

- Convert prescription images to structured JSON:
  - Patient name, date, doctor
  - Medication list (name, dosage, instructions, duration)
  - Clinic information
- LLM-based normalization of noisy OCR output:
  - Fixes common OCR mistakes in drug names
  - Extracts dosage, instructions, and duration fields
- Streamlit UI for:
  - Uploading a prescription image
  - Viewing structured JSON
  - Viewing human-readable patient summary
- Designed with a **human-in-the-loop (HIL)** mindset for clinical settings.

---

