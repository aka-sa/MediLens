#!/usr/bin/env python3
"""
💊 PRESCRIPTION PRO — All-in-One Prescription Analyst
✅ Extract from image
✅ Analyze medicines (Indian brands supported)
✅ Check drug interactions
✅ Allergy safety checker
✅ Generate patient handout in Hindi & Marathi
✅ Streamlit Web UI

Run: streamlit run prescription_pro.py
"""

import json
import re
import logging
import time
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PrescriptionProAgent:
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct", device="auto", torch_dtype=torch.float16):
        """
        Initialize agent with multimodal LLM for prescription analysis.
        """
        logger.info(f"🚀 Loading model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device
        )
        logger.info("✅ Model loaded successfully.")

        # Predefined hints for common Indian brand drugs
        self.brand_hints = {
            "enzoflam": "likely Aceclofenac + Paracetamol (NSAID + Analgesic)",
            "pan-d": "Pantoprazole + Domperidone (PPI + Prokinetic)",
            "hexigel": "Topical antiseptic/analgesic gel for gums (Chlorhexidine + Lidocaine)",
            "augmentin": "Amoxicillin + Clavulanate (Antibiotic)",
            "dolo": "Paracetamol (Analgesic/Antipyretic)",
            "crocin": "Paracetamol (Analgesic/Antipyretic)",
            "rantac": "Ranitidine (H2 Blocker)",
            "nexium": "Esomeprazole (PPI)",
        }

        # Common allergens → trigger flags
        self.allergen_triggers = {
            "penicillin": ["amoxicillin", "augmentin", "ampicillin"],
            "nsaid": ["diclofenac", "aceclofenac", "ibuprofen", "aspirin", "enzoflam", "voveran"],
            "sulfonamide": ["sulfamethoxazole", "bactrim", "septran"],
            "paracetamol": ["paracetamol", "dolo", "crocin", "calpol"],
        }

    # =============================================
    # UTIL: Clean Medicine Name
    # =============================================
    def _clean_medicine_name(self, name):
        if not isinstance(name, str):
            return str(name)
        noise_words = [
            "massage", "apply", "use", "take", "tablet", "capsule", "oral", "topical",
            "gel", "cream", "ointment", "paint", "solution", "syrup", "once", "daily", "twice"
        ]
        name = name.strip()
        for word in noise_words:
            name = re.sub(rf"\b{word}\b", "", name, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", name).strip()

    # =============================================
    # UTIL: Get Generic Hint for Brand Name
    # =============================================
    def _get_generic_hint(self, name):
        name_lower = name.lower()
        for key, hint in self.brand_hints.items():
            if key in name_lower:
                return f"Note: {hint}"
        return ""

    # =============================================
    # STEP 1: EXTRACT PRESCRIPTION FROM IMAGE
    # =============================================
    def extract_prescription(self, image):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._get_extraction_prompt()}
                ]
            }
        ]

        try:
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=1024)
            raw_output = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            logger.info("✅ Prescription extracted.")
            return self._clean_and_parse_json(raw_output)

        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return {"error": "Extraction failed", "details": str(e)}

    def _get_extraction_prompt(self):
        return """
You are a medical prescription extraction assistant. Read the image and output ONLY valid JSON.

Required fields:
- patient_name: string
- date: string (empty if not found)
- doctor: string
- summary: plain English summary of all instructions
- medications: list of objects with:
    - name: string or list (if 'OR' alternatives)
    - instruction: string or list (convert "1-0-1" → plain text)
    - duration: string or list

Rules:
1. If 'OR' → group alternatives in same object.
2. Convert dosing shorthand: e.g., “1-0-1” → “Take one dose in morning and evening”.
3. Return ONLY raw JSON. No "assistant", markdown, or prefixes.

Example: {"patient_name": "John", ...}
"<tool_call><tool_call><tool_call>"
Return ONLY raw JSON.
"""

    # =============================================
    # STEP 2: ANALYZE EACH MEDICINE (SMART GUESS MODE)
    # =============================================
    def analyze_medicine(self, name, instruction="", duration=""):
        clean_name = self._clean_medicine_name(name)
        hint = self._get_generic_hint(clean_name)
        prompt = self._get_analysis_prompt(clean_name, instruction, duration, hint)

        try:
            messages = [{"role": "user", "content": prompt}]
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_prompt], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=512)
            raw_output = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            logger.info(f"✅ Analyzed: {clean_name}")

            result = self._clean_and_parse_json(raw_output)
            if isinstance(result, dict):
                result["original_name"] = name
            return result

        except Exception as e:
            logger.error(f"❌ Analysis failed for {name}: {e}")
            return {"medicine": clean_name, "original_name": name, "error": "Analysis failed", "details": str(e)}

    def _get_analysis_prompt(self, name, instruction, duration, hint=""):
        return f"""
You are a clinical pharmacist AI. Analyze this medicine using general medical knowledge.

Medicine: {name}
Instruction: {instruction}
Duration: {duration}
{hint}

If brand name, infer generic or class. If unsure, guess based on suffix/prefix/context.
NEVER return "Unknown".

Return ONLY raw JSON with:
{{
  "medicine": "{name}",
  "therapeutic_class": "guess if needed",
  "common_uses": ["guess if needed"],
  "typical_dosage_range": "infer from instruction",
  "safety_notes": "if unsure, 'Use as directed'",
  "possible_interactions": ["if unsure, 'None known'"],
  "analysis_summary": "1-2 sentence note",
  "red_flags": ["if unsure, 'None identified'"]
}}

Example for "Enzoflam":
{{
  "medicine": "Enzoflam",
  "therapeutic_class": "NSAID / Analgesic",
  "common_uses": ["Pain relief", "Inflammation reduction"],
  "typical_dosage_range": "1 tablet twice daily",
  "safety_notes": "Avoid in peptic ulcer or asthma",
  "possible_interactions": ["Warfarin", "Aspirin"],
  "analysis_summary": "Likely contains Aceclofenac or Diclofenac.",
  "red_flags": ["GI bleeding risk"]
}}

"<tool_call><tool_call><tool_call>"
Return ONLY raw JSON.
"""

    # =============================================
    # STEP 3: CHECK DRUG INTERACTIONS
    # =============================================
    def check_interactions(self, medicine_names):
        if len(medicine_names) < 2:
            return {"interaction_found": False, "pairs": []}

        prompt = self._get_interaction_prompt(medicine_names)
        try:
            messages = [{"role": "user", "content": prompt}]
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_prompt], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=512)
            raw_output = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            logger.info("✅ Interaction check completed.")
            return self._clean_and_parse_json(raw_output)

        except Exception as e:
            logger.error(f"❌ Interaction check failed: {e}")
            return {"error": "Interaction check failed", "details": str(e)}

    def _get_interaction_prompt(self, med_list):
        meds_str = ", ".join(med_list)
        return f"""
You are a drug safety AI. Check for significant interactions between: [{meds_str}].

Return ONLY raw JSON:
{{
  "interaction_found": true/false,
  "pairs": [
    {{
      "drug_a": "Name",
      "drug_b": "Name",
      "risk": "e.g., Bleeding risk",
      "severity": "Low/Moderate/High",
      "recommendation": "e.g., Monitor INR"
    }}
  ]
}}

Rules:
- Only report well-known, clinically significant.
- If none, return {{"interaction_found": false, "pairs": []}}.
- Return ONLY raw JSON.

"<tool_call><tool_call><tool_call>"
Return ONLY raw JSON.
"""

    # =============================================
    # STEP 4: ALLERGY CHECKER
    # =============================================
    def check_allergies(self, medications, allergies):
        """
        Check if any medicine triggers patient's allergies.
        """
        alerts = []
        allergy_list = [a.strip().lower() for a in allergies] if allergies else []

        for med in medications:
            med_name = med.get("medicine", "").lower()
            for allergy in allergy_list:
                triggers = self.allergen_triggers.get(allergy, [])
                if any(trigger in med_name for trigger in triggers):
                    alerts.append({
                        "medicine": med.get("medicine", ""),
                        "allergy": allergy,
                        "alert": f"⚠️ {med.get('medicine')} may trigger {allergy} allergy."
                    })

        return alerts

    # =============================================
    # STEP 5: GENERATE PATIENT HANDOUT (Hindi + Marathi)
    # =============================================
    def generate_patient_handout(self, medications, language="en"):
        """
        Generate simple patient instructions in English, Hindi, or Marathi.
        """
        handout = []

        for med in medications:
            name = med.get("medicine", "")
            instr = med.get("instruction", "as directed")
            duration = med.get("duration", "as prescribed")

            if language == "hi":  # Hindi
                handout.append(f"• {name}: {instr} लें। अवधि: {duration}।")
            elif language == "mr":  # Marathi
                handout.append(f"• {name}: {instr} घ्या. कालावधी: {duration}.")
            else:  # English
                handout.append(f"• {name}: {instr}. Duration: {duration}.")

        return "\n".join(handout)

    # =============================================
    # UTIL: Robust JSON Parser
    # =============================================
    def _clean_and_parse_json(self, text):
        text = text.strip()
        if "assistant" in text:
            text = text.split("assistant")[-1].strip()
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end] if end != -1 else text[start:]
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end] if end != -1 else text[start:]

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"error": "No JSON found", "raw": text}

        json_str = match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return {"error": "JSON parsing failed", "details": str(e), "raw": text}

    # =============================================
    # MAIN: FULL ANALYSIS PIPELINE
    # =============================================
    def full_analysis(self, image, patient_allergies=None):
        # Step 1: Extract
        extraction = self.extract_prescription(image)
        if "error" in extraction:
            return {"error": "Extraction failed", "details": extraction.get("details", "")}

        # Step 2: Analyze meds
        meds = extraction.get("medications", [])
        analyzed_meds = []
        clean_names = []

        for med in meds:
            names = med["name"] if isinstance(med["name"], list) else [med["name"]]
            instrs = med["instruction"] if isinstance(med["instruction"], list) else [med["instruction"]] * len(names)
            durs = med["duration"] if isinstance(med["duration"], list) else [med["duration"]] * len(names)

            for i, name in enumerate(names):
                instr = instrs[i] if i < len(instrs) else ""
                dur = durs[i] if i < len(durs) else ""
                analysis = self.analyze_medicine(name, instr, dur)
                analyzed_meds.append(analysis)
                clean_names.append(self._clean_medicine_name(name))

        # Step 3: Interactions
        interactions = self.check_interactions(clean_names)

        # Step 4: Allergy check
        allergy_alerts = self.check_allergies(analyzed_meds, patient_allergies)

        # Step 5: Generate handouts
        handout_en = self.generate_patient_handout(analyzed_meds, "en")
        handout_hi = self.generate_patient_handout(analyzed_meds, "hi")
        handout_mr = self.generate_patient_handout(analyzed_meds, "mr")

        # Final report
        report = {
            "patient_name": extraction.get("patient_name", ""),
            "date": extraction.get("date", ""),
            "doctor": extraction.get("doctor", ""),
            "summary": extraction.get("summary", ""),
            "medications": analyzed_meds,
            "drug_interactions": interactions,
            "allergy_alerts": allergy_alerts,
            "patient_handout": {
                "english": handout_en,
                "hindi": handout_hi,
                "marathi": handout_mr
            },
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "warnings": []
        }

        # Add interaction warnings
        if interactions.get("interaction_found") and "pairs" in interactions:
            for pair in interactions["pairs"]:
                if pair.get("severity", "").lower() in ["high", "moderate"]:
                    msg = f"⚠️ {pair['drug_a']} + {pair['drug_b']}: {pair['risk']} (Severity: {pair['severity']})"
                    report["warnings"].append(msg)

        # Add allergy warnings
        for alert in allergy_alerts:
            report["warnings"].append(alert["alert"])

        return report


# =============================================
# STREAMLIT WEB UI
# =============================================
def main():
    st.set_page_config(page_title="💊 Medilens", layout="centered")
    st.title("💊 Medilens")
    st.markdown("Upload a prescription image → Get full analysis + patient handout in Hindi/Marathi")

    uploaded_file = st.file_uploader("📤 Upload Prescription Image", type=["png", "jpg", "jpeg"])

    allergies = st.text_input("🩺 Patient Allergies (comma-separated, e.g., penicillin, nsaid)")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Prescription", use_column_width=True)

        with st.spinner("🧠 Analyzing prescription... (may take 20-60 sec)"):
            agent = PrescriptionProAgent()
            allergy_list = [a.strip() for a in allergies.split(",")] if allergies else None
            result = agent.full_analysis(image, allergy_list)

        if "error" in result:
            st.error(f"❌ {result['error']}")
            st.json(result)
        else:
            # Display Summary
            st.success("✅ Analysis Complete!")
            st.subheader("📋 Patient Summary")
            st.write(f"**Name**: {result['patient_name']}")
            st.write(f"**Date**: {result['date']}")
            st.write(f"**Doctor**: {result['doctor']}")

            # Warnings
            if result["warnings"]:
                st.subheader("🚨 Warnings")
                for warning in result["warnings"]:
                    st.warning(warning)

            # Medications
            st.subheader("💊 Medications")
            for med in result["medications"]:
                with st.expander(f"{med.get('medicine', 'Unknown')}"):
                    st.json(med)

            # Patient Handouts
            st.subheader("📄 Patient Instructions")
            tab1, tab2, tab3 = st.tabs(["English", "हिंदी", "मराठी"])

            with tab1:
                st.text(result["patient_handout"]["english"])
            with tab2:
                st.text(result["patient_handout"]["hindi"])
            with tab3:
                st.text(result["patient_handout"]["marathi"])

            # Raw JSON
            st.subheader("💾 Raw JSON Output")
            st.json(result)

            # Download button
            st.download_button(
                "📥 Download Full Report (JSON)",
                data=json.dumps(result, indent=2, ensure_ascii=False),
                file_name=f"prescription_report_{int(time.time())}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()