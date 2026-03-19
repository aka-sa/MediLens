from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import json
import re
from PIL import Image

# Load processor + model
model_id = "Qwen/Qwen2-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def extract_prescription(image_path, output_file="prescription5.json"):
    # Open image
    image = Image.open(image_path).convert("RGB")

    # Instruction prompt
    prompt = """
You are a medical prescription extraction assistant.
Read the prescription image carefully and output ONLY valid JSON.

Required JSON fields:
- patient_name: string
- date: string (if not mentioned, return empty string "")
- doctor: string
- summary: Write full instructions in plain language about when and how to take the medicine(s).
  If there are alternatives (written as OR), clearly explain that the patient should take either
  medicine A OR medicine B, not both.
- medications: list of objects. Each object may represent:
    - a single medicine, OR
    - a group of alternative medicines (if the prescription uses 'OR').
    
  Fields inside each object:
    - name: string or list of alternative names if 'OR' is mentioned
    - instruction: string or list (aligned with names if alternatives exist and expand shorthand schedules into plain language)
    - duration: string or list (aligned with names if alternatives exist)

IMPORTANT RULES:
1. If 'OR' is written, put both medicines inside the SAME object (not separate ones).
2. Do not invent medicine names. Use exactly what is written in the prescription.
3. Convert shorthand schedules like "1-0-1" or "1-1-1" into plain text:
   - "1-0-1" → "Take one dose in the morning and one dose in the evening" (2 times daily).
   - "1-1-1" → "Take one dose in the morning, one at midday, and one in the evening" (3 times daily).
   - "1-0-0" → "Take one dose in the morning only".
   - "0-1-0" → "Take one dose at midday only".
   - "0-0-1" → "Take one dose in the evening only".
4. Instructions should be written in simple plain English.
5. Always return valid JSON only. Do not add explanations or text outside JSON.

"<|vision_start|><|image_pad|><|vision_end|>\n"
Return only JSON, no extra text.
"""


    # Prepare inputs (include both text and image)
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1024)

    # Decode
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print("Raw Model Output:\n", response)

    # Try extracting JSON
    parsed = None
    try:
        parsed = json.loads(response)
    except Exception:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except Exception:
                parsed = {"error": "JSON found but parsing failed", "raw": response}
        else:
            parsed = {"error": "No JSON found", "raw": response}

    # Save JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    print(f"\nSaved structured output to {output_file}")
    return parsed


if __name__ == "__main__":
    result = extract_prescription("hand.png")
    print("\nFinal Parsed JSON:\n", result)
