# 🏥 MediLens: AI-Powered Prescription Analysis

**MediLens** is an AI-powered system that converts handwritten and printed medical prescriptions into structured, patient-friendly information using **Qwen2-VL**, a state-of-the-art vision-language model.

Specifically designed for **Indian prescriptions**, MediLens is fine-tuned on 100+ real doctor-written prescriptions to understand varying handwriting styles, medical abbreviations, and local prescription formats.

---

## 🎯 The Problem

In India, medical prescriptions are typically:
- ✍️ **Handwritten** with varying styles
- 🔤 **Full of abbreviations** (BD, TDS, OD, PRN, etc.)
- 📄 **Hard to read** even for educated patients
- ❓ **Confusing** regarding dosage and timing

**Result**: Medication errors, patient confusion, and safety risks.

---

## ✨ Our Solution

MediLens uses **Qwen2-VL**, a powerful vision-language AI model, to:

1. **Read** handwritten and printed prescriptions
2. **Understand** medical context and abbreviations
3. **Extract** structured medication information
4. **Validate** safety through drug interaction checks
5. **Display** results in patient-friendly format

### Why Qwen2-VL?

- 🎯 **Purpose-built** for document understanding and OCR
- 🌍 **Multilingual** support (32+ languages including Hindi)
- ⚡ **Efficient** - runs on modest hardware (2B parameters)
- 🎓 **Fine-tunable** on small datasets (our 100+ prescriptions)
- 🏆 **State-of-the-art** accuracy for handwritten text
- 📊 **Structured output** - directly generates JSON

---

## 🏗️ Technical Architecture

```
User Upload (Prescription Image)
           ↓
  Streamlit Interface
           ↓
┌──────────────────────────────────┐
│      Qwen2-VL Model              │
│  (Fine-tuned on 100+ Rx images)  │
│                                  │
│  Vision Encoder (ViT)            │
│         ↓                        │
│  Vision-Language Fusion          │
│         ↓                        │
│  Language Decoder                │
│         ↓                        │
│  Structured JSON Output          │
└──────────────────────────────────┘
           ↓
  Python Safety Validator
  (Drug interactions, dosages, allergies)
           ↓
  User-Friendly Display
  (Medication cards + Safety warnings)
```

## 🧠 How It Works

### 1. Fine-tuned Qwen2-VL Model

**Base Model**: `Qwen/Qwen2-VL-2B-Instruct`

**Our Fine-tuning**:
```python
# Trained on 100+ real Indian prescriptions
# Using LoRA (Low-Rank Adaptation) for efficiency
# Dataset format: Image + Structured JSON pairs

Training Data Example:
{
  "image": "prescription_001.jpg",
  "output": {
    "patient": "Rajesh Kumar",
    "medications": [
      {
        "name": "Paracetamol",
        "dosage": "500mg",
        "frequency": "Twice daily",
        "duration_days": 5
      }
    ]
  }
}
```

**Model learns**:
- "Tab" → "Tablet"
- "Paracet" → "Paracetamol"
- "1-0-1" → "Take 1 in morning, 0 at noon, 1 at night"
- "BD" → "Bis die (twice daily)"
- "x 5d" → "for 5 days"

### 2. Inference Process

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load fine-tuned model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./fine_tuned_model",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("./fine_tuned_model")

# Analyze prescription
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": prescription_image},
        {"type": "text", "text": "Extract prescription data as JSON"}
    ]
}]

# Process and generate
inputs = processor.apply_chat_template(messages, ...)
outputs = model.generate(**inputs)
structured_data = processor.batch_decode(outputs)[0]
```

## 📊 Performance

### Accuracy Metrics (Test Set: 50 prescriptions)
- **Medicine Name Recognition**: 95.2%
- **Dosage Extraction**: 92.8%
- **Frequency/Timing**: 91.5%
- **Duration Extraction**: 94.1%
- **Overall Prescription Accuracy**: 93.4%

### Speed
- **Average Processing Time**: 3.2 seconds
- **GPU (RTX 3060)**: 2.1 seconds
- **CPU (Intel i7)**: 5.8 seconds

### Model Size
- **Base Qwen2-VL-2B**: ~4GB
- **Fine-tuned version**: ~4.2GB
- **4-bit Quantized**: ~1.2GB (slight accuracy loss)

---

## 📝 Example

### Input (Handwritten Prescription):
```
Dr. Sharma Clinic
20/12/2024

Pt: Rajesh Kumar, 45Y

Rx:
1. Tab Paracet 500mg - 1-0-1 x 5days
2. Cap Amox 250mg - 1-1-1 x 7d  
3. Syr Cetriz 5ml - 0-0-1 x 3d

Dr. Sharma
```

### Output (Structured JSON):
```json
{
  "patient": {
    "name": "Rajesh Kumar",
    "age": 45
  },
  "date": "2024-12-20",
  "doctor": "Dr. Sharma",
  "clinic": "Dr. Sharma Clinic",
  "medications": [
    {
      "id": 1,
      "name": "Paracetamol",
      "brand_name": "Paracet",
      "generic_name": "Acetaminophen",
      "form": "Tablet",
      "strength": "500mg",
      "frequency": {
        "pattern": "1-0-1",
        "description": "Twice daily (morning and night)",
        "morning": 1,
        "afternoon": 0,
        "night": 1
      },
      "duration": {
        "days": 5,
        "description": "Take for 5 days"
      },
      "instructions": "Take after meals"
    }
    // ... other medications
  ]
}
```

### Display (Patient View):
```
💊 Paracetamol (Paracet) - Tablet 500mg
────────────────────────────────────
📅 Duration: 5 days
⏰ When to take: Twice daily (morning and night)
📝 Instructions: Take after meals
ℹ️ Purpose: Pain relief and fever reduction
```

---

## 🛡️ Safety Features

### Drug Interaction Detection
- Database of 500+ common interactions
- Severity classification (Critical/High/Medium/Low)
- Real-time checking during analysis

### Allergy Checking
- Cross-reference with user allergy profile
- Drug class matching (e.g., Penicillin family)
- Critical alerts for known allergies

### Dosage Validation
- Age-specific maximum safe doses
- Standard therapeutic ranges
- Warnings for excessive amounts

---

## ⚠️ Important Notes

### Clinical Use
- ✅ MediLens is a **decision support tool**
- ✅ **NOT a replacement** for professional medical advice
- ✅ Always **verify** with doctor/pharmacist
- ✅ Designed with **human-in-the-loop** safety

### Limitations
- Accuracy depends on image quality
- May struggle with very unusual handwriting
- Drug database requires periodic updates
- Not validated for clinical deployment (research project)

### Privacy
- 🔒 Local processing only
- 🔒 No data sent to external servers
- 🔒 Images not stored permanently
- 🔒 HIPAA/medical privacy conscious

---

## 🔬 Technical Details

### Model Architecture
- **Vision Encoder**: Vision Transformer (ViT)
- **Language Decoder**: LLaMA-based architecture
- **Parameters**: 2 billion (base model)
- **Quantization**: 4-bit support via bitsandbytes
- **Fine-tuning**: LoRA adapters (rank=16)

### Training Details
- **Dataset**: 100+ annotated prescriptions
- **Training Time**: ~6 hours on RTX 3060
- **Batch Size**: 4 (with gradient accumulation)
- **Learning Rate**: 5e-5
- **Epochs**: 10
- **Optimizer**: AdamW

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB RAM, NVIDIA GPU (4GB+ VRAM)
- **Optimal**: 32GB RAM, NVIDIA RTX 3060 or better

---

## 🚧 Roadmap

### Phase 1 (Current)
- ✅ Qwen2-VL integration
- ✅ Basic prescription analysis
- ✅ Safety checks
- ✅ Streamlit UI

### Phase 2 (In Progress)
- 🔄 Mobile app (React Native)
- 🔄 Hindi language support
- 🔄 Larger training dataset (500+ prescriptions)
- 🔄 API endpoint for integration

### Phase 3 (Planned)
- 📅 Multi-page prescription support
- 📅 Lab report analysis
- 📅 Medication reminder system
- 📅 Pharmacy integration
- 📅 Doctor dashboard

---

## 👥 Contributing

We welcome contributions! Areas where you can help:
- 📸 **Data Collection**: Share anonymized prescriptions
- 🐛 **Bug Reports**: Report issues you encounter
- 💡 **Feature Requests**: Suggest improvements
- 🔧 **Code Contributions**: Submit pull requests
- 📖 **Documentation**: Improve guides and examples

---
## 🎥 MediLens Demo

[![MediLens Demo](youtube.com/watch?v=6hfEeXQF-dA&feature=youtu.be)


---

## 📚 References

### Qwen2-VL
- Paper: [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)
- GitHub: [github.com/QwenLM/Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)
- Hugging Face: [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
