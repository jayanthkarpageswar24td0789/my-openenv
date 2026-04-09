---
title: PharmaGuard AI
emoji: "💊"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 💊 PharmaGuard AI - Clinical Prescription Validator

> AI-powered clinical pharmacist that prevents medication errors before they reach patients.

An OpenEnv-powered AI environment where an intelligent agent acts as a clinical pharmacist, validating prescriptions for patient safety, drug interactions, and dosage correctness.

---

## 🚨 Problem

Medication errors are a major global healthcare issue:

- 200,000+ deaths annually due to adverse drug events
- $30B+ healthcare costs from preventable medication errors
- Pharmacists manually review 100+ prescriptions daily

Many errors involve:

- Drug interactions
- Incorrect dosage
- Patient-specific contraindications

---

## 💡 Solution

PharmaGuard AI simulates real-world pharmacy validation using:

- 🧠 Hybrid AI Agent (Rules + LLM)
- ⚕️ Clinical safety reasoning
- 🔍 Explainable decision-making
- 🧪 Multi-scenario evaluation environment

---

## ⚙️ System Architecture

```text
Client → FastAPI Server → OpenEnv Environment → AI Agent
                                              |- Clinical Rules
                                              '- LLM Reasoning
```

---

## 🚀 Key Features

### ✅ Hybrid Intelligence

- Rule-based clinical safety checks
- LLM-powered reasoning (fallback-safe)

### ✅ Explainable AI (XAI)

- Every decision includes human-readable reasoning

### ✅ Robust and Reliable

- Safe fallback system to avoid crashes
- Retry-based API communication

### ✅ Real-World Simulation

- Based on real pharmacology principles
- Covers multiple clinical scenarios

---

## 🧪 Tasks (Easy -> Hard)

### 🔹 Task 1: Contraindication Detection

Detect unsafe ingredients for specific patients.

Example: Lactose in diabetic patient

### 🔹 Task 2: Dosage Adjustment

Adjust drug dosage based on organ function.

Example: NSAIDs in CKD patients

### 🔹 Task 3: Drug Interaction Analysis

Detect multi-drug interaction risks.

Example: Warfarin + Antibiotics -> bleeding risk

---

## 🧠 Agent Design

### Step 1: Clinical Rules (Deterministic)

- Warfarin + high-risk interacting drugs -> REJECT
- Kidney disease + nephrotoxic drug -> MODIFY

### Step 2: LLM Reasoning

- Context-aware medical decision making

### Step 3: Safe Fallback

- Always returns valid output (no crashes)

---

## 📊 Example Output

```text
[START] task=pharma_task_1 episode=1
[STEP] step=1 action=REJECT reward=0.0 done=false error=null
[END] task=pharma_task_1 score=1.0 steps=1
```

---

## ⚡ Quick Start

### Local Run

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run inference (Windows)
set LOCAL_DEV=1
python inference.py

# Run inference (macOS/Linux)
export LOCAL_DEV=1
python inference.py
```

### Docker (HF Spaces Compatible)

```bash
docker build -t pharma-env .
docker run -p 7860:7860 pharma-env
```

---

## 🔌 API

### POST /reset

Returns a new patient case.

### POST /step

Submit decision payload:

```json
{
  "decision": "REJECT",
  "reasoning": "Drug interaction risk",
  "suggested_changes": "Use safer alternative"
}
```

---

## 📈 Scoring System

| Score | Meaning |
| --- | --- |
| 1.0 | Fully correct |
| 0.7 | Partially correct |
| 0.4 | Weak reasoning |
| 0.0 | Unsafe decision |

---

## 🧪 Real-World Medical Coverage

- Drug interactions (Warfarin, NSAIDs)
- Chronic diseases (Diabetes, CKD)
- Elderly polypharmacy cases
- Dosage safety checks

---

## 🔬 Data Sources

- FDA Drug Interaction Data
- RxNorm (NIH)
- Clinical guidelines (synthetic cases)

---

## ⚠️ Disclaimer

For research and educational use only.

Not intended for real clinical decisions.

---

## 🌐 Why This Environment Matters

This project demonstrates the power of OpenEnv in real-world domains:

- Converts healthcare decision-making into a structured agent environment
- Enables benchmarking of AI agents on safety-critical tasks
- Supports reinforcement learning for medical reasoning
- Shows how LLMs + rules can work together in production systems

This makes PharmaGuard AI not just a demo, but a scalable foundation for real-world AI deployment.

---

## 📊 Evaluation Capability

PharmaGuard AI enables benchmarking of different AI agents:

- Compare rule-based vs LLM-based performance
- Measure safety accuracy across tasks
- Track reward scores across scenarios

This allows researchers to evaluate how well AI systems handle real-world clinical decision-making.

---

## 🔮 Future Scope

- Integration with real-world drug databases (FDA APIs)
- Support for personalized medicine (genomics, allergies)
- Multi-step clinical decision workflows
- RL training for improving agent performance
- Deployment as clinical decision support tool

This project can evolve into a real-world AI assistant for pharmacists.

---

## 🏆 Hackathon Submission

- Event: Meta OpenEnv Hackathon
- Track: Real-World AI Environments

### 🔥 Highlights

- OpenEnv compliant
- Docker deployed (HF Spaces)
- Hybrid AI agent (Rules + LLM)
- Explainable medical reasoning
- Zero-crash inference system

### 👨‍💻 Author

Jayanth

HF Space: https://huggingface.co/spaces/YOUR_USERNAME/pharma-env

### 💡 Vision

"AI-assisted pharmacists to reduce medication errors and improve patient safety globally."

---

## 🧪 Strong Medical Test Cases (Append Safely)

Add these entries inside `data/task_cases.json`.

- Do not remove old cases.
- Append only.

### ✅ Case 1: Elderly + Polypharmacy

```json
{
  "patient": {
    "age": 75,
    "conditions": ["hypertension"],
    "current_medications": ["warfarin"]
  },
  "proposed_formula": {
    "active_ingredients": ["aspirin"],
    "frequency": "once daily"
  },
  "correct_action": "REJECT"
}
```

Reason: Bleeding risk (warfarin + aspirin)

### ✅ Case 2: Kidney Disease Risk

```json
{
  "patient": {
    "age": 60,
    "conditions": ["kidney_disease"],
    "current_medications": []
  },
  "proposed_formula": {
    "active_ingredients": ["ibuprofen"],
    "frequency": "three times daily"
  },
  "correct_action": "MODIFY"
}
```

Reason: NSAIDs can be harmful in CKD

### ✅ Case 3: Pregnancy Contraindication

```json
{
  "patient": {
    "age": 30,
    "conditions": ["pregnancy"],
    "current_medications": []
  },
  "proposed_formula": {
    "active_ingredients": ["isotretinoin"],
    "frequency": "once daily"
  },
  "correct_action": "REJECT"
}
```

Reason: Highly teratogenic drug

### ✅ Case 4: Diabetes + Sugar Excipient

```json
{
  "patient": {
    "age": 55,
    "conditions": ["diabetes"],
    "current_medications": []
  },
  "proposed_formula": {
    "active_ingredients": ["cough syrup"],
    "excipients": ["sucrose"],
    "frequency": "twice daily"
  },
  "correct_action": "MODIFY"
}
```

Reason: Sugar content risk in diabetes

---

**Built for safer healthcare through AI**
