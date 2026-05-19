# DermaGuard 🩺
**Multi-Agent AI System for Skin Lesion Triage**
UIR — AI & Big Data Program | S8 Integrated Project | 2025–2026

---

## Project Overview
DermaGuard is a multi-agent AI system that classifies skin lesion images as
Benign, Malignant, or Suspicious using a trained PyTorch CNN (ResNet-50),
generates a structured clinical report via Gemini API, and includes a
human-in-the-loop approval checkpoint.

---

## Setup

### 1. Clone & install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Prepare ISIC dataset
Download from https://challenge.isic-archive.com and organize as:
```
data/isic/
  train/
    benign/      *.jpg
    malignant/   *.jpg
    suspicious/  *.jpg
  val/
    benign/      *.jpg
    malignant/   *.jpg
    suspicious/  *.jpg
```

### 4. Train the CNN model
```bash
python model/train.py
```

### 5. Evaluate the model
```bash
python model/evaluate.py
```

### 6. Run the full pipeline
```bash
python main.py path/to/lesion_image.jpg
```

---

## Project Structure
```
dermaguard/
├── main.py                  ← Entry point
├── config.py                ← Settings & paths
├── requirements.txt
├── agents/
│   ├── classifier_agent.py  ← Agent 1: CNN Classifier
│   ├── report_agent.py      ← Agent 2: Report Generator
│   └── orchestrator.py      ← Agent 3: Orchestrator + logging
├── tools/
│   ├── cnn_tool.py          ← PyTorch model wrapped as CrewAI tool
│   ├── report_tool.py       ← Gemini API report generation
│   └── hitl_tool.py         ← Human-in-the-loop checkpoint
├── model/
│   ├── model_def.py         ← ResNet-50 architecture
│   ├── train.py             ← Training script
│   └── evaluate.py          ← Confusion matrix & metrics
├── data/
│   ├── dataset.py           ← ISIC dataset loader
│   └── transforms.py        ← Image augmentations
├── logs/                    ← Auto-generated JSON logs
└── outputs/                 ← Approved reports saved here
```

---

## Tech Stack
| Component | Technology |
|---|---|
| Agent Framework | CrewAI |
| LLM Backend | Gemini 1.5 Flash (free API) |
| DL Framework | PyTorch ≥ 2.0 + ResNet-50 |
| Dataset | ISIC Skin Lesion Archive |
| Language | Python 3.10+ |

---

## Academic Integrity
LLM tools were used as coding assistants.
Every line of code was reviewed, understood, and can be explained during the oral defense.
