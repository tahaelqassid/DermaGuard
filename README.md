# 🩺 DermaGuard UIR
**Multi-Agent AI System for Skin Lesion Triage**

> *"The goal is not to build the most complex system. It is to build a system where every component exists for a reason, every agent has a job, and you can explain why."* —

---

## 📌 Overview

DermaGuard is a multi-agent AI system that classifies skin lesion images as **Benign**, **Malignant**, or **Suspicious** using a fine-tuned ResNet-50 CNN, generates a structured clinical report via OpenAI GPT-4o-mini, and enforces mandatory human approval before saving anything.

Built for the **S8 Integrated Project — Building Multi-Agent AI Systems**
UIR · AI & Big Data Program · 2025–2026

---

## 🏗️ Architecture
Skin Lesion Image
↓
🧠 Agent 3 — Orchestrator      (coordinates · logs · handles errors)
↓
🔬 Agent 1 — CNN Classifier    (ResNet-50 · ISIC HAM10000)
↓
📝 Agent 2 — Report Generator  (OpenAI GPT-4o-mini · 5-section report)
↓
👨‍⚕️ Human-in-the-Loop          (approve · reject · revise)
↓
✅ Approved Clinical Report    (saved to outputs/)

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | CrewAI |
| LLM Backend | OpenAI GPT-4o-mini |
| Deep Learning | PyTorch ≥ 2.0 · ResNet-50 |
| Dataset | HAM10000 / ISIC Archive (10k images) |
| Web Interface | Flask · HTML/CSS/JS |
| Language | Python 3.11 |

---

## 📁 Project Structure

dermaguard/
├── main.py                  # CLI entry point
├── app.py                   # Flask web server
├── dermaguard_uir.html      # Clinical web interface
├── config.py                # All settings and paths
├── requirements.txt         # Dependencies
├── prepare_data.py          # Organizes HAM10000 dataset
├── .env.example             # Environment template
│
├── agents/
│   ├── orchestrator.py      # Agent 3 — coordinates pipeline
│   ├── classifier_agent.py  # Agent 1 — CNN classifier
│   └── report_agent.py      # Agent 2 — report generator
│
├── tools/
│   ├── cnn_tool.py          # ResNet-50 wrapped as CrewAI tool
│   ├── report_tool.py       # OpenAI report generation
│   └── hitl_tool.py         # Human-in-the-loop checkpoint
│
├── model/
│   ├── model_def.py         # ResNet-50 architecture
│   ├── train.py             # Training script
│   └── evaluate.py          # Metrics + confusion matrix
│
└── data/
├── dataset.py           # ISIC dataset loader
└── transforms.py        # Data augmentation pipeline

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/tahaelqassid/DermaGuard.git
cd DermaGuard
```

### 2. Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
```
Edit `.env` and add your OpenAI API key:
OPENAI_API_KEY=your_key_here
OPENAI_MODEL_NAME=gpt-4o-mini

---

## 📦 Dataset Setup

Download **HAM10000** from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and extract to a folder, then run:

```bash
python prepare_data.py
```

Expected output:
train/benign:     5570 images
train/malignant:  1301 images
train/suspicious: 1141 images
val/benign:        887 images
val/malignant:     326 images
val/suspicious:    285 images

---

## 🧠 Train the Model

```bash
python model/train.py
```

- ResNet-50 fine-tuned on HAM10000
- 15 epochs · batch size 32 · AdamW optimizer
- Class weights [1.0, 5.0, 4.0] for imbalance correction
- Best model saved to `model/dermaguard_cnn.pth`

---

## 📊 Evaluate the Model

```bash
python model/evaluate.py
```

| Metric | Value |
|---|---|
| Validation Accuracy | 80.0% |
| Malignant Recall | 44% |
| Macro F1 | 0.66 |
| Benign F1 | 0.90 |

Confusion matrix saved to `outputs/confusion_matrix.png`

---

## 🖥️ Run the Web Interface

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

Features:
- Upload skin lesion image (JPEG, PNG, TIFF, BMP)
- Live agent pipeline visualization
- Real CNN classification with probability bars
- Structured 5-section clinical report
- Human approval checkpoint (Approve / Reject / Revise)
- Download approved report as `.txt`
- Session history tracking

---

## 💻 Run via CLI

```bash
python main.py path/to/image.jpg
```

---

## 📋 Minimum Requirements Met

- ✅ 2 specialist agents + 1 orchestrator
- ✅ PyTorch CNN trained on HAM10000 with evaluation
- ✅ DL model wrapped as functional CrewAI tool
- ✅ Human-in-the-loop checkpoint with revision loop
- ✅ Error handling — no crashes
- ✅ JSON logging with UTC timestamps
- ✅ Web interface with clinical dashboard
- ✅ Reproducible setup via this README

---






**Program:** AI & Big Data · UIR · S8 · 2025–2026

---
