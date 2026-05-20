"""
DermaGuard UIR — Flask Backend
Connects the web interface to the real PyTorch CNN model
Run: python app.py
Then open: http://localhost:5000
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (MODEL_PATH, NUM_CLASSES, CLASS_NAMES,
                    CONFIDENCE_THRESHOLD, LOG_DIR, OUTPUT_DIR)
from model.model_def import load_model
from data.transforms import inference_transforms

import torch
import torch.nn.functional as F
from PIL import Image

# ── SETUP ─────────────────────────────────────────────
app = Flask(__name__, static_folder='.', template_folder='.')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads_web')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'web_actions.log'),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# ── LOAD MODEL ONCE AT STARTUP ─────────────────────────
print("Loading DermaGuard CNN model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

try:
    model = load_model(MODEL_PATH, NUM_CLASSES, str(device))
    print(f"✓ Model loaded successfully on {device}")
except Exception as e:
    print(f"✗ Model not found: {e}")
    print("  Train the model first: python model/train.py")

# ── HELPERS ───────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_action(action, payload):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "source": "web_interface",
        "action": action,
        "payload": payload
    }
    logging.info(json.dumps(entry))

def classify_image(image_path):
    """Run CNN inference on image, return dict with results."""
    if model is None:
        raise RuntimeError("Model not loaded. Run python model/train.py first.")

    image = Image.open(image_path).convert("RGB")
    tensor = inference_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().tolist()

    pred_idx    = int(torch.tensor(probs).argmax())
    pred_class  = CLASS_NAMES[pred_idx]
    confidence  = round(probs[pred_idx] * 100, 1)
    flag_review = probs[pred_idx] < CONFIDENCE_THRESHOLD

    return {
        "predicted_class": pred_class,
        "confidence":      confidence,
        "probabilities": {
            "Benign":     round(probs[0] * 100, 1),
            "Malignant":  round(probs[1] * 100, 1),
            "Suspicious": round(probs[2] * 100, 1),
        },
        "flag_for_review": flag_review,
    }

def generate_report(pred_class, confidence, probs, filename):
    """Generate clinical report text."""
    date = datetime.now().strftime("%d %B %Y")
    time = datetime.now().strftime("%H:%M:%S")

    severity_map = {
        "Benign":     "LOW RISK",
        "Malignant":  "HIGH RISK — URGENT",
        "Suspicious": "MODERATE RISK"
    }
    action_map = {
        "Benign": (
            "Routine follow-up recommended in 6–12 months. "
            "Patient should perform regular self-examination and "
            "report any changes in size, color, or morphology."
        ),
        "Malignant": (
            "URGENT: Immediate dermatological referral required. "
            "Excisional biopsy strongly recommended within 2 weeks. "
            "Patient should be advised to avoid sun exposure and seek immediate specialist consultation."
        ),
        "Suspicious": (
            "Schedule dermatological follow-up within 4 weeks. "
            "Dermoscopic examination recommended. "
            "Monitor for changes in size, color, border irregularity, or morphology."
        )
    }
    assessment_map = {
        "Benign": (
            f"The ResNet-50 CNN analysis indicates benign characteristics with {confidence}% confidence. "
            "Lesion morphology is consistent with common benign dermatological conditions. "
            "Standard monitoring protocol applies."
        ),
        "Malignant": (
            f"The ResNet-50 CNN analysis indicates potential malignant characteristics with {confidence}% confidence. "
            "Morphological patterns may be consistent with irregular border definition and "
            "atypical pigmentation distribution. Immediate clinical evaluation is critical."
        ),
        "Suspicious": (
            f"The ResNet-50 CNN analysis indicates suspicious lesion characteristics with {confidence}% confidence. "
            "Features warrant careful clinical examination. "
            "Differential diagnosis should include dysplastic nevus and early-stage melanoma."
        )
    }

    flag_note = ""
    if confidence < 75:
        flag_note = f"\n⚠️  CONFIDENCE FLAG: Score {confidence}% is below the 75% threshold.\n   Mandatory human review required before any clinical decision.\n"

    report = f"""DERMAGUARD UIR — CLINICAL TRIAGE REPORT
Generated : {date} at {time}
Image     : {filename}
System    : ResNet-50 CNN · HAM10000 · 10,000+ images · PyTorch
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PATIENT SUMMARY
   Dermatological lesion submitted for AI-assisted triage analysis
   via the DermaGuard UIR Clinical System. Image processed through
   a fine-tuned ResNet-50 CNN trained on the HAM10000 ISIC archive.

2. CLASSIFICATION RESULT
   Predicted Class  : {pred_class.upper()}
   Confidence Score : {confidence}%
   Severity Level   : {severity_map[pred_class]}
{flag_note}
   Probability Distribution:
   • Benign     : {probs['Benign']}%
   • Malignant  : {probs['Malignant']}%
   • Suspicious : {probs['Suspicious']}%

3. SEVERITY ASSESSMENT
   {assessment_map[pred_class]}

4. RECOMMENDED ACTION
   {action_map[pred_class]}

5. DISCLAIMER
   This report is generated by DermaGuard UIR (AI-assisted triage system)
   and must not replace clinical judgment or be used as a standalone
   diagnostic tool. Final diagnosis must be established by a qualified
   dermatologist through comprehensive clinical evaluation including
   patient history, physical examination, and additional investigations.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DermaGuard UIR · PyTorch · CrewAI · ResNet-50 · HAM10000
UIR AI & Big Data Program · S8 2025–2026"""

    return report

# ── ROUTES ────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the web interface."""
    return send_from_directory('.', 'dermaguard_uir.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    POST /api/analyze
    Upload image → CNN inference → return classification + report
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Use JPEG, PNG, BMP, TIFF."}), 400

    # Save uploaded file
    filename  = secure_filename(file.filename)
    unique_fn = f"{uuid.uuid4().hex[:8]}_{filename}"
    filepath  = os.path.join(UPLOAD_FOLDER, unique_fn)
    file.save(filepath)

    log_action("image_uploaded", {"filename": filename, "saved_as": unique_fn})

    try:
        # Run real CNN inference
        result = classify_image(filepath)

        # Generate clinical report
        report = generate_report(
            result["predicted_class"],
            result["confidence"],
            result["probabilities"],
            filename
        )

        result["report"]   = report
        result["filename"] = filename
        result["timestamp"]= datetime.utcnow().isoformat()

        log_action("classification_complete", {
            "filename":        filename,
            "predicted_class": result["predicted_class"],
            "confidence":      result["confidence"],
            "flag_for_review": result["flag_for_review"],
        })

        return jsonify(result)

    except Exception as e:
        log_action("classification_error", {"filename": filename, "error": str(e)})
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

@app.route('/api/approve', methods=['POST'])
def approve():
    """
    POST /api/approve
    Save approved report to outputs/
    """
    data     = request.get_json()
    report   = data.get("report", "")
    filename = data.get("filename", "unknown")

    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_name = f"report_{filename.replace('.', '_')}_{ts}.txt"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    log_action("report_approved", {
        "filename":    filename,
        "saved_to":    out_path,
        "timestamp":   datetime.utcnow().isoformat(),
    })

    return jsonify({"status": "approved", "saved_to": out_name})

@app.route('/api/reject', methods=['POST'])
def reject():
    """POST /api/reject — Log rejection."""
    data = request.get_json()
    log_action("report_rejected", {
        "filename":  data.get("filename", "unknown"),
        "timestamp": datetime.utcnow().isoformat(),
    })
    return jsonify({"status": "rejected"})

@app.route('/api/status', methods=['GET'])
def status():
    """GET /api/status — Check if model is loaded."""
    return jsonify({
        "model_loaded": model is not None,
        "device":       str(device),
        "classes":      CLASS_NAMES,
        "threshold":    CONFIDENCE_THRESHOLD,
        "model_path":   MODEL_PATH,
    })

# ── RUN ───────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  DermaGuard UIR — Clinical Web Interface")
    print("="*55)
    print(f"  Model  : {'✓ Loaded' if model else '✗ Not found — train first'}")
    print(f"  Device : {device}")
    print(f"  URL    : http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
