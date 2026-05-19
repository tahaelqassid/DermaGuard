import os
from dotenv import load_dotenv

load_dotenv()

# API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Paths
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model", "dermaguard_cnn.pth")
DATA_DIR     = os.path.join(BASE_DIR, "data", "isic")
LOG_DIR      = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model
NUM_CLASSES   = 3
CLASS_NAMES   = ["Benign", "Malignant", "Suspicious"]
IMAGE_SIZE    = 224
BATCH_SIZE    = 32
NUM_EPOCHS    = 15
LEARNING_RATE = 1e-4

# Agent
CONFIDENCE_THRESHOLD = 0.75