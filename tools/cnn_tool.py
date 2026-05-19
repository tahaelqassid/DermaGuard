import torch
import torch.nn.functional as F
from PIL import Image
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_PATH, NUM_CLASSES, CLASS_NAMES, CONFIDENCE_THRESHOLD
from model.model_def import load_model
from data.transforms import inference_transforms

class CNNInput(BaseModel):
    image_path: str = Field(..., description="Absolute path to the skin lesion image (JPEG or PNG).")

class CNNClassifierTool(BaseTool):
    name: str = "skin_lesion_classifier"
    description: str = (
        "Classifies a skin lesion image as Benign, Malignant, or Suspicious "
        "using a fine-tuned ResNet-50 CNN. Returns the predicted class, "
        "confidence score, and all class probabilities."
    )
    args_schema: Type[BaseModel] = CNNInput

    _model = None
    _device = None

    def _load(self):
        if self._model is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = load_model(MODEL_PATH, NUM_CLASSES, str(self._device))

    def _run(self, image_path: str) -> dict:
        self._load()
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        tensor = inference_transforms(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze().cpu().tolist()

        pred_idx    = int(torch.tensor(probs).argmax())
        pred_class  = CLASS_NAMES[pred_idx]
        confidence  = round(probs[pred_idx], 4)
        flag_review = confidence < CONFIDENCE_THRESHOLD

        return {
            "predicted_class": pred_class,
            "confidence":      confidence,
            "probabilities":   {CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(probs)},
            "flag_for_review": flag_review,
            "image_path":      image_path,
        }
