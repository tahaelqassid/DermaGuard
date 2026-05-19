import os
from openai import OpenAI
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ReportInput(BaseModel):
    predicted_class: str   = Field(..., description="CNN output: Benign, Malignant, or Suspicious.")
    confidence:      float = Field(..., description="Confidence score between 0 and 1.")
    probabilities:   dict  = Field(..., description="Per-class probability dict.")
    image_path:      str   = Field(..., description="Path to the original image.")

class ReportGeneratorTool(BaseTool):
    name: str = "clinical_report_generator"
    description: str = (
        "Generates a structured clinical triage report from CNN classification results."
    )
    args_schema: Type[BaseModel] = ReportInput

    def _run(self, predicted_class: str, confidence: float,
             probabilities: dict, image_path: str) -> str:

        prob_lines = "\n".join(
            f"  - {cls}: {prob*100:.1f}%" for cls, prob in probabilities.items()
        )
        prompt = f"""You are a clinical AI assistant generating a dermatology triage report.

CNN Results:
- Predicted class: {predicted_class}
- Confidence: {confidence*100:.1f}%
- Probabilities:
{prob_lines}
- Image: {os.path.basename(image_path)}

Generate a professional clinical triage report with these sections:
1. PATIENT SUMMARY
2. CLASSIFICATION RESULT
3. SEVERITY ASSESSMENT
4. RECOMMENDED ACTION
5. DISCLAIMER

Be concise and clinically appropriate."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()