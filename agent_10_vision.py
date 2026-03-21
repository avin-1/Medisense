import os
import json
import logging
from typing import Dict, Any
from PIL import Image
import pytesseract
import warnings
from dotenv import load_dotenv
import base64
from groq import Groq
import requests

load_dotenv()

# Suppress the deprecation warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import google.generativeai as genai

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

VISION_SYSTEM_PROMPT = """
You are a medical visual observation assistant for the Medisense AI chatbot.
Your job is ONLY to observe and describe the visual contents of the provided image.

CRITICAL RULES:
1. NEVER produce a diagnosis. Only describe visual observations.
2. If the image is unclear, blurry, or not a valid medical/health image, set confidence to "low" and explain carefully in the limitations field.
3. You must output a valid JSON object matching the exact schema provided.

Provide your response as a JSON object with EXACTLY these keys:
{
  "image_type": "string (e.g., 'lab_report', 'prescription', 'symptom_photo', 'x_ray', 'unclear')",
  "visual_findings": "string (detailed, objective description of what is visible)",
  "possible_symptoms_detected": ["list", "of", "strings (e.g., 'redness', 'swelling', 'elevated glucose')"],
  "confidence": "string (high, medium, low)",
  "urgency_flag": boolean (True if visual signs indicate immediate severe emergency),
  "notes": "string (any additional context)",
  "limitations": "string (reasons for low confidence, or what cannot be determined visually)"
}
"""

class VisionAgent:
    def __init__(self):
        pass

    def _run_tesseract_ocr(self, image: Image.Image) -> str:
        try:
            text = pytesseract.image_to_string(image).strip()
            return text
        except Exception as e:
            logger.warning(f"Failed to run Tesseract: {e}")
            return ""

    def _build_error_response(self, reason: str) -> Dict[str, Any]:
        return {
            "image_type": "unclear",
            "visual_findings": "None",
            "possible_symptoms_detected": [],
            "confidence": "low",
            "urgency_flag": False,
            "notes": "Error during vision processing.",
            "limitations": reason
        }

    def _run_gemini(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        # Using the verified working model
        model = genai.GenerativeModel('gemini-2.0-flash')
        try:
            response = model.generate_content(
                [prompt, image], 
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            raise e

    def _run_groq(self, image_path: str, prompt: str) -> Dict[str, Any]:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY missing in environment.")
            
        client = Groq(api_key=api_key)
        
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}",
                                },
                            },
                        ],
                    }
                ],
                model="llama-3.2-11b-vision-preview",
                response_format={"type": "json_object"},
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error during Groq API call: {e}")
            raise e

    def _run_roboflow(self, image_path: str) -> Dict[str, Any]:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY missing.")
        
        model_id = os.environ.get("ROBOFLOW_MODEL_ID", "skin-disease-yup5l-eobng/1")
        url = f"https://detect.roboflow.com/{model_id}?api_key={api_key}"
        
        try:
            with open(image_path, "rb") as image_file:
                response = requests.post(url, files={"file": image_file})
                response.raise_for_status()
                data = response.json()
                
                predictions = data.get("predictions", [])
                primary_prediction = predictions[0] if predictions else None
                
                if primary_prediction:
                    disease_class = primary_prediction.get("class", "unknown")
                    confidence_val = primary_prediction.get("confidence", 0)
                    
                    return {
                        "image_type": "symptom_photo",
                        "visual_findings": f"Classification result from specialized skin model: {disease_class}",
                        "possible_symptoms_detected": [disease_class],
                        "confidence": "high" if confidence_val > 0.8 else "medium" if confidence_val > 0.5 else "low",
                        "urgency_flag": False,
                        "notes": f"Detected {disease_class} with confidence {confidence_val:.2f}",
                        "limitations": "Model specialized for skin conditions."
                    }
                else:
                    return {
                        "image_type": "unclear",
                        "visual_findings": "No specific skin condition detected.",
                        "possible_symptoms_detected": [],
                        "confidence": "low",
                        "urgency_flag": False,
                        "notes": "No predictions found.",
                        "limitations": "Image might not be a skin condition."
                    }
        except Exception as e:
            logger.error(f"Error during Roboflow API call: {e}")
            raise e

    def analyze(self, image_path: str, is_document: bool = False) -> Dict[str, Any]:
        provider = os.environ.get("VISION_PROVIDER", "gemini").lower()
        
        if not image_path or not os.path.exists(image_path):
            return self._build_error_response("Image path is missing.")

        try:
            image = Image.open(image_path)
        except Exception:
            return self._build_error_response("Failed to open image.")

        ocr_text = ""
        if is_document:
            ocr_text = self._run_tesseract_ocr(image)
        
        prompt = VISION_SYSTEM_PROMPT
        if ocr_text:
            prompt += f"\n\n[OCR DATA ENCLOSED]\nTesseract extracted text:\n{ocr_text}"

        try:
            if provider == "roboflow":
                return self._run_roboflow(image_path)
            elif provider == "groq":
                return self._run_groq(image_path, prompt)
            else:
                return self._run_gemini(image, prompt)
        except Exception as e:
            return self._build_error_response(str(e))
