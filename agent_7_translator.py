import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class TranslationAgent:
    def __init__(self, model="openai/gpt-oss-120b"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

    def translate_response(self, user_input: str, response_data: dict) -> dict:
        """
        Translates the user-facing text fields in the response_data 
        to match the language of the user_input.
        """
        # We only need to translate string fields that the user will read.
        # type, symptoms_identified, diseases, risk_score can remain as-is for the frontend logic.
        fields_to_translate = {}
        if "message" in response_data:
            fields_to_translate["message"] = response_data["message"]
        if "reasoning" in response_data:
            fields_to_translate["reasoning"] = response_data["reasoning"]
        if "overall_reasoning" in response_data:
            fields_to_translate["overall_reasoning"] = response_data["overall_reasoning"]
        if "description" in response_data:
            fields_to_translate["description"] = response_data["description"]
        if "precautions" in response_data:
            fields_to_translate["precautions"] = response_data["precautions"]
        if "followup_questions" in response_data:
            fields_to_translate["followup_questions"] = response_data["followup_questions"]
        if "hypotheses" in response_data:
            fields_to_translate["hypotheses"] = response_data["hypotheses"]
        if "urgency" in response_data:
            fields_to_translate["urgency"] = response_data["urgency"]

        if not fields_to_translate:
            return response_data

        system_prompt = """
        You are an expert Medical Translator Agent.
        Your task is to translate medical responses into either **English**, **Hindi**, or **Marathi** based on the 'User Input'.
        
        CRITICAL CONSTRAINTS:
        1. **STRICTLY PROHIBITED**: Never use Urdu script (Perso-Arabic) or any other language like Korean.
        2. If the user speaks Hindi/Marathi but uses Urdu script by mistake, you MUST translate the response into Devanagari Hindi or Marathi.
        3. Only support: [English, Hindi, Marathi].
        4. If the User Input is in Devanagari, the output MUST be in Devanagari. If the User Input is in Latin/Roman script, the output MUST be in Latin/Roman script.
        5. Polite greetings in the target language are allowed before the medical content.
        
        Output ONLY a valid JSON object.
        """
        
        user_prompt = f"User Input: {user_input}\n\nJSON to Translate:\n{json.dumps(fields_to_translate, indent=2)}"
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            content = completion.choices[0].message.content.strip()
            
            # Clean up markdown if present
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            translated_fields = json.loads(content)
            
            # Merge translated fields back into a copy of the original response
            final_response = dict(response_data)
            for k, v in translated_fields.items():
                final_response[k] = v
                
            return final_response
            
        except Exception as e:
            print(f"Translation Error: {e}")
            return response_data
