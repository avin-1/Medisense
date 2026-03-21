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
        if "description" in response_data:
            fields_to_translate["description"] = response_data["description"]
        if "precautions" in response_data:
            fields_to_translate["precautions"] = response_data["precautions"]

        if not fields_to_translate:
            return response_data

        system_prompt = """
        You are an expert Medical Translator Agent.
        Your task is to detect the language, script, and tone of the 'User Input'.
        Then, translate ALL the values in the provided JSON object to that EXACT SAME language and script.
        If the 'User Input' is a conversational greeting (e.g., "Kaise hai aap", "Hello"), ensure the translated 'message' politely acknowledges the greeting before delivering the medical text.
        
        Output ONLY a valid JSON object with the exact same keys as the input JSON, but with translated string values. Do not output markdown code blocks.
        If the value is a list of strings (like precautions), translate each string in the list.
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
