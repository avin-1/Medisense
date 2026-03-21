import os
import json
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class ClinicalIntakeAgent:
    def __init__(self, data_path='Data/Training.csv', model="openai/gpt-oss-120b"):
        # We read the training CSV solely to get the exact list of valid symptoms
        df = pd.read_csv(data_path)
        self.valid_symptoms = list(df.columns[:-1])
        
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY")
        )
        # Or if "gpt-oss-120b" is required we pass it to the model arg
        self.model = model

    def extract_symptoms(self, user_input: str, chat_history: list[dict] = None) -> list[str]:
        # Build the conversation context for the LLM
        history_str = ""
        if chat_history:
            for msg in chat_history:
                role = "Patient" if msg['role'] == 'user' else "Assistant"
                history_str += f"{role}: {msg['content']}\n"
        
        system_prompt = f"""
You are an expert Clinical Intake AI Agent. Your job is to extract medical symptoms from the patient's conversational input and the provided history, then map them strictly to the approved list of symptoms.
Context is key: look for symptoms mentioned in the latest message AND those confirmed in the previous history.

Approved list of symptoms:
{", ".join(self.valid_symptoms)}

Output ONLY a JSON array of strings containing the exact matched symptom names. Do not provide any conversational text, markdown, or explanations. 
If no clear symptoms match, return an empty JSON array: []
"""
        
        user_prompt = f"Conversation History:\n{history_str}\nLatest User Input: {user_input}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        try:
            content = response.choices[0].message.content.strip()
            # Clean up markdown if present
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            symptoms = json.loads(content)
            
            # Additional validation loop
            validated_symptoms = [s for s in symptoms if s in self.valid_symptoms]
            return validated_symptoms
        except Exception as e:
            print(f"Error parsing LLM response: {e}. Raw response: {response.choices[0].message.content}")
            return []
