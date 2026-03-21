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
You are an expert Clinical Intake AI Agent. Your job is to extract medical symptoms from the patient's conversational input and the provided history.

CRITICAL INSTRUCTIONS:
1. Contextual Extraction: If the user provides a descriptive answer to a previous question (e.g., Assistant: "Did it start suddenly?", Patient: "No, gradually"), you MUST retain the primary symptom being discussed (e.g., "headache") in your output.
2. Symptom Mapping: Map all descriptions (Hindi, Marathi, or English) strictly to the approved list below.
3. Multi-turn Memory: Use the "Conversation History" to understand what has already been established. Do not lose symptoms that were identified in previous turns if they are still relevant.

Approved list of symptoms:
{", ".join(self.valid_symptoms)}

Output ONLY a JSON array of strings containing the exact matched symptom names. 
If the user provides an answer that clarifies a symptom but doesn't name a NEW one, still include the symptom being clarified.
Example: History shows "headache". User says "it's on the left side". Output: ["headache"]
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
