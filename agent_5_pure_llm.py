import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class PureLLMAgent:
    def __init__(self, model="openai/gpt-oss-120b"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

    def predict(self, symptoms: list[str], patient_info: str = "", chat_history: list[dict] = []) -> list[str]:
        symptoms_str = ", ".join(symptoms)
        
        # Build conversation history for the LLM 
        history_lines = []
        for m in chat_history:
            role = "Doctor" if m['role'] == 'assistant' else "Patient"
            history_lines.append(f"{role}: {m['content']}")
        history_summary = "\n".join(history_lines) if history_lines else "(No conversation history yet)"

        system_prompt = """You are a world-class Pure LLM Diagnostician. You rely entirely on your vast internal pre-trained medical knowledge.

CLINICAL REASONING PROTOCOL:
1. Read the conversation history carefully for ALL clinical clues (onset, severity, duration, associated symptoms).
2. Use these clues to select the 3 most likely diseases.
3. If the patient said "gradual onset" → do NOT include subarachnoid hemorrhage as a top choice.
4. Be specific. Do not output generic catches like "Other causes".

Output ONLY valid JSON — no markdown fences:
{
    "diseases": ["Most Likely Disease", "Second Most Likely", "Third Most Likely"]
}
"""
        
        user_prompt = f"""Patient Profile: {patient_info}
Symptoms Identified: {symptoms_str}

Conversation History:
{history_summary}

Based on all evidence, predict the top 3 most likely diseases:"""
        
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
            
            # Clean up potential markdown formatting
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            data = json.loads(content)
            return data.get("diseases", [])
            
        except Exception as e:
            print(f"Error in Agent 5 (Pure LLM): {e}")
            return []
