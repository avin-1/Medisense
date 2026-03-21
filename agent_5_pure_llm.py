import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class PureLLMAgent:
    def __init__(self, model="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

    def predict(self, symptoms: list[str]) -> list[str]:
        symptoms_str = ", ".join(symptoms)
        
        system_prompt = """
        You are a world-class Pure LLM Diagnostician. You rely entirely on your vast internal pre-trained medical knowledge.
        Given a list of patient symptoms, predict the top 3 most likely medical conditions.
        Output ONLY valid JSON in this exact format, with no conversational filler or markdown blocks:
        {
            "diseases": ["Disease Name 1", "Disease Name 2", "Disease Name 3"]
        }
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Symptoms: {symptoms_str}"}
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
