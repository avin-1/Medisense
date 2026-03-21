import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class ConsensusSynthesizerAgent:
    def __init__(self, model="openai/gpt-oss-120b"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

    def synthesize(self, symptoms: list[str], db_predictions: list[str], llm_predictions: list[str], patient_info: str = "", chat_history: list[dict] = []) -> dict:
        symptoms_str = ", ".join(symptoms)
        db_str = ", ".join(db_predictions) if db_predictions else "None"
        llm_str = ", ".join(llm_predictions) if llm_predictions else "None"
        
        system_prompt = """
        Your goal is to iteratively reduce diagnostic uncertainty by managing a set of clinical hypotheses.
        
        You are given:
        1. A patient's profile (age, gender, location, history).
        2. A patient's extracted symptoms (from the full conversational history).
        3. Predicted diseases from a Vector DB and a Pure LLM.
        
        Follow this strict protocol:
        - Maintain top 3-5 hypotheses (candidate diseases).
        - Assign a confidence score (%) to each.
        - Decide between "status": "IN_PROGRESS" or "status": "FINAL".
        - Use "IN_PROGRESS" if no disease has >80% confidence and more info can be gained.
        - Specifically identify what information is missing to rule in/out the current hypotheses.
        
        Output ONLY valid JSON structure:
        {
            "status": "FINAL" or "IN_PROGRESS",
            "hypotheses": [
                {"disease": "Name", "confidence": 85, "reasoning": "Why?"},
                {"disease": "Alternative", "confidence": 10, "reasoning": "Why?"}
            ],
            "overall_reasoning": "Concise medical logic for the current state.",
            "missing_info": ["Specific symptom or characteristic needed to differentiate"]
        }
        Do not output markdown code blocks.
        """
        
        history_summary = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
        user_prompt = f"""
        Conversation History:
        {history_summary}
        
        Patient Profile: {patient_info}
        Symptoms Extracted So Far: {symptoms_str}
        DB Predictions: {db_str}
        LLM Predictions: {llm_str}
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            content = completion.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            data = json.loads(content)
            return {
                "status": data.get("status", "FINAL"),
                "hypotheses": data.get("hypotheses", []),
                "overall_reasoning": data.get("overall_reasoning", ""),
                "missing_info": data.get("missing_info", [])
            }
            
        except Exception as e:
            print(f"Error in Agent 6 (Synthesizer): {e}")
            # Fallback
            fallback_hypotheses = [{"disease": d, "confidence": 33 if i < 3 else 0, "reasoning": "DB prediction"} for i, d in enumerate(db_predictions)]
            return {
                "status": "FINAL",
                "hypotheses": fallback_hypotheses,
                "overall_reasoning": "Fallback to database due to synthesizer error.",
                "missing_info": []
            }
