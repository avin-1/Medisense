import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class ConsensusSynthesizerAgent:
    def __init__(self, model="openai/gpt-oss-120b"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

    def synthesize(self, symptoms: list[str], db_predictions: list[str], llm_predictions: list[str]) -> dict:
        symptoms_str = ", ".join(symptoms)
        db_str = ", ".join(db_predictions) if db_predictions else "None"
        llm_str = ", ".join(llm_predictions) if llm_predictions else "None"
        
        system_prompt = """
        You are the Chief Medical Synthesizer.
        You are given:
        1. A patient's extracted symptoms.
        2. A list of predicted diseases from a local Vector database (RAG).
        3. A list of predicted diseases from a Pure LLM Diagnostician.
        
        Your job is to safely cross-verify these two lists, analyze the symptoms, resolve any discrepancies, and synthesize a final ordered list of the top 3 most probable conditions.
        You must also provide a short, professional paragraph explaining your medical reasoning for this synthesis.
        
        Output ONLY valid JSON structure:
        {
            "final_rankings": ["Disease1", "Disease2", "Disease3"],
            "reasoning": "A concise paragraph explaining why these diseases were chosen based on bridging the DB and LLM predictions with the symptoms."
        }
        Do not output markdown code blocks.
        """
        
        user_prompt = f"""
        Symptoms: {symptoms_str}
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
                "final_rankings": data.get("final_rankings", []),
                "reasoning": data.get("reasoning", "Synthesizer logic failed to extract reasoning.")
            }
            
        except Exception as e:
            print(f"Error in Agent 6 (Synthesizer): {e}")
            # Fallback
            return {
                "final_rankings": db_predictions,
                "reasoning": "Fallback to database due to synthesizer error."
            }
