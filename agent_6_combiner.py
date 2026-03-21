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

        # Build a clean, labeled history for the prompt
        history_lines = []
        for m in chat_history:
            role = "Doctor" if m['role'] == 'assistant' else "Patient"
            history_lines.append(f"  {role}: {m['content']}")
        history_summary = "\n".join(history_lines) if history_lines else "  (No prior conversation)"

        system_prompt = """You are the Chief Medical Consultant AI. Your job is to synthesize all available evidence into a ranked differential diagnosis.

EVIDENCE INTEGRATION PROTOCOL:
1. Read the entire Conversation History to extract all clinical clues provided by the patient.
2. Use these clues to ADJUST confidence scores. Examples:
   - Patient says "gradual onset" → LOWER confidence for brain hemorrhage; RAISE for tension headache/migraine.
   - Patient says "sudden thunderclap" → RAISE confidence for subarachnoid hemorrhage significantly.
   - Patient says "no nausea, no light sensitivity" → LOWER migraine confidence slightly.
   - Patient says "stiff neck" → RAISE meningitis confidence immediately.
3. The confidence scores MUST CHANGE based on what the patient has revealed compared to the initial message.
4. Use "IN_PROGRESS" status if confidence is still distributed and more useful info can be gathered.
5. Use "FINAL" if one disease has clearly >70% confidence or 4+ rounds of questioning have happened.

Output ONLY valid JSON — no markdown fences:
{
    "status": "IN_PROGRESS" or "FINAL",
    "hypotheses": [
        {"disease": "Disease Name", "confidence": 75, "reasoning": "Explain how history informs this score"}
    ],
    "overall_reasoning": "A clinical paragraph synthesizing all evidence collected in the conversation.",
    "missing_info": ["What specific detail would most improve confidence?"]
}"""

        user_prompt = f"""
Patient Profile: {patient_info}
Symptoms Extracted: {symptoms_str}

Conversation History:
{history_summary}

Database Agent Predictions: {db_str}
LLM Agent Predictions: {llm_str}

Based on ALL of the above, synthesize the differential diagnosis with updated confidence scores:"""

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
