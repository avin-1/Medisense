import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class FollowupAgent:
    def __init__(self, model="openai/gpt-oss-120b"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

    def generate_questions(self, extracted_symptoms: list[str], potential_diseases: list[str], patient_info: str = "", missing_info: list[str] = [], chat_history: list[dict] = []) -> list[str]:
        """
        Generates exactly ONE targeted follow-up question, never repeating what has been asked.
        """
        if not extracted_symptoms:
            return ["Could you please describe your symptoms in more detail? For example, where is the pain, or how long have you felt this way?"]

        # Build a clear history for the prompt
        history_lines = []
        for m in chat_history:
            role = "Doctor" if m['role'] == 'assistant' else "Patient"
            history_lines.append(f"{role}: {m['content']}")
        history_summary = "\n".join(history_lines)

        system_prompt = """You are an expert Clinical Interviewer AI. Your goal is to ask EXACTLY ONE highly targeted, professional follow-up question to narrow the diagnosis.

STRICT MEMORY PROTOCOL — YOU MUST FOLLOW THIS:
1. Read the ENTIRE Conversation History below.
2. List every topic the Doctor has ALREADY asked about.
3. List every key detail the Patient has ALREADY mentioned (e.g., onset, duration, severity, location).
4. Your new question MUST be on a COMPLETELY DIFFERENT TOPIC from anything already discussed.
5. If onset (sudden vs gradual) has been addressed → ask about ASSOCIATED SYMPTOMS (nausea, vision changes, neck stiffness, fever).
6. If associated symptoms have been addressed → ask about TRIGGERS (stress, food, sleep, posture).
7. If triggers have been addressed → ask about FAMILY HISTORY or past episodes.

NEVER ask: "Did it come on suddenly?" or any variant if ANY answer about onset timing exists in the history.

Output ONLY a valid JSON array with EXACTLY ONE string question.
Example: ["Have you noticed any nausea, sensitivity to light, or blurred vision along with the headache?"]
"""

        user_prompt = f"""Conversation History:
{history_summary}

Patient Profile: {patient_info}
Symptoms Identified: {', '.join(extracted_symptoms)}
Top Hypotheses: {potential_diseases}
Missing Info Per CMO: {', '.join(missing_info)}

Now generate ONE new follow-up question that has NOT been asked before:"""

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
            
            # Clean up markdown
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            questions = json.loads(content)
            return questions if isinstance(questions, list) else []
            
        except Exception as e:
            print(f"Follow-up Generation Error: {e}")
            return ["Have you noticed any nausea, sensitivity to light, or neck stiffness along with your main symptom?"]
