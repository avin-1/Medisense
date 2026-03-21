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
        Generates 2-3 follow-up questions to clarify symptoms or differentiate between diseases.
        """
        if not extracted_symptoms:
            return ["Could you please describe your symptoms in more detail? For example, where is the pain, or how long have you felt this way?"]

        system_prompt = """
        You are an expert Clinical Interviewer AI. 
        Your goal is to ask EXACTLY ONE highly targeted, professional follow-up question to a patient.
        This question must maximize "Information Gain" to help differentiate between the current hypotheses.
        
        CRITICAL: Review the "Conversation History" provided. NEVER ask a question that has already been asked or addressed in the history. Focus on PROGRESSION.
        
        Focus on:
        1. Missing information identified by the Chief Medical Officer.
        2. Ruling out high-risk "Red Flag" conditions.
        3. Differentiating between two similar candidate diseases.
        
        Output ONLY a RAW JSON array containing EXACTLY ONE string (the question).
        Example: ["Does the pain radiate to your left arm or jaw?"]
        """
        
        history_summary = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
        user_prompt = f"Conversation History:\n{history_summary}\n\nPatient Profile: {patient_info}\nExtracted Symptoms: {', '.join(extracted_symptoms)}\nMissing Info Needed: {', '.join(missing_info)}\nTop Hypotheses: {potential_diseases}"
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4
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
            return ["Could you tell me more about any other symptoms you might be experiencing?"]
