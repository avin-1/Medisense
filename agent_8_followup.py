import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class FollowupAgent:
    def __init__(self, model="openai/gpt-oss-120b"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

    def generate_questions(self, extracted_symptoms: list[str], potential_diseases: list[str], patient_info: str = "") -> list[str]:
        """
        Generates 2-3 follow-up questions to clarify symptoms or differentiate between diseases.
        """
        if not extracted_symptoms:
            return ["Could you please describe your symptoms in more detail? For example, where is the pain, or how long have you felt this way?"]

        system_prompt = """
        You are an expert Clinical Interviewer AI. 
        Your goal is to ask 2-3 highly targeted, professional follow-up questions to a patient.
        These questions should help differentiate between the 'Potential Diseases' based on the 'Extracted Symptoms' and the 'Patient Profile'.
        Focus on:
        1. Duration of symptoms.
        2. Severity or specific characteristics (e.g., type of cough, exact location of pain).
        3. Presence or absence of key differentiating symptoms related to their medical history or location.
        
        Output ONLY a RAW JSON array of 2-3 strings (the questions). Do not include markdown or conversational filler.
        Example: ["How many days have you had this fever?", "Is your cough dry or productive?"]
        """
        
        user_prompt = f"Patient Profile: {patient_info}\nExtracted Symptoms: {', '.join(extracted_symptoms)}\nPotential Diseases: {', '.join(potential_diseases)}"
        
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
