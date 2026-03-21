import csv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent_1_extraction import ClinicalIntakeAgent
from agent_2_rag import DiagnosticRAGAgent
from agent_3_triage import TriageAgent
from agent_4_updater import MedicalUpdaterAgent
from agent_5_pure_llm import PureLLMAgent
from agent_6_combiner import ConsensusSynthesizerAgent
from agent_7_translator import TranslationAgent
from agent_8_followup import FollowupAgent
from agent_9_whisper import WhisperAgent
import os
import shutil
from fastapi import UploadFile, File

app = FastAPI(title="Medical Agent Workflow API")

# Allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserData(BaseModel):
    age: str
    gender: str
    location: str
    medical_history: str

class ChatRequest(BaseModel):
    user_input: str
    user_data: UserData | None = None
    chat_history: list[dict] | None = []

class UpdateRequest(BaseModel):
    disease_name: str

class APIController:
    def __init__(self):
        print("Starting API Controller Agents...")
        self.agent_1 = ClinicalIntakeAgent()
        self.agent_2 = DiagnosticRAGAgent()
        self.agent_3 = TriageAgent()
        self.agent_4 = MedicalUpdaterAgent()
        self.agent_5 = PureLLMAgent()
        self.agent_6 = ConsensusSynthesizerAgent()
        self.translator = TranslationAgent()
        self.agent_8 = FollowupAgent()
        self.agent_9 = WhisperAgent()
        
        self.description_dict = {}
        self.precaution_dict = {}
        self._load_master_data()

    def _load_master_data(self):
        self.description_dict.clear()
        self.precaution_dict.clear()
        try:
            with open('MasterData/symptom_Description.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        self.description_dict[row[0].strip()] = row[1].strip()
                        
            with open('MasterData/symptom_precaution.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 5:
                        self.precaution_dict[row[0].strip()] = [row[1], row[2], row[3], row[4]]
        except Exception as e:
            print(f"Warn: unable to fully load master data: {e}")

    def _get_fuzzy_match(self, disease_name: str, data_dict: dict):
        """Finds a key in the dictionary that closely matches the disease_name."""
        # Exact match first
        if disease_name in data_dict:
            return data_dict[disease_name]
        
        # Simple normalization: lowercase and strip
        norm_name = disease_name.lower().strip().replace("-", " ")
        for key in data_dict.keys():
            if key.lower().strip().replace("-", " ") == norm_name:
                return data_dict[key]
        
        # Substring match (e.g. "Tension Headache" in "Tension-type headache")
        for key in data_dict.keys():
            if norm_name in key.lower() or key.lower() in norm_name:
                return data_dict[key]
                
        return None

controller = APIController()

@app.post("/chat")
async def process_chat(request: ChatRequest):
    user_input = request.user_input
    user_metadata = request.user_data
    chat_history = request.chat_history or []

    # Step 0: Check for Closing Statements
    closing_keywords = ["thank you", "thanks", "okay bye", "goodbye", "shukriya", "dhanyavad"]
    if any(k in user_input.lower() for k in closing_keywords) and len(chat_history) > 2:
        return {
            "type": "info",
            "message": "You're very welcome. I'm glad I could help. Please don't hesitate to reach out if your symptoms change or if you have more questions. Take care!",
            "symptoms_identified": []
        }

    # Step 1: Extract Symptoms from current input and history
    context_prefix = ""
    if user_metadata:
        age_str = user_metadata.age if user_metadata.age.lower() not in ["i don't know", "unknown", "dont know"] else "Unknown age"
        gender_str = user_metadata.gender if user_metadata.gender.lower() not in ["i don't know", "unknown", "dont know"] else "Unknown gender"
        context_prefix = f"[User Profile: {age_str} {gender_str}, Location: {user_metadata.location}, History: {user_metadata.medical_history}]\n"
    
    # We pass the history so Agent 1 can see previous symptoms/confirmations
    extracted_symptoms = controller.agent_1.extract_symptoms(user_input, chat_history=chat_history)
    
    if not extracted_symptoms:
        response_data = {
            "type": "clarification",
            "message": "I couldn't clearly identify any medical symptoms from that. Could you describe them differently?",
            "symptoms_identified": []
        }
        return controller.translator.translate_response(user_input, response_data)

    # Step 2: Risk Evaluation (Triage)
    is_emergency, risk_score = controller.agent_3.evaluate_risk(extracted_symptoms)
    if is_emergency:
        db_predictions = controller.agent_2.retrieve_disease(extracted_symptoms, top_k=3)
        llm_predictions = controller.agent_5.predict(extracted_symptoms, patient_info=context_prefix)
        consensus = controller.agent_6.synthesize(extracted_symptoms, db_predictions, llm_predictions, patient_info=context_prefix, chat_history=chat_history)
        hypotheses = consensus["hypotheses"]
        
        response_data = {
            "type": "alert",
            "message": f"EMERGENCY ALERT! Your symptoms calculate to a critical risk score ({risk_score}). Halting normal flow. Please seek immediate medical consultation or visit the ER.",
            "diseases": [h["disease"] for h in hypotheses],
            "hypotheses": hypotheses,
            "reasoning": consensus["overall_reasoning"],
            "risk_score": risk_score,
            "symptoms_identified": extracted_symptoms
        }
        return controller.translator.translate_response(user_input, response_data)

    # Step 3: Multi-Agent Diagnostic Consensus (Iterative)
    db_predictions = controller.agent_2.retrieve_disease(extracted_symptoms, top_k=3)
    llm_predictions = controller.agent_5.predict(extracted_symptoms, patient_info=context_prefix)
    consensus = controller.agent_6.synthesize(extracted_symptoms, db_predictions, llm_predictions, patient_info=context_prefix, chat_history=chat_history)
    
    hypotheses = consensus["hypotheses"]
    status = consensus["status"]
    missing_info = consensus["missing_info"]
    top_rankings = [h["disease"] for h in hypotheses]

    # Generate EXACTLY ONE follow-up question
    followups = controller.agent_8.generate_questions(extracted_symptoms, top_rankings, patient_info=context_prefix, missing_info=missing_info, chat_history=chat_history)

    # If the synthesizer says we need more info (IN_PROGRESS)
    if status == "IN_PROGRESS" and len(chat_history) < 14: # Allow ~7 full exchanges
        response_data = {
            "type": "clarification",
            "message": consensus["overall_reasoning"],
            "followup_questions": followups[:1], # Strictly enforce ONE question
            "hypotheses": hypotheses,
            "symptoms_identified": extracted_symptoms
        }
        return controller.translator.translate_response(user_input, response_data)

    # Final Diagnosis reached (or limit reached)
    if hypotheses:
        primary_h = hypotheses[0]
        primary_disease = primary_h["disease"]
        desc = controller._get_fuzzy_match(primary_disease, controller.description_dict) or "Detailed description not currently available."
        prec = controller._get_fuzzy_match(primary_disease, controller.precaution_dict) or []
        
        # Calculate urgency based on risk score + medical context
        urgency = "Normal"
        if risk_score > 7: urgency = "Immediate (Emergency)"
        elif risk_score > 4: urgency = "Urgent (24-48 hours)"

        response_data = {
            "type": "diagnosis",
            "message": f"Based on our clinical interview, the most likely diagnosis is **{primary_disease}** (Confidence: {primary_h['confidence']}%).\n\n{desc}",
            "diseases": [h["disease"] for h in hypotheses],
            "hypotheses": hypotheses,
            "reasoning": consensus["overall_reasoning"],
            "description": desc,
            "precautions": [p for p in prec if p.strip()],
            "urgency": urgency,
            "symptoms_identified": extracted_symptoms
        }
        return controller.translator.translate_response(user_input, response_data)
    else:
        response_data = {
            "type": "error",
            "message": "Clinical consensus could not be reached with the current information.",
            "symptoms_identified": extracted_symptoms
        }
        return controller.translator.translate_response(user_input, response_data)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), language: str = None):
    # Save temp file
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        text = controller.agent_9.transcribe(file_path, language=language)
        return {"text": text}
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/update_disease")
async def update_disease(request: UpdateRequest):
    success = controller.agent_4.update_knowledge(request.disease_name)
    controller._load_master_data()
    
    if success:
        return {"status": "success", "message": f"{request.disease_name} has been successfully registered in the database."}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to fetch details for {request.disease_name}.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
