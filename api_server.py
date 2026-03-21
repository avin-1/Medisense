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

app = FastAPI(title="Medical Agent Workflow API")

# Allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str

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

controller = APIController()

@app.post("/chat")
async def process_chat(request: ChatRequest):
    user_input = request.user_input

    # Step 1: Extract Symptoms
    extracted_symptoms = controller.agent_1.extract_symptoms(user_input)
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
        llm_predictions = controller.agent_5.predict(extracted_symptoms)
        consensus = controller.agent_6.synthesize(extracted_symptoms, db_predictions, llm_predictions)
        
        response_data = {
            "type": "alert",
            "message": f"EMERGENCY ALERT! Your symptoms calculate to a critical risk score ({risk_score}). Halting normal flow. Please seek immediate medical consultation or visit the ER.",
            "diseases": consensus["final_rankings"],
            "reasoning": consensus["reasoning"],
            "risk_score": risk_score,
            "symptoms_identified": extracted_symptoms
        }
        return controller.translator.translate_response(user_input, response_data)

    # Step 3: Multi-Agent Diagnostic Consensus
    db_predictions = controller.agent_2.retrieve_disease(extracted_symptoms, top_k=3)
    llm_predictions = controller.agent_5.predict(extracted_symptoms)
    consensus = controller.agent_6.synthesize(extracted_symptoms, db_predictions, llm_predictions)
    
    final_rankings = consensus["final_rankings"]
    if final_rankings:
        primary_disease = final_rankings[0]
        desc = controller.description_dict.get(primary_disease, "Description not currently available.")
        prec = controller.precaution_dict.get(primary_disease, [])
        response_data = {
            "type": "diagnosis",
            "message": f"Based on the analysis, you may have **{primary_disease}**.\n\n{desc}",
            "diseases": final_rankings,
            "reasoning": consensus["reasoning"],
            "description": desc,
            "precautions": [p for p in prec if p.strip()],
            "risk_score": risk_score,
            "symptoms_identified": extracted_symptoms
        }
        return controller.translator.translate_response(user_input, response_data)
    else:
        response_data = {
            "type": "error",
            "message": "Could not match your symptoms to a specific disease.",
            "symptoms_identified": extracted_symptoms
        }
        return controller.translator.translate_response(user_input, response_data)

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
