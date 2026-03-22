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
from agent_10_vision import VisionAgent
import os
import shutil
import logging
from fastapi import UploadFile, File, Form

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.agent_10 = VisionAgent()
        
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

class ExplainRequest(BaseModel):
    selected_text: str
    chat_history: list = []
    language: str = "en"

@app.post("/explain")
async def explain_text(req: ExplainRequest):
    """Takes a selected medical term and returns a simple layman explanation in the user's language."""
    import json, os
    from groq import Groq
    
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    lang_map = {
        "en": "English — use simple, everyday English words. Avoid all medical jargon.",
        "hi": "Hindi — write in simple Hindi. Use Devanagari script.",
        "mr": "Marathi — write in simple Marathi. Use Devanagari script.",
        "hinglish": "Hinglish — mix simple Hindi words with English like a normal Indian conversation."
    }
    lang_instruction = lang_map.get(req.language, lang_map["en"])
    
    history_summary = "\n".join([
        f"{'Patient' if m.get('role') == 'user' else 'Doctor'}: {m.get('content', '')}"
        for m in req.chat_history[-10:]  # Last 10 turns for context
    ])
    
    system_prompt = f"""You are a friendly health educator explaining medical terms to a common person.
    
LANGUAGE: {lang_instruction}

RULES:
1. Explain the selected text in 3-5 SIMPLE sentences.
2. Use analogies, everyday examples (like comparing a virus to a thief, a headache to pressure in a balloon).
3. NEVER use complicated medical terms without explaining them.
4. End with ONE simple practical tip the person can follow.
5. Be warm, reassuring, not scary.
6. Use the conversation context to personalize the explanation (e.g., mention their specific symptoms if relevant).
"""
    
    user_prompt = f"""The patient selected this text to understand better:
"{req.selected_text}"

Conversation context:
{history_summary}

Explain this in a very simple, warm, friendly way:"""
    
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=400
        )
        explanation = completion.choices[0].message.content.strip()
        return {"explanation": explanation}
    except Exception as e:
        logger.error(f"Explain endpoint error: {e}")
        return {"explanation": "Sorry, I couldn't generate an explanation right now. Please try again."}

@app.post("/chat")
async def process_chat(
    user_data_json: str = Form(...), 
    file: UploadFile = File(None)
):
    import json
    try:
        request_data = json.loads(user_data_json)
        user_input = request_data.get("user_input", "")
        user_metadata_dict = request_data.get("user_data", {})
        chat_history = request_data.get("chat_history", [])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in user_data_json: {e}")

    # Step 0: Handle Image if present
    vision_findings = ""
    if file:
        temp_dir = "temp_vision"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            v_res = controller.agent_10.analyze(file_path)
            logger.info(f"Vision Analysis Result: {v_res}")
            
            # Make findings very explicit for the extraction agent
            findings_text = v_res.get('visual_findings', 'No specific findings')
            detected_symptoms = v_res.get('possible_symptoms_detected', [])
            
            vision_findings = f"[VISUAL EVIDENCE: The patient's submitted image shows {findings_text}. Potential symptoms identified visually: {', '.join(detected_symptoms)}.]"
            # Augment user input with vision findings
            user_input = f"{vision_findings}\n{user_input}"
            logger.info(f"Augmented User Input: {user_input}")
        except Exception as ve:
            logger.error(f"Vision error in chat flow: {ve}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    # Step 1: Check for Closing Statements
    closing_keywords = ["thank you", "thankyou", "thanks", "thx", "okay bye", "ok bye", "bye", "goodbye", "good bye", "shukriya", "dhanyavad", "shukriya", "badhli"]
    if any(k in user_input.lower() for k in closing_keywords) and len(chat_history) > 2:
        return {
            "type": "info",
            "message": "You're very welcome. I'm glad I could help. Please don't hesitate to reach out if your symptoms change or if you have more questions. Take care!",
            "vision_context": vision_findings
        }

    # Step 2: Dynamic Onboarding Check
    if not user_metadata_dict.get("age"):
        return {"type": "clarification", "message": "Welcome to Aarogya Doot. To provide a high-precision analysis, I need a few details first. **What is your age?**"}
    if not user_metadata_dict.get("gender"):
        return {"type": "clarification", "message": "Got it. **What is your gender?**"}
    if not user_metadata_dict.get("location"):
        return {"type": "clarification", "message": "Thank you. **Where are you currently located?**"}
    if not user_metadata_dict.get("medical_history"):
        return {"type": "clarification", "message": "Nearly there. **Do you have any existing medical conditions or history I should know about?** (Type 'None' if not)"}

    # Step 3: Dynamic Onboarding & Transition Logic
    # 3.1: Check what we HAVE
    has_age = bool(user_metadata_dict.get("age"))
    has_gender = bool(user_metadata_dict.get("gender"))
    has_location = bool(user_metadata_dict.get("location"))
    has_history = bool(user_metadata_dict.get("medical_history"))

    # 3.2: If we are MISSING anything, we are still in onboarding
    if not (has_age and has_gender and has_location and has_history):
        return {"type": "clarification", "message": "Proceeding with onboarding..."}

    # 3.3: If we JUST finished onboarding (History is present), but haven't asked for symptoms yet
    # check history for the "describe symptoms" prompt
    has_asked_for_symptoms = any(m.get('role') == 'assistant' and "describe the symptoms" in m.get('content', '') for m in chat_history)
    
    if not has_asked_for_symptoms:
        # We just completed the profile with the LATEST message (which was history)
        # So don't extract symptoms yet, just ask for them.
        return {
            "type": "clarification", 
            "message": "Profile updated. Now, please **describe the symptoms** you are currently experiencing (you can also attach an image)."
        }

    # Step 4: Extract Symptoms (ONLY once profile is done and we've asked for symptoms)
    age_str = user_metadata_dict.get("age", "Unknown")
    gender_str = user_metadata_dict.get("gender", "Unknown")
    loc = user_metadata_dict.get("location", "Unknown")
    hist = user_metadata_dict.get("medical_history", "None")
    context_prefix = f"[User Profile: {age_str} {gender_str}, Location: {loc}, History: {hist}]\n"

    # Build the COMPLETE history including current user message for agents to use
    # This is the key fix: agents must see the CURRENT answer, not just previous ones
    full_history = list(chat_history) + [{"role": "user", "content": user_input}]

    extracted_symptoms = controller.agent_1.extract_symptoms(user_input, chat_history=full_history)
    
    # If no symptoms extracted but we were in the middle of a diagnostic flow,
    # re-extract from the whole conversation
    if not extracted_symptoms:
        was_asking_clarification = len(chat_history) > 0 and chat_history[-1].get('role') == 'assistant'
        if was_asking_clarification and len(chat_history) > 2:
            logger.info("No new symptoms extracted for follow-up answer. Re-extracting from full history.")
            combined_input = "\n".join([m['content'] for m in full_history if m['role'] == 'user'])
            extracted_symptoms = controller.agent_1.extract_symptoms(combined_input)

    if not extracted_symptoms:
        response_data = {
            "type": "clarification",
            "message": "I couldn't clearly identify any medical symptoms from that. Could you describe them differently?",
            "symptoms_identified": [],
            "vision_context": vision_findings
        }
        return controller.translator.translate_response(user_input, response_data)

    # Step 4: Risk Evaluation (Triage)
    is_emergency, risk_score = controller.agent_3.evaluate_risk(extracted_symptoms)
    if is_emergency:
        db_predictions = controller.agent_2.retrieve_disease(extracted_symptoms, top_k=3)
        llm_predictions = controller.agent_5.predict(extracted_symptoms, patient_info=context_prefix, chat_history=full_history)
        consensus = controller.agent_6.synthesize(extracted_symptoms, db_predictions, llm_predictions, patient_info=context_prefix, chat_history=full_history)
        hypotheses = consensus["hypotheses"]
        
        response_data = {
            "type": "alert",
            "message": f"EMERGENCY ALERT! Your symptoms calculate to a critical risk score ({risk_score}). Halting normal flow. Please seek immediate medical consultation or visit the ER.",
            "diseases": [h["disease"] for h in hypotheses],
            "hypotheses": hypotheses,
            "reasoning": consensus["overall_reasoning"],
            "risk_score": risk_score,
            "symptoms_identified": extracted_symptoms,
            "vision_context": vision_findings
        }
        return controller.translator.translate_response(user_input, response_data)

    # Step 5: Multi-Agent Diagnostic Consensus
    db_predictions = controller.agent_2.retrieve_disease(extracted_symptoms, top_k=3)
    llm_predictions = controller.agent_5.predict(extracted_symptoms, patient_info=context_prefix, chat_history=full_history)
    consensus = controller.agent_6.synthesize(extracted_symptoms, db_predictions, llm_predictions, patient_info=context_prefix, chat_history=full_history)
    
    hypotheses = consensus["hypotheses"]
    status = consensus["status"]
    missing_info = consensus["missing_info"]
    top_rankings = [h["disease"] for h in hypotheses]

    followups = controller.agent_8.generate_questions(extracted_symptoms, top_rankings, patient_info=context_prefix, missing_info=missing_info, chat_history=full_history)

    if status == "IN_PROGRESS" and len(chat_history) < 14:
        response_data = {
            "type": "clarification",
            "message": consensus["overall_reasoning"],
            "followup_questions": followups[:1],
            "hypotheses": hypotheses,
            "symptoms_identified": extracted_symptoms,
            "vision_context": vision_findings
        }
        return controller.translator.translate_response(user_input, response_data)

    if hypotheses:
        primary_h = hypotheses[0]
        primary_disease = primary_h["disease"]
        desc = controller._get_fuzzy_match(primary_disease, controller.description_dict) or "Detailed description not currently available."
        prec = controller._get_fuzzy_match(primary_disease, controller.precaution_dict) or []
        
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
            "symptoms_identified": extracted_symptoms,
            "vision_context": vision_findings
        }
        return controller.translator.translate_response(user_input, response_data)
    else:
        response_data = {
            "type": "error",
            "message": "Clinical consensus could not be reached with the current information.",
            "symptoms_identified": extracted_symptoms,
            "vision_context": vision_findings
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

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...), is_document: bool = False):
    temp_dir = "temp_vision"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        result = controller.agent_10.analyze(file_path, is_document=is_document)
        return result
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
