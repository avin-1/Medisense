import csv
import pyttsx3
from agent_1_extraction import ClinicalIntakeAgent
from agent_2_rag import DiagnosticRAGAgent
from agent_3_triage import TriageAgent
from agent_4_updater import MedicalUpdaterAgent

def readn(nstr):
    try:
        engine = pyttsx3.init()
        engine.setProperty('voice', "english+f5")
        engine.setProperty('rate', 130)
        engine.say(nstr)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"Text-to-speech error: {e}")

class MainWorkflow:
    def __init__(self):
        print("Initializing Agents...")
        self.agent_1 = ClinicalIntakeAgent()
        self.agent_2 = DiagnosticRAGAgent()
        self.agent_3 = TriageAgent()
        self.agent_4 = MedicalUpdaterAgent()
        
        # Load descriptions and precautions
        self.description_dict = {}
        self.precaution_dict = {}
        self._load_master_data()

    def _load_master_data(self):
        with open('MasterData/symptom_Description.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    self.description_dict[row[0].strip()] = row[1].strip()
                    
        with open('MasterData/symptom_precaution.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 5:
                    self.precaution_dict[row[0].strip()] = [row[1], row[2], row[3], row[4]]

    def run(self):
        print("-----------------------------------HealthCare Agentic ChatBot-----------------------------------")
        name = input("\nYour Name? \t-> ")
        print(f"Hello, {name}!")

        while True:
            user_input = input(f"\n[{name}] Please describe your symptoms in your own words (or type 'quit' to exit)\n\t-> ")
            
            if user_input.lower() in ['quit', 'exit']:
                print("Take care! Goodbye.")
                break
                
            if user_input.lower().startswith('/update '):
                disease_to_update = user_input[8:].strip()
                if disease_to_update:
                    self.agent_4.update_knowledge(disease_to_update)
                else:
                    print("Please provide a disease name. Example: /update Covid-19")
                continue
                
            print("\n[Agent 1: Extraction] Analyzing your symptoms...")
            extracted_symptoms = self.agent_1.extract_symptoms(user_input)
            
            if not extracted_symptoms:
                print("I couldn't clearly identify the medical symptoms from that. Could you describe them differently?")
                continue
                
            print(f"> Extracted Symptoms: {', '.join(extracted_symptoms)}")
            
            print("\n[Agent 3: Triage] Evaluating real-time risk...")
            is_emergency, risk_score = self.agent_3.evaluate_risk(extracted_symptoms)
            
            if is_emergency:
                alert_msg = f"EMERGENCY ALERT! Your symptoms calculate to a critical risk score ({risk_score}). Halting normal flow. Please seek immediate medical consultation or visit the ER."
                print(f"> \033[91m{alert_msg}\033[0m")
                
                predicted_diseases = self.agent_2.retrieve_disease(extracted_symptoms, top_k=1)
                if predicted_diseases:
                    print(f"> \033[91mImmediate Attention Required for Possible: {predicted_diseases[0]}\033[0m")
                continue 
            else:
                print(f"> Risk Score: {risk_score} (Non-critical). Proceeding to diagnostics...")
                
            print("\n[Agent 2: RAG Diagnostic] Querying Vector Database for disease matching...")
            predicted_diseases = self.agent_2.retrieve_disease(extracted_symptoms, top_k=2)
            
            if predicted_diseases:
                primary_disease = predicted_diseases[0]
                print(f"\nBased on the analysis, you may have: {primary_disease}")
                if primary_disease in self.description_dict:
                    print(f"Description: {self.description_dict[primary_disease]}")
                    
                if primary_disease in self.precaution_dict:
                    print("\nTake the following preventive measures:")
                    for idx, measure in enumerate(self.precaution_dict[primary_disease]):
                        if measure and measure.strip():
                            print(f"{idx+1}) {measure.strip()}")
            else:
                print("Could not match your symptoms to a specific disease in our database.")

if __name__ == "__main__":
    app = MainWorkflow()
    app.run()
