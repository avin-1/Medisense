import os
import csv
import json
import time
import chromadb
from chromadb.utils import embedding_functions
from tavily import TavilyClient
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class BatchDatabaseExpander:
    def __init__(self, db_path="./chroma_db"):
        self.tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.structurer_model = "openai/gpt-oss-120b"
        
        # Setup Chroma
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.db_client.get_or_create_collection(
            name="disease_symptoms",
            embedding_function=self.sentence_transformer_ef
        )
        
        self.existing_diseases = self._load_existing_diseases()

    def _load_existing_diseases(self):
        try:
            existing = set()
            with open('MasterData/symptom_Description.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        existing.add(row[0].strip().lower())
            return existing
        except Exception:
            return set()

    def fetch_all_details_via_web(self, disease_name: str):
        # 1. Search Web
        query = f"What are the symptoms, description, and precautions for medical disease {disease_name}? Provide clinical details."
        print(f"  -> Searching web for: {disease_name}")
        try:
            response = self.tavily_client.search(query, search_depth="advanced", max_results=3)
            context = " ".join([res.get('content', '') for res in response.get('results', [])])
        except Exception as e:
            print(f"  -> Web search failed: {e}")
            return None

        # 2. Extract structured data using LLM
        print(f"  -> Structuring data using LLM...")
        system_prompt = f"""
You are an expert Medical Data Structurer. Based on the provided raw web context about '{disease_name}', extract:
1. "symptoms": a list of symptoms (lowercase, underscore separated like "high_fever").
2. "description": A short, 1-2 sentence description of the disease.
3. "precautions": A list of exactly 4 short prevention/care measures.

Output ONLY valid JSON in this exact format:
{{
    "disease": "{disease_name}",
    "symptoms": ["symptom_1", "symptom_2"],
    "description": "Short description here...",
    "precautions": ["measure 1", "measure 2", "measure 3", "measure 4"]
}}
Do not include markdown blocks (```json) or any conversational text.
"""
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.structurer_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.0
            )
            content = completion.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            data = json.loads(content)
            
            # Pad precautions to ensure exactly 4
            precautions = data.get("precautions", [])
            while len(precautions) < 4:
                precautions.append("Consult a doctor")
            data["precautions"] = precautions[:4]
            
            return data
        except Exception as e:
            print(f"  -> LLM Extraction failed or returned invalid JSON: {e}")
            return None

    def expand_database(self, disease_list):
        print(f"Starting mass expansion for {len(disease_list)} diseases...")
        
        for idx, disease in enumerate(disease_list):
            if disease.lower() in self.existing_diseases:
                print(f"[{idx+1}/{len(disease_list)}] Skipping '{disease}' - already exists in database.")
                continue
                
            print(f"\n[{idx+1}/{len(disease_list)}] Processing '{disease}'...")
            
            try:
                data = self.fetch_all_details_via_web(disease)
                if not data or not data.get("symptoms"):
                    print(f"  -> Failed to acquire sufficient data. Skipping.")
                    continue
                    
                # 1. Update CSVs so main application functions seamlessly
                with open('MasterData/symptom_Description.csv', 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([data["disease"], data["description"]])
                    
                with open('MasterData/symptom_precaution.csv', 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    row = [data["disease"]] + data["precautions"]
                    writer.writerow(row)
                    
                # 2. Update Vector DB
                doc_text = f"Symptoms for {data['disease']}: " + ", ".join(data["symptoms"])
                doc_id = f"disease_custom_{data['disease'].replace(' ', '_').lower()}"
                
                self.collection.upsert(
                    documents=[doc_text],
                    metadatas=[{"disease": data["disease"], "symptoms": ", ".join(data["symptoms"])}],
                    ids=[doc_id]
                )
                
                self.existing_diseases.add(data["disease"].lower())
                print(f"  -> Successfully added {data['disease']} into Vector DB and CSVs!")
                
                # Respect Groq/Tavily rate limits
                time.sleep(3)
                
            except Exception as e:
                print(f"  -> Unexpected error on {disease}: {e}")

if __name__ == "__main__":
    # Add hundreds of diseases here to expand as needed.
    # We are starting with these 50 common/global diseases not originally in the default 41 set.
    NEW_DISEASES_TO_ADD = [
        "COVID-19", "Influenza", "Strep Throat", "Lyme Disease", "Pneumonia",
        "Zika Virus", "Cholera", "Tuberculosis", "Measles", "Mumps", 
        "Rubella", "Tetanus", "Rabies", "Anthrax", "Malaria", 
        "Dengue Fever", "Yellow Fever", "Typhoid Fever", "Chikungunya", "Ebola",
        "Polio", "Leprosy", "Syphilis", "Gonorrhea", "Chlamydia",
        "HIV/AIDS", "Hepatitis A", "Hepatitis B", "Hepatitis C", "Herpes",
        "Meningitis", "Encephalitis", "Sepsis", "Rheumatic Fever", "Scarlet Fever",
        "Whooping Cough", "Diphtheria", "Botulism", "Toxoplasmosis", "Giardiasis",
        "Amebiasis", "Schistosomiasis", "Leishmaniasis", "Chagas Disease", "Sleeping Sickness",
        "Scabies", "Ringworm", "Candidiasis", "Tinea", "Aspergillosis"
    ]
    
    expander = BatchDatabaseExpander()
    expander.expand_database(NEW_DISEASES_TO_ADD)
    
    print("\nBatch process completed. Vector DB and CSVs updated.")
