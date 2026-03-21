import os
import json
import chromadb
from chromadb.utils import embedding_functions
from tavily import TavilyClient
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class WebSearchSubagent:
    def __init__(self):
        self.tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        
    def search_disease(self, disease_name: str) -> str:
        query = f"What are the medical symptoms of {disease_name}? Provide a detailed list."
        try:
            response = self.tavily_client.search(query, search_depth="advanced", max_results=3)
            # Combine the content from the search results
            context = ""
            for result in response.get('results', []):
                context += result.get('content', '') + "\n"
            return context
        except Exception as e:
            print(f"Error during web search: {e}")
            return ""

class LLMStructurerSubagent:
    def __init__(self, model="openai/gpt-oss-120b"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model
        
    def structure_symptoms(self, disease_name: str, raw_context: str) -> list[str]:
        system_prompt = f"""
You are an expert Medical Data Structurer. Your task is to extract a list of medical symptoms for '{disease_name}' from the provided raw web context.
Output ONLY a raw JSON array of strings representing the symptoms in lowercase, using underscores instead of spaces (e.g., ["high_fever", "chest_pain"]).
Do not include any conversational text, explanations, or markdown blocks. If no symptoms are found, return [].
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_context}
            ],
            temperature=0.0
        )
        
        try:
            content = response.choices[0].message.content.strip()
            # Clean up markdown if present
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            symptoms = json.loads(content)
            return symptoms
        except Exception as e:
            print(f"Error structuring LLM response: {e}")
            return []

class MedicalUpdaterAgent:
    def __init__(self, db_path="./chroma_db"):
        self.search_agent = WebSearchSubagent()
        self.structurer_agent = LLMStructurerSubagent()
        
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.db_client.get_or_create_collection(
            name="disease_symptoms",
            embedding_function=self.sentence_transformer_ef
        )

    def update_knowledge(self, disease_name: str):
        print(f"[{disease_name} Updater] Searching the web for latest symptoms...")
        raw_context = self.search_agent.search_disease(disease_name)
        
        if not raw_context:
            print(f"[{disease_name} Updater] Could not retrieve meaningful search results.")
            return False
            
        print(f"[{disease_name} Updater] Structuring data using LLM...")
        structured_symptoms = self.structurer_agent.structure_symptoms(disease_name, raw_context)
        
        if not structured_symptoms:
            print(f"[{disease_name} Updater] LLM failed to extract symptoms.")
            return False
            
        print(f"[{disease_name} Updater] Found symptoms: {', '.join(structured_symptoms)}")
        print(f"[{disease_name} Updater] Injecting into ChromaDB...")
        
        doc_text = f"Symptoms for {disease_name}: " + ", ".join(structured_symptoms)
        doc_id = f"disease_custom_{disease_name.replace(' ', '_').lower()}"
        
        # Upsert into DB
        self.collection.upsert(
            documents=[doc_text],
            metadatas=[{"disease": disease_name, "symptoms": ", ".join(structured_symptoms)}],
            ids=[doc_id]
        )
        
        print(f"[{disease_name} Updater] Success! {disease_name} is now in your local medical knowledge base.")
        return True
