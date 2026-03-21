import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

def setup_chroma():
    # Load dataset
    df = pd.read_csv('Data/Training.csv')
    symptoms_list = df.columns[:-1]
    
    # We will aggregate symptoms per disease
    # A single disease can have multiple rows representing different combinations.
    # To be concise, we can group by prognosis and take all symptoms that ever appear for that disease.
    grouped = df.groupby('prognosis').max()
    
    # Initialize Chroma local client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create or get collection
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(
        name="disease_symptoms", 
        embedding_function=sentence_transformer_ef
    )
    
    documents = []
    metadatas = []
    ids = []
    
    # Prepare documents
    for index, (disease, row) in enumerate(grouped.iterrows()):
        # Get symptoms that are 1
        active_symptoms = row[row == 1].index.tolist()
        doc_text = f"Symptoms for {disease}: " + ", ".join(active_symptoms)
        
        documents.append(doc_text)
        metadatas.append({"disease": disease, "symptoms": ", ".join(active_symptoms)})
        ids.append(f"disease_{index}")
        
    print(f"Adding {len(documents)} diseases to the Vector DB...")
    
    # Upsert to Chroma DB
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print("Vector DB initialized successfully at ./chroma_db")

if __name__ == "__main__":
    setup_chroma()
