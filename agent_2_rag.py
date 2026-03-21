import chromadb
from chromadb.utils import embedding_functions

class DiagnosticRAGAgent:
    def __init__(self, db_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_collection(
            name="disease_symptoms",
            embedding_function=self.sentence_transformer_ef
        )

    def retrieve_disease(self, extracted_symptoms: list[str], top_k=2):
        if not extracted_symptoms:
            return []

        query_text = ", ".join(extracted_symptoms)
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        # Parse the results
        predictions = []
        if results and 'metadatas' in results and results['metadatas']:
            for metadata_row in results['metadatas'][0]:
                predictions.append(metadata_row['disease'])
                
        return predictions
