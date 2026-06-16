import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

class RealEstateVectorStore:
    def __init__(self, csv_path="real_estate_llm_qa.csv"):
        self.csv_path = csv_path
        # Lightweight local transformer engine embedding model
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.df = None
        self.build_store()

    def build_store(self):
        if not os.path.exists(self.csv_path):
            # Create a base skeleton if the file is missing or wiped
            self.df = pd.DataFrame(columns=["Category", "Question", "Answer"])
            self.df.to_csv(self.csv_path, index=False)
        else:
            self.df = pd.read_csv(self.csv_path)

        if not self.df.empty:
            # Clean missing rows if any exist
            self.df = self.df.dropna(subset=["Question", "Answer"])
            # Generate vectors
            questions = self.df["Question"].astype(str).tolist()
            embeddings = self.encoder.encode(questions, show_progress_bar=False)
            embeddings = np.array(embeddings).astype("float32")
            
            # Setup Flat L2 Indexing 
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
        else:
            self.index = None

    def search_similar(self, query, top_k=2):
        if self.index is None or self.df.empty:
            return []
            
        query_vector = np.array([self.encoder.encode(query)]).astype("float32")
        distances, indices = self.index.search(query_vector, min(top_k, len(self.df)))
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                results.append({
                    "category": self.df.iloc[idx]["Category"],
                    "question": self.df.iloc[idx]["Question"],
                    "answer": self.df.iloc[idx]["Answer"],
                    "confidence_score": float(dist)
                })
        return results

    def append_data(self, category, question, answer):
        new_row = pd.DataFrame([{"Category": category, "Question": question, "Answer": answer}])
        new_row.to_csv(self.csv_path, mode='a', header=False, index=False)
        self.build_store() # Re-index vectors