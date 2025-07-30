import faiss
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from scipy.sparse import vstack
import os

class HybridSearcher:
    def __init__(self):
        # Load FAISS index
        self.index = faiss.read_index("vectorstore/faiss_index.index")
        
        # Load metadata
        with open("vectorstore/metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        # Load TF-IDF data
        with open("vectorstore/tfidf.pkl", "rb") as f:
            tfidf_data = pickle.load(f)
            self.tfidf_vectorizer = tfidf_data['vectorizer']
            self.tfidf_vectors = tfidf_data['vectors']
        
        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def search(self, query, k=5, semantic_weight=0.7):
        # Semantic search
        query_vector = self.model.encode([query])
        semantic_distances, semantic_indices = self.index.search(query_vector, k)
        
        # TF-IDF search
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_vectors @ query_tfidf.T).toarray().flatten()
        
        # Fuzzy matching
        fuzzy_scores = np.array([
            max(fuzz.ratio(query.lower(), text.lower()) for text in self.tfidf_vectorizer.get_feature_names_out())
            for _ in range(len(self.metadata))
        ])
        
        # Combine scores
        semantic_scores = 1 / (1 + semantic_distances[0])  # Convert distances to similarities
        tfidf_scores = tfidf_scores / np.max(tfidf_scores) if np.max(tfidf_scores) > 0 else tfidf_scores
        fuzzy_scores = fuzzy_scores / 100  # Normalize fuzzy scores
        
        combined_scores = (
            semantic_weight * semantic_scores +
            (1 - semantic_weight) * 0.7 * tfidf_scores +
            (1 - semantic_weight) * 0.3 * fuzzy_scores
        )
        
        # Get top results
        top_indices = np.argsort(combined_scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'metadata': self.metadata[idx],
                'score': float(combined_scores[idx]),
                'semantic_score': float(semantic_scores[idx]),
                'tfidf_score': float(tfidf_scores[idx]),
                'fuzzy_score': float(fuzzy_scores[idx])
            })
        
        return results

# Example usage
if __name__ == "__main__":
    searcher = HybridSearcher()
    results = searcher.search("machine learning", k=5)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Values: {result['metadata']['values']}")
        print(f"Total Score: {result['score']:.3f}")
        print(f"Semantic: {result['semantic_score']:.3f}")
        print(f"TF-IDF: {result['tfidf_score']:.3f}")
        print(f"Fuzzy: {result['fuzzy_score']:.3f}")