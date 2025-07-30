# app_main.py

import os
import faiss
import json
from sentence_transformers import SentenceTransformer
import streamlit as st
import pickle
import numpy as np

class HybridSearcher:
    def __init__(self):
        self.index = faiss.read_index("vectorstore/faiss_index.index")
        with open("vectorstore/metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        with open("vectorstore/tfidf.pkl", "rb") as f:
            tfidf_data = pickle.load(f)
            self.tfidf_vectorizer = tfidf_data['vectorizer']
            self.tfidf_vectors = tfidf_data['vectors']
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, query, k=5, min_score=0.4):  # Lowered min_score default
        # Get semantic search results
        query_vector = self.model.encode([query])
        D, I = self.index.search(query_vector, k*2)
        
        # Get TF-IDF similarity
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_vectors @ query_tfidf.T).toarray().flatten()
        
        # Calculate vocabulary overlap
        query_terms = set(query.lower().split())
        
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx >= len(self.metadata):
                continue
                
            semantic_score = 1 / (1 + dist)
            tfidf_score = tfidf_scores[idx]
            
            # Check text overlap with more lenient criteria
            metadata_text = " ".join(str(v) for v in self.metadata[idx]['values']).lower()
            text_overlap = any(term in metadata_text for term in query_terms)
            
            # Calculate combined score
            combined_score = (0.7 * semantic_score) + (0.3 * tfidf_score)
            
            # More lenient inclusion criteria
            if combined_score >= min_score:
                metadata = self.metadata[idx]
                results.append({
                    'metadata': metadata,
                    'score': float(combined_score),
                    'semantic_score': float(semantic_score),
                    'tfidf_score': float(tfidf_score)
                })
        
        # Sort by combined score
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results[:k] if results else []

@st.cache_resource
def initialize_searcher():
    return HybridSearcher()

def main():
    st.set_page_config(
        page_title="Hybrid Search System",
        page_icon="🔎",
        layout="wide"
    )

    st.title("🔎 Hybrid Search System")
    st.markdown("### Search across tools, service providers, and training courses")

    try:
        searcher = initialize_searcher()
    except FileNotFoundError:
        st.error("⚠️ Vector store not found. Please run generate_embeddings.py first.")
        return

    query = st.text_input("Enter your search query:", placeholder="Example: machine learning tools")
    num_results = st.slider("Number of results:", min_value=1, max_value=20, value=5)  # Changed max_value to 20
    min_score = st.slider("Minimum relevance score:", 
                         min_value=0.3, 
                         max_value=1.0, 
                         value=0.4, 
                         step=0.05)

    if query:
        if len(query.strip()) < 3:
            st.warning("Please enter a longer search query")
            return

        with st.spinner("🔍 Searching..."):
            results = searcher.search(query, k=num_results, min_score=min_score)

            if results:
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i}: {' | '.join(result['metadata']['values'])} (Score: {result['score']:.3f})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("#### Details")
                            for header, value in zip(result['metadata']['column_headers'], 
                                                   result['metadata']['values']):
                                st.write(f"**{header}:** {value}")
                            st.write(f"**Source:** {result['metadata']['sheet']}")

                        with col2:
                            st.markdown("#### Relevance Scores")
                            st.progress(result['score'])
                            st.write(f"Total Score: {result['score']:.3f}")
                            st.write(f"Semantic Score: {result['semantic_score']:.3f}")
                            st.write(f"TF-IDF Score: {result['tfidf_score']:.3f}")
            else:
                st.info("No relevant results found for your query. Please try different search terms.")

    with st.sidebar:
        st.markdown("### About")
        st.write("""
        This search system combines:
        - Semantic Search (FAISS)
        - TF-IDF Keyword Matching
        - Text Overlap Verification
        """)

if __name__ == "__main__":
    main()

# Example search queries and results (to be removed in production)

# 🔎 Searching for: 'machine learning'
# Rank 1 (Score: 0.842):
# Sheet: training-courses
# Skill: Machine Learning
# Topic: Deep Learning
# ...

# 🔎 Searching for: 'project management'
# ...
