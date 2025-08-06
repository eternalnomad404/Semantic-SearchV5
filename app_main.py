# app_main.py

import os
import faiss
import json
from sentence_transformers import SentenceTransformer
import streamlit as st
import numpy as np

class SemanticSearcher:
    def __init__(self):
        # Update paths to be relative
        self.index = faiss.read_index("vectorstore/faiss_index.index")
        with open("vectorstore/metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, query, k=20, min_score=0.30):
        # Get semantic search results only
        query_vector = self.model.encode([query])
        D, I = self.index.search(query_vector, k*2)  # Getting more candidates initially
        
        results = []
        seen_content = set()  # Track unique content
        
        for dist, idx in zip(D[0], I[0]):
            if idx >= len(self.metadata):
                continue
            
            # Create a unique identifier for the content
            content_key = tuple(str(v) for v in self.metadata[idx]['values'])
            
            # Skip if we've seen this content before
            if content_key in seen_content:
                continue
            
            # Only semantic score - no TF-IDF
            semantic_score = 1 / (1 + dist)
            
            if semantic_score >= min_score:
                metadata = self.metadata[idx]
                results.append({
                    'metadata': metadata,
                    'score': float(semantic_score)  # Only one score now
                })
                seen_content.add(content_key)  # Add to seen content
        
        # Sort by semantic score and take top k unique results
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results[:k] if results else []

@st.cache_resource
def initialize_searcher():
    return SemanticSearcher()

def main():
    st.set_page_config(
        page_title="Semantic Search System",
        page_icon="🔎",
        layout="wide"
    )

    st.title("🔎 Semantic Search System")
    st.markdown("### Search across tools, service providers, and training courses")

    try:
        searcher = initialize_searcher()
    except FileNotFoundError:
        st.error("⚠️ Vector store not found. Please run generate_embeddings.py first.")
        return

    # Search interface - removed sliders, using fixed values
    query = st.text_input("Enter your search query:", placeholder="Example: machine learning tools")

    if query:
        if len(query.strip()) < 3:
            st.warning("Please enter a longer search query")
            return

        with st.spinner("🔍 Searching..."):
            # Fixed values: k=20, min_score=0.30
            results = searcher.search(query, k=20, min_score=0.30)

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
                            st.markdown("#### Relevance Score")
                            st.progress(result['score'])
                            st.write(f"Semantic Score: {result['score']:.3f}")
            else:
                st.info("No relevant results found for your query. Please try different search terms.")

    with st.sidebar:
        st.markdown("### About")
        st.write("""
        This search system uses:
        - Pure Semantic Search (FAISS)
        - Neural Language Model (all-MiniLM-L6-v2)
        """)

if __name__ == "__main__":
    main()
