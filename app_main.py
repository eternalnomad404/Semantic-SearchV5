# app_main.py

import os
import faiss
import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


class SemanticSearcher:
    """
    A pure semantic searcher using FAISS and a SentenceTransformer model.
    """
    def __init__(self, 
                 index_path: str = "vectorstore/faiss_index.index", 
                 metadata_path: str = "vectorstore/metadata.json",
                 model_name: str = "all-MiniLM-L6-v2"):
        # Load FAISS index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.index = faiss.read_index(index_path)

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata JSON not found at {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Load embedding model
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, k: int = 20, min_score: float = 0.30) -> list[dict]:
        """
        Perform a semantic-only search and return up to `k` results 
        with score >= `min_score`.
        """
        # Encode query and search in FAISS
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(query_vector, k * 2)

        results: list[dict] = []
        seen_keys: set[tuple] = set()

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            item = self.metadata[idx]

            # Unique key to avoid duplicates
            key = tuple(str(v) for v in item.get('values', []))
            if key in seen_keys:
                continue

            score = 1 / (1 + dist)
            if score < min_score:
                continue

            results.append({
                'metadata': item,
                'score': float(score)
            })
            seen_keys.add(key)

        # Sort by score descending and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]


@st.cache_resource
def initialize_searcher() -> SemanticSearcher:
    """Initialize and cache the SemanticSearcher resource."""
    return SemanticSearcher()


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(
        page_title="Semantic Search System",
        page_icon="🔎",
        layout="wide"
    )

    st.title("🔎 Semantic Search System")
    st.markdown("### Search across tools, service providers, and training courses")

    # Initialize searcher and handle missing data
    try:
        searcher = initialize_searcher()
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        return

    # Simple search input - no filter
    query = st.text_input("Enter your search query:", placeholder="Example: machine learning tools")

    if query:
        if len(query.strip()) < 3:
            st.warning("Please enter a longer search query.")
        else:
            with st.spinner("🔍 Searching..."):
                results = searcher.search(query, k=20, min_score=0.30)

                if results:
                    for i, res in enumerate(results, start=1):
                        header = ' | '.join(res['metadata'].get('values', []))
                        with st.expander(f"Result {i}: {header} (Score: {res['score']:.3f})"):
                            detail_col, score_col = st.columns([2, 1])
                            with detail_col:
                                st.markdown("#### Details")
                                for key, value in zip(
                                    res['metadata'].get('column_headers', []),
                                    res['metadata'].get('values', [])
                                ):
                                    st.write(f"**{key}:** {value}")
                                st.write(f"**Source:** {res['metadata'].get('sheet', 'Unknown')}")

                            with score_col:
                                st.markdown("#### Relevance Score")
                                st.progress(res['score'])
                                st.write(f"Semantic Score: {res['score']:.3f}")
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
    