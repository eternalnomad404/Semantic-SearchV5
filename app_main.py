# app_main.py

import os
import faiss
import json
import numpy as np
import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearcher:
    """
    Hybrid searcher combining semantic search with TF-IDF keyword matching.
    Final score = 0.7 * semantic_score + 0.3 * tfidf_score
    """
    def __init__(self, 
                 index_path: str = "vectorstore/faiss_index.index", 
                 metadata_path: str = "vectorstore/metadata.json",
                 tfidf_path: str = "vectorstore/tfidf.pkl",
                 model_name: str = "all-MiniLM-L6-v2"):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.index = faiss.read_index(index_path)
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata JSON not found at {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        if not os.path.exists(tfidf_path):
            raise FileNotFoundError(f"TF-IDF data not found at {tfidf_path}")
        with open(tfidf_path, "rb") as f:
            tfidf_data = pickle.load(f)
            self.tfidf_vectorizer = tfidf_data['vectorizer']
            self.tfidf_vectors = tfidf_data['vectors']
            
        self.model = SentenceTransformer(model_name)
        self.category_patterns = {
            'tools': [
                'tool', 'software', 'platform', 'application', 'system', 'technology',
                'ai tool', 'productivity', 'design', 'automation', 'workflow', 'dashboard', 'solution', 'interface'
            ],
            'courses': [
                'learn', 'course', 'training', 'education', 'program', 'curriculum', 'study',
                'tutorial', 'workshop', 'certification', 'skill', 'knowledge', 'teach', 'instructor', 'class', 'lesson', 'academy', 'university', 'institute'
            ],
            'service-providers': [
                'vendor', 'provider', 'company', 'agency', 'firm', 'consultant', 'service',
                'business', 'organization', 'supplier', 'contractor', 'partner', 'client', 'professional', 'expert', 'specialist', 'freelancer', 'team'
            ]
        }

    def detect_query_intent(self, query: str) -> str:
        query_lower = query.lower()
        category_scores = {}
        for category, patterns in self.category_patterns.items():
            pattern_text = ' '.join(patterns)
            query_embedding = self.model.encode([query_lower])
            pattern_embedding = self.model.encode([pattern_text])
            similarity = np.dot(query_embedding, pattern_embedding.T) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(pattern_embedding)
            )
            category_scores[category] = similarity[0][0]
            for pattern in patterns:
                if pattern in query_lower:
                    category_scores[category] += 0.3
        if not category_scores:
            return 'all'
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        if best_score > 0.6:
            return best_category
        else:
            return 'all'

    def filter_by_category(self, results: list[dict], target_category: str) -> list[dict]:
        if target_category == 'all':
            return results
        filtered_results = []
        for result in results:
            sheet_name = result['metadata'].get('sheet', '').lower()
            values = result['metadata'].get('values', [])
            category_val = str(values[0]).lower() if values else ''
            # For tools, check both sheet and category value
            if target_category == 'tools' and (
                'tools' in sheet_name or 'tool' in category_val or 'ai tools' in category_val):
                filtered_results.append(result)
            elif target_category == 'courses' and (
                'training' in sheet_name or 'program' in sheet_name or 'course' in category_val):
                filtered_results.append(result)
            elif target_category == 'service-providers' and (
                'service' in sheet_name or 'provider' in sheet_name or 'provider' in category_val or 'vendor' in category_val):
                filtered_results.append(result)
        # If filter is too strict and nothing is found, fall back to all
        if not filtered_results:
            return results
        return filtered_results

    def search(self, query: str, k: int = 20, min_score: float = 0.30) -> tuple[list[dict], str]:
        detected_category = self.detect_query_intent(query)
        
        # Semantic search component
        query_vector = self.model.encode([query])
        search_multiplier = 3 if detected_category != 'all' else 2
        distances, indices = self.index.search(query_vector, k * search_multiplier)
        
        # TF-IDF search component
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_vectors).flatten()
        
        results: list[dict] = []
        seen_keys: set[tuple] = set()
        tool_names_seen: dict[str, dict] = {}  # Track tool names with their highest scores
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            key = tuple(str(v) for v in item.get('values', []))
            if key in seen_keys:
                continue
                
            # Calculate semantic score (0-1 range)
            semantic_score = 1 / (1 + dist)
            
            # Get TF-IDF score for this document (0-1 range)
            tfidf_score = float(tfidf_similarities[idx]) if idx < len(tfidf_similarities) else 0.0
            
            # Calculate hybrid score: 70% semantic + 30% TF-IDF
            hybrid_score = 0.7 * semantic_score + 0.3 * tfidf_score
            
            if hybrid_score < min_score:
                continue
            
            result_entry = {
                'metadata': item,
                'score': float(hybrid_score),
                'semantic_score': float(semantic_score),
                'tfidf_score': float(tfidf_score)
            }
            
            # Check if this is from tools sheet and handle deduplication by tool name
            sheet_name = item.get('sheet', '').lower()
            values = item.get('values', [])
            
            if 'cleaned sheet' in sheet_name and len(values) >= 3:
                # This is from tools sheet, check for duplicate tool names
                tool_name = str(values[2]).strip().lower()  # Name of Tool is in index 2
                
                if tool_name in tool_names_seen:
                    # We've seen this tool name before, keep only the higher scoring one
                    if hybrid_score > tool_names_seen[tool_name]['score']:
                        # Remove the previous lower-scoring entry
                        results = [r for r in results if not (
                            r['metadata'].get('sheet', '').lower() == 'cleaned sheet' and
                            len(r['metadata'].get('values', [])) >= 3 and
                            str(r['metadata']['values'][2]).strip().lower() == tool_name
                        )]
                        # Add the new higher-scoring entry
                        results.append(result_entry)
                        tool_names_seen[tool_name] = result_entry
                    # If current score is lower, don't add it
                else:
                    # First time seeing this tool name
                    results.append(result_entry)
                    tool_names_seen[tool_name] = result_entry
            else:
                # Not from tools sheet, add normally
                results.append(result_entry)
            
            seen_keys.add(key)
        
        filtered_results = self.filter_by_category(results, detected_category)
        filtered_results.sort(key=lambda x: x['score'], reverse=True)
        return filtered_results[:k], detected_category


@st.cache_resource
def initialize_searcher() -> SemanticSearcher:
    """Initialize and cache the SemanticSearcher resource."""
    return SemanticSearcher()


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(
        page_title="Hybrid Search System",
        page_icon="🔎",
        layout="wide"
    )

    st.title("🔎 Hybrid Search System")
    st.markdown("### Search across tools, service providers, and training courses")
    st.markdown("*🤖 Powered by **Semantic Search (70%) + TF-IDF Keyword Matching (30%)**  for the best results*")

    # Initialize searcher and handle missing data
    try:
        searcher = initialize_searcher()
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        return

    query = st.text_input("Enter your search query:", placeholder="e.g. best AI tools, learn python, find a vendor")
    if query:
        if len(query.strip()) < 3:
            st.warning("Please enter a longer search query.")
        else:
            with st.spinner("🔍 Searching..."):
                results, detected_category = searcher.search(query, k=20, min_score=0.30)
                category_icons = {
                    'tools': '🛠️ Tools',
                    'courses': '📚 Courses',
                    'service-providers': '🏢 Providers',
                    'all': '🌐 All Categories'
                }
                detected_display = category_icons.get(detected_category, detected_category)
                st.info(f"🤖 **AI Detected Intent:** {detected_display}")
                if results:
                    for i, res in enumerate(results, start=1):
                        header = ' | '.join(res['metadata'].get('values', []))
                        source_sheet = res['metadata'].get('sheet', 'Unknown')
                        source_emoji = "🛠️" if "tools" in source_sheet.lower() else "📚" if "training" in source_sheet.lower() else "🏢"
                        with st.expander(f"{source_emoji} Result {i}: {header} (Hybrid Score: {res['score']:.3f})"):
                            detail_col, score_col = st.columns([2, 1])
                            with detail_col:
                                st.markdown("#### Details")
                                for key, value in zip(
                                    res['metadata'].get('column_headers', []),
                                    res['metadata'].get('values', [])
                                ):
                                    st.write(f"**{key}:** {value}")
                                st.write(f"**Source:** {source_emoji} {source_sheet}")
                            with score_col:
                                st.markdown("#### Relevance Scores")
                                st.progress(res['score'])
                                st.write(f"**Hybrid Score:** {res['score']:.3f}")
                                st.write(f"🧠 Semantic: {res['semantic_score']:.3f} (70%)")
                                st.write(f"🔍 TF-IDF: {res['tfidf_score']:.3f} (30%)")
                else:
                    st.info(f"No {detected_display.lower()} found for your query. Try different search terms or be more specific.")

    with st.sidebar:
        st.markdown("### About")
        st.write("""
        🤖 **Hybrid AI Search** combining:
        - **70% Semantic Search**: Understanding context and meaning
        - **30% TF-IDF Keyword**: Exact keyword matching
        
        Results are filtered by your intent when clear, or show all when ambiguous. No manual selection needed.
        """)
        
        st.markdown("### Search Tips")
        st.write("""
        - **Specific terms**: Use exact keywords for better TF-IDF matching
        - **Concepts**: Use descriptive phrases for better semantic matching
        - **Best results**: Combine both approaches in your query
        """)


if __name__ == "__main__":
    main()
    