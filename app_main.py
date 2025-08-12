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
import re
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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
        
        # Initialize GROQ client with environment variable
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            error_msg = """
            GROQ_API_KEY environment variable is not set.
            
            For local development:
            1. Copy .env.example to .env
            2. Add your GROQ API key to the .env file
            
            For deployment:
            1. Set GROQ_API_KEY as an environment variable in your deployment platform
            2. Get your API key from: https://console.groq.com/
            """
            raise ValueError(error_msg)
        
        self.groq_client = Groq(api_key=groq_api_key)

    def detect_query_intent(self, query: str) -> str:
        """
        Use GROQ LLM to detect query intent and classify into categories.
        """
        try:
            prompt = f"""
You are an expert query classifier for a search system. Analyze the user's query and classify it into exactly one category.

Categories and Examples:
1. "tools" - software, applications, platforms, systems, technology, AI tools, productivity tools, design software, automation tools, dashboards, solutions, interfaces
   Examples: "project management tool", "design software", "AI automation platform", "productivity app"

2. "courses" - learning, training, education, courses, tutorials, workshops, certifications, skills, teaching, programs, curriculum, classes
   Examples: "Python course", "data science training", "web development tutorial", "certification program"

3. "service-providers" - vendors, providers, companies, agencies, consultants, businesses, professionals, experts, freelancers, firms, contractors
   Examples: "web development company", "consulting firm", "graphic designer", "IT services", "marketing agency"

4. "case-studies" - case studies, examples, success stories, implementations, experiences, best practices, projects, real-world examples
   Examples: "success story", "implementation example", "case study", "real world example", "business case"

5. "all" - very general queries, ambiguous queries, or queries that could apply to multiple categories
   Examples: "help", "search", "what do you have", "show me options"

User Query: "{query}"

Important Rules:
- Respond with ONLY the category name (exactly one of: tools, courses, service-providers, case-studies, all)
- Focus on the main intent and key words in the query
- If the query mentions software, applications, platforms, or technology solutions, classify as "tools"
- If the query mentions companies, agencies, providers, or professional services, classify as "service-providers"
- If the query is very general or unclear, classify as "all"

Category:"""

            completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=10
            )
            
            result = completion.choices[0].message.content.strip().lower()
            
            # Validate the result
            valid_categories = ['tools', 'courses', 'service-providers', 'case-studies', 'all']
            if result in valid_categories:
                return result
            else:
                # Fallback to 'all' if invalid response
                return 'all'
                
        except Exception as e:
            print(f"Error in GROQ intent detection: {e}")
            # Fallback to 'all' on error
            return 'all'
    
    def filter_by_category(self, results: list[dict], target_category: str) -> list[dict]:
        if target_category == 'all':
            return results
        
        filtered_results = []
        for result in results:
            sheet_name = result['metadata'].get('sheet', '').lower()
            values = result['metadata'].get('values', [])
            category_val = str(values[0]).lower() if values else ''
            
            # Match based on actual sheet names from the data
            if target_category == 'tools' and (
                'cleaned sheet' in sheet_name or 
                'tool' in category_val or 
                'ai tools' in category_val or
                'software' in category_val.lower()):
                filtered_results.append(result)
                
            elif target_category == 'courses' and (
                'training program' in sheet_name or 
                'training' in sheet_name or 
                'course' in category_val or
                'education' in category_val):
                filtered_results.append(result)
                
            elif target_category == 'service-providers' and (
                'service provider profiles' in sheet_name or 
                'service' in sheet_name or 
                'provider' in sheet_name or 
                'provider' in category_val or 
                'vendor' in category_val or
                'company' in category_val):
                filtered_results.append(result)
                
            elif target_category == 'case-studies' and (
                'case-studies' in sheet_name or 
                'case study' in sheet_name or
                sheet_name == 'case-studies'):
                filtered_results.append(result)
        
        # If filter is too strict and nothing is found, fall back to all results
        if not filtered_results and target_category != 'all':
            print(f"Warning: No results found for category '{target_category}', returning all results")
            return results
            
        return filtered_results

    def search(self, query: str, k: int = 20, min_score: float = 0.30) -> tuple[list[dict], str]:
        detected_category = self.detect_query_intent(query)
        
        # Semantic search component
        query_vector = self.model.encode([query])
        # Use larger multiplier for case-studies to ensure all case studies are found
        if detected_category == 'case-studies':
            search_multiplier = 30  # Even larger to catch ALL case studies including low-similarity ones
        elif detected_category != 'all':
            search_multiplier = 3
        else:
            search_multiplier = 2
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
            
            # Boost case study scores to ensure they stay well above threshold
            if detected_category == 'case-studies' and item.get('sheet', '').lower().find('case') != -1:
                hybrid_score = min(1.0, hybrid_score + 0.05)  # Small boost, cap at 1.0
            
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
    st.markdown("### Search across tools, service providers, training courses, and case studies")
    st.markdown("*🤖 Powered by **Semantic Search (70%) + TF-IDF Keyword Matching (30%)**  for the best results*")

    # Initialize searcher and handle missing data
    try:
        searcher = initialize_searcher()
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        return

    query = st.text_input("Enter your search query:", placeholder="e.g. best AI tools, learn python, find a vendor, digital transformation case study")
    if query:
        if len(query.strip()) < 3:
            st.warning("Please enter a longer search query.")
        else:
            with st.spinner("🔍 Searching..."):
                results, detected_category = searcher.search(query, k=20, min_score=0.3)
                category_icons = {
                    'tools': '🛠️ Tools',
                    'courses': '📚 Courses',
                    'service-providers': '🏢 Providers',
                    'case-studies': '📋 Case Studies',
                    'all': '🌐 All Categories'
                }
                detected_display = category_icons.get(detected_category, detected_category)
                st.info(f"🤖 **AI Detected Intent:** {detected_display}")
                if results:
                    for i, res in enumerate(results, start=1):
                        header = ' | '.join(res['metadata'].get('values', []))
                        source_sheet = res['metadata'].get('sheet', 'Unknown')
                        
                        # Determine emoji based on source
                        if "case-studies" in source_sheet.lower():
                            source_emoji = "�"
                        elif "tools" in source_sheet.lower():
                            source_emoji = "🛠️"
                        elif "training" in source_sheet.lower():
                            source_emoji = "📚"
                        else:
                            source_emoji = "🏢"
                        
                        # Handle case studies display
                        if "case-studies" in source_sheet.lower():
                            case_study_title = res['metadata'].get('values', ['Unknown'])[0]
                            industry = res['metadata'].get('industry', 'Unknown')
                            problem_type = res['metadata'].get('problem_type', 'Unknown')
                            word_count = res['metadata'].get('word_count', 0)
                            
                            with st.expander(f"{source_emoji} Case Study {i}: {case_study_title} (Score: {res['score']:.3f})"):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown("#### Case Study Details")
                                    st.write(f"**Title:** {case_study_title}")
                                    st.write(f"**Industry:** {industry}")
                                    st.write(f"**Problem Type:** {problem_type}")
                                
                                with col2:
                                    st.markdown("#### Relevance Scores")
                                    st.progress(res['score'])
                                    st.write(f"**Hybrid Score:** {res['score']:.3f}")
                                    st.write(f"🧠 Semantic: {res['semantic_score']:.3f} (70%)")
                                    st.write(f"🔍 TF-IDF: {res['tfidf_score']:.3f} (30%)")
                        else:
                            # Original display for other types
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
    