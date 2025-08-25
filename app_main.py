# app_main.py

import os
import faiss
import json
import numpy as np
import streamlit as st
import pickle
import subprocess
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_git_commit_hash():
    """Get current git commit hash"""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        return commit_hash
    except:
        return "unknown"


class SemanticSearcher:
    """
    Hybrid searcher combining semantic sear                        # Create clickable title with hyperlink
                        clickable_title = f"[{display_header}]({result_url})"
                        
                        # Handle case studies display
                        if "case-studies" in source_sheet.lower():keyword matching.
    Final score = 0.7 * semantic_score + 0.3 * tfidf_score
    """
    def __init__(self, 
                 index_path: str = "vectorstore/faiss_index.index", 
                 metadata_path: str = "vectorstore/metadata.json",
                 tfidf_path: str = "vectorstore/tfidf.pkl",
                 model_name: str = "all-MiniLM-L6-v2"):
        
        # Define case study URLs mapping
        self.case_study_urls = {
            "learning link foundation": "https://dt4si.com/case-studies/learning-link-foundation",
            "farmers for forest": "https://dt4si.com/case-studies/farmer-for-forest", 
            "i-saksham": "https://dt4si.com/case-studies/i-saksham",
            "vipla foundation": "https://dt4si.com/case-studies/vipla-foundation",
            "educate girls predictive targeting to enroll girls": "https://dt4si.com/case-studies/educate-girls-predictive-targeting-to-enroll-girls",
            "the akshaya patra foundation": "https://dt4si.com/case-studies/the-akshaya-patra-foundation",
            "armman": "https://dt4si.com/case-studies/armman",
            "lend a hand india": "https://dt4si.com/case-studies/lend-a-hand-india",
            "anudip": "https://dt4si.com/case-studies/anudip",
            "fmch": "https://dt4si.com/case-studies/fmch",
            "educate girls": "https://dt4si.com/case-studies/educate-girls"
        }
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
        
        # Initialize GROQ client lazily (only when needed) - NOT USED ANYMORE
        # Keeping for backward compatibility in case other code references it
        self.groq_client = None

    def _create_url_slug(self, text: str) -> str:
        """Convert text to URL-friendly slug."""
        import re
        # Convert to lowercase
        slug = text.lower()
        # Replace spaces and special characters with hyphens
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'\s+', '-', slug)
        # Remove multiple hyphens
        slug = re.sub(r'-+', '-', slug)
        # Remove leading/trailing hyphens
        slug = slug.strip('-')
        return slug
    
    def _generate_result_url(self, result: dict) -> str:
        """Generate appropriate URL for each result based on category."""
        sheet_name = result['metadata'].get('sheet', '').lower()
        values = result['metadata'].get('values', [])
        
        try:
            # Tools - from "Cleaned Sheet"
            if 'cleaned sheet' in sheet_name and len(values) >= 3:
                tool_name = str(values[2]).strip()  # "Name of Tool" is at index 2
                slug = self._create_url_slug(tool_name)
                return f"https://dt4si.com/tools/{slug}"
            
            # Service Providers - from "Service Provider Profiles"  
            elif 'service provider profiles' in sheet_name and len(values) >= 1:
                provider_name = str(values[0]).strip()  # "Name of Service Provider" is at index 0
                slug = self._create_url_slug(provider_name)
                return f"https://dt4si.com/services/{slug}"
            
            # Courses - from "Training Program"
            elif 'training program' in sheet_name and len(values) >= 3:
                course_title = str(values[2]).strip()  # "Course Title" is at index 2
                slug = self._create_url_slug(course_title)
                return f"https://dt4si.com/courses/{slug}"
            
            # Case Studies - use predefined URLs
            elif 'case-studies' in sheet_name or 'case study' in sheet_name:
                if len(values) >= 1:
                    case_study_title = str(values[0]).strip().lower()
                    # Clean up the title for matching
                    case_study_title = case_study_title.replace('- ', '').replace('(keyword:', '').split('(')[0].strip()
                    
                    # Find matching URL from predefined list
                    for key, url in self.case_study_urls.items():
                        if key in case_study_title or case_study_title in key:
                            return url
                    
                    # Fallback: create generic case study URL
                    slug = self._create_url_slug(case_study_title)
                    return f"https://dt4si.com/case-studies/{slug}"
            
            # Fallback for any unmatched items
            return "https://dt4si.com/"
            
        except Exception as e:
            print(f"Error generating URL: {e}")
            return "https://dt4si.com/"

    def search(self, query: str, k: int = 20, min_score: float = 0.30) -> tuple[list[dict], str]:
        # Always use 'all' categories - no intent detection
        detected_category = 'all'
        
        # Semantic search component
        query_vector = self.model.encode([query])
        # Use consistent multiplier for comprehensive search across all categories
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
        
        # Always show all categories with semantic-based stacking
        # No category filtering - show results from all categories
        filtered_results = results
        filtered_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Always reorganize by category groups while preserving semantic scores
        # Define category order and identification
        def get_result_category(result):
            sheet_name = result['metadata'].get('sheet', '').lower()
            values = result['metadata'].get('values', [])
            category_val = str(values[0]).lower() if values else ''
            
            # IMPORTANT: Prioritize sheet-based identification to avoid misclassification
            # Check case studies first since they can have misleading content keywords
            if ('case-studies' in sheet_name or 
                'case study' in sheet_name or
                sheet_name == 'case-studies'):
                return 'case-studies'
            # Then check other categories based on sheet name and content
            elif ('cleaned sheet' in sheet_name or 
                  'tool' in category_val or 
                  'ai tools' in category_val or
                  'software' in category_val.lower()):
                return 'tools'
            elif ('training program' in sheet_name or 
                  'training' in sheet_name or 
                  'course' in category_val or
                  'education' in category_val):
                return 'courses'
            elif ('service provider profiles' in sheet_name or 
                  'service' in sheet_name or 
                  'provider' in sheet_name or 
                  'provider' in category_val or 
                  'vendor' in category_val or
                  'company' in category_val):
                return 'service-providers'
            elif ('case-studies' in sheet_name or 
                  'case study' in sheet_name or
                  sheet_name == 'case-studies'):
                return 'case-studies'
            else:
                return 'other'
        
        # Group results by category
        tools_results = []
        courses_results = []
        service_providers_results = []
        case_studies_results = []
        other_results = []
        
        for result in filtered_results:
            category = get_result_category(result)
            if category == 'tools':
                tools_results.append(result)
            elif category == 'courses':
                courses_results.append(result)
            elif category == 'service-providers':
                service_providers_results.append(result)
            elif category == 'case-studies':
                case_studies_results.append(result)
            else:
                other_results.append(result)
        
        # Sort each category by score (highest first)
        tools_results.sort(key=lambda x: x['score'], reverse=True)
        courses_results.sort(key=lambda x: x['score'], reverse=True)
        service_providers_results.sort(key=lambda x: x['score'], reverse=True)
        case_studies_results.sort(key=lambda x: x['score'], reverse=True)
        other_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Create category groups with their highest scores for ranking
        category_groups = []
        if tools_results:
            category_groups.append(('tools', tools_results, tools_results[0]['score']))
        if courses_results:
            category_groups.append(('courses', courses_results, courses_results[0]['score']))
        if service_providers_results:
            category_groups.append(('service-providers', service_providers_results, service_providers_results[0]['score']))
        if case_studies_results:
            category_groups.append(('case-studies', case_studies_results, case_studies_results[0]['score']))
        if other_results:
            category_groups.append(('other', other_results, other_results[0]['score']))
        
        # Sort category groups by their highest score (highest first)
        category_groups.sort(key=lambda x: x[2], reverse=True)
        
        # Combine results in order of category ranking by highest score
        top_results = []
        for category_name, category_results, highest_score in category_groups:
            top_results.extend(category_results)
        
        # Take the top k results after stacking
        top_results = top_results[:k]
        
        return top_results, detected_category


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
                
                # Show search results grouped by semantic relevance
                st.info("🌐 **All Categories**: Results organized by semantic relevance")
                if results:
                    for i, res in enumerate(results, start=1):
                        header = ' | '.join(res['metadata'].get('values', []))
                        source_sheet = res['metadata'].get('sheet', 'Unknown')
                        
                        # Generate URL for this result
                        result_url = searcher._generate_result_url(res)
                        
                        # Determine category type and display info based on source
                        if "case-studies" in source_sheet.lower():
                            category_type = "CASE STUDY"
                            source_emoji = "📋"
                            # For case studies, use the clean title as header
                            case_study_title = res['metadata'].get('values', ['Unknown'])[0]
                            # Clean up the title display
                            clean_title = case_study_title.replace('- ', '').split('(')[0].strip()
                            display_header = clean_title
                        elif "tools" in source_sheet.lower() or "cleaned sheet" in source_sheet.lower():
                            category_type = "TOOL"
                            source_emoji = "🛠️"
                            # For tools, use the tool name (index 2)
                            values = res['metadata'].get('values', [])
                            display_header = values[2] if len(values) >= 3 else header
                        elif "training" in source_sheet.lower():
                            category_type = "COURSE"
                            source_emoji = "📚"
                            # For courses, use the course title (index 2)
                            values = res['metadata'].get('values', [])
                            display_header = values[2] if len(values) >= 3 else header
                        else:
                            category_type = "SERVICE PROVIDER"
                            source_emoji = "🏢"
                            # For service providers, use the provider name (index 0)
                            values = res['metadata'].get('values', [])
                            display_header = values[0] if len(values) >= 1 else header
                        
                        # Create clickable title with hyperlink
                        clickable_title = f"[{display_header}]({result_url})"
                        
                        # Handle case studies display
                        if "case-studies" in source_sheet.lower():
                            case_study_title = res['metadata'].get('values', ['Unknown'])[0]
                            industry = res['metadata'].get('industry', 'Unknown')
                            problem_type = res['metadata'].get('problem_type', 'Unknown')
                            word_count = res['metadata'].get('word_count', 0)
                            
                            with st.expander(f"{source_emoji} {category_type}: {display_header} (Score: {res['score']:.3f})"):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown("#### Case Study Details")
                                    st.markdown(f"**Title:** {clickable_title}")
                                    st.write(f"**Industry:** {industry}")
                                    st.write(f"**Problem Type:** {problem_type}")
                                
                                with col2:
                                    st.markdown("#### Relevance Scores")
                                    st.progress(res['score'])
                                    st.write(f"**Hybrid Score:** {res['score']:.3f}")
                                    st.write(f"🧠 Semantic: {res['semantic_score']:.3f} (70%)")
                                    st.write(f"🔍 TF-IDF: {res['tfidf_score']:.3f} (30%)")
                        else:
                            # Display for tools, courses, and service providers
                            with st.expander(f"{source_emoji} {category_type}: {display_header} (Score: {res['score']:.3f})"):
                                detail_col, score_col = st.columns([2, 1])
                                with detail_col:
                                    st.markdown("#### Details")
                                    st.markdown(f"**Title:** {clickable_title}")
                                    
                                    # Display remaining details based on category
                                    if "cleaned sheet" in source_sheet.lower():  # Tools
                                        values = res['metadata'].get('values', [])
                                        if len(values) >= 2:
                                            st.write(f"**Category:** {values[0]}")
                                            st.write(f"**Sub-Category:** {values[1]}")
                                    elif "service provider profiles" in source_sheet.lower():  # Service Providers
                                        # Only show the provider name as title, no additional details needed
                                        pass
                                    elif "training program" in source_sheet.lower():  # Courses
                                        values = res['metadata'].get('values', [])
                                        if len(values) >= 2:
                                            st.write(f"**Skill:** {values[0]}")
                                            st.write(f"**Topic:** {values[1]}")
                                    
                                    st.write(f"**Source:** {source_emoji} {source_sheet}")
                                with score_col:
                                    st.markdown("#### Relevance Scores")
                                    st.progress(res['score'])
                                    st.write(f"**Hybrid Score:** {res['score']:.3f}")
                                    st.write(f"🧠 Semantic: {res['semantic_score']:.3f} (70%)")
                                    st.write(f"🔍 TF-IDF: {res['tfidf_score']:.3f} (30%)")
                else:
                    st.info(f"No results found for your query. Try different search terms or be more specific.")

    with st.sidebar:
        st.markdown("### About")
        
        st.write(f"""
        🤖 **Hybrid AI Search** combining:
        - **70% Semantic Search**: Understanding context and meaning
        - **30% TF-IDF Keyword**: Exact keyword matching
        - **Smart Category Stacking**: Results organized by semantic relevance
        
        All categories are searched (checking commit) and results are dynamically stacked based on semantic similarity scores.
        """)
        
        st.markdown("### Debug Info")
        commit_hash = get_git_commit_hash()
        st.code(f"Commit: {commit_hash}")
        st.write(f"**Python Version:** {st.__version__}")
        
        st.markdown("### Search Tips")
        st.write("""
        - **Specific terms**: Use exact keywords for better TF-IDF matching
        - **Concepts**: Use descriptive phrases for better semantic matching
        - **Best results**: Combine both approaches in your query
        """)


if __name__ == "__main__":
    main()
    