"""
Case Studies Document Processor
Extracts and processes case studies from Word document for semantic search
"""

import os
import re
from docx import Document
import json
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_case_studies_from_docx(docx_path: str) -> list[dict]:
    """
    Extract case studies from Word document using page break detection.
    Returns list of case study dictionaries with metadata.
    """
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"Document not found: {docx_path}")
    
    print(f"📄 Processing document: {docx_path}")
    doc = Document(docx_path)
    
    case_studies = []
    current_case_study = []
    case_study_count = 0
    
    # Extract all text paragraphs
    all_text = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:  # Only add non-empty paragraphs
            all_text.append(text)
    
    # Join all text and split by common case study indicators
    full_text = '\n'.join(all_text)
    
    # Try multiple splitting strategies
    case_study_patterns = [
        r'Case Study \d+',
        r'CASE STUDY \d+',
        r'Case \d+:',
        r'Study \d+:',
        r'\n\d+\.\s+[A-Z]',  # Pattern like "1. TITLE"
        r'\n[A-Z][A-Z\s]+\n'  # All caps titles
    ]
    
    # Find the best splitting pattern
    best_splits = []
    best_pattern = None
    
    for pattern in case_study_patterns:
        splits = re.split(pattern, full_text, flags=re.IGNORECASE)
        if len(splits) >= 10:  # We expect 11 case studies, so 10+ splits is good
            best_splits = splits
            best_pattern = pattern
            break
    
    # If pattern-based splitting works
    if best_splits and len(best_splits) >= 10:
        print(f"✅ Found {len(best_splits)} sections using pattern: {best_pattern}")
        
        for i, section in enumerate(best_splits[1:], 1):  # Skip first empty section
            if len(section.strip()) > 100:  # Only process substantial sections
                case_study_count += 1
                
                # Extract title (first meaningful line)
                lines = [line.strip() for line in section.split('\n') if line.strip()]
                title = lines[0] if lines else f"Case Study {case_study_count}"
                
                # Clean title
                title = re.sub(r'^[\d\.\s:]+', '', title).strip()
                if len(title) > 100:
                    title = title[:100] + "..."
                
                # Extract key information
                text_content = section.strip()
                
                # Try to identify industry/domain from content
                industry = extract_industry_from_text(text_content)
                problem_type = extract_problem_type_from_text(text_content)
                
                case_study_data = {
                    'id': case_study_count,
                    'title': title,
                    'industry': industry,
                    'problem_type': problem_type,
                    'full_text': text_content,
                    'summary': create_summary(text_content),
                    'word_count': len(text_content.split())
                }
                
                case_studies.append(case_study_data)
                
                if case_study_count >= 11:  # Stop after 11 case studies
                    break
    
    # Fallback: Split by length if pattern-based splitting fails
    if len(case_studies) < 5:
        print("⚠️ Pattern-based splitting didn't work well. Using length-based splitting...")
        case_studies = split_by_length(full_text)
    
    print(f"✅ Successfully extracted {len(case_studies)} case studies")
    return case_studies

def split_by_length(full_text: str) -> list[dict]:
    """Fallback method: Split document into roughly equal parts."""
    words = full_text.split()
    total_words = len(words)
    words_per_case_study = total_words // 11
    
    case_studies = []
    
    for i in range(11):
        start_idx = i * words_per_case_study
        end_idx = (i + 1) * words_per_case_study if i < 10 else total_words
        
        section_text = ' '.join(words[start_idx:end_idx])
        
        # Extract title from first sentence
        sentences = section_text.split('.')
        title = sentences[0].strip()[:100] if sentences else f"Case Study {i+1}"
        
        case_study_data = {
            'id': i + 1,
            'title': title,
            'industry': extract_industry_from_text(section_text),
            'problem_type': extract_problem_type_from_text(section_text),
            'full_text': section_text,
            'summary': create_summary(section_text),
            'word_count': len(section_text.split())
        }
        
        case_studies.append(case_study_data)
    
    return case_studies

def extract_industry_from_text(text: str) -> str:
    """Extract industry/domain from case study text."""
    industry_keywords = {
        'Education': ['education', 'school', 'university', 'student', 'learning', 'academic', 'teach', 'curriculum', 'classroom', 'edu-leader', 'fellowship', 'training', 'skill', 'course', 'balwadi', 'preschool', 'child', 'enrollment', 'dropout'],
        'Healthcare': ['health', 'medical', 'hospital', 'patient', 'clinical', 'healthcare', 'nutrition', 'wellness', 'maternal', 'child health'],
        'Finance': ['bank', 'financial', 'investment', 'loan', 'credit', 'finance', 'payment', 'micro-finance'],
        'Technology': ['software', 'tech', 'IT', 'digital', 'system', 'platform', 'automation', 'AI', 'data'],
        'Agriculture': ['farm', 'agriculture', 'crop', 'forest', 'agroforestry', 'rural', 'land', 'soil'],
        'Non-Profit': ['non-profit', 'charity', 'foundation', 'volunteer', 'community', 'social', 'impact', 'ngo'],
        'Manufacturing': ['production', 'factory', 'manufacturing', 'assembly', 'industrial'],
        'Retail': ['retail', 'store', 'customer', 'sales', 'shopping', 'commerce'],
        'Government': ['government', 'public', 'municipal', 'federal', 'state', 'agency']
    }
    
    text_lower = text.lower()
    
    # Score each industry based on keyword matches
    industry_scores = {}
    for industry, keywords in industry_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            industry_scores[industry] = score
    
    # Return the industry with the highest score
    if industry_scores:
        return max(industry_scores, key=industry_scores.get)
    
    return 'General'

def extract_problem_type_from_text(text: str) -> str:
    """Extract problem type from case study text."""
    problem_keywords = {
        'Digital Transformation': ['digital', 'automation', 'technology', 'transform', 'modernize', 'platform', 'system', 'tech integration'],
        'Operational Excellence': ['operational', 'operations', 'process', 'efficiency', 'workflow', 'optimization', 'streamline', 'scale'],
        'Education Innovation': ['education', 'learning', 'teaching', 'curriculum', 'student', 'academic', 'school'],
        'Impact Measurement': ['impact', 'measurement', 'monitoring', 'evaluation', 'assessment', 'tracking', 'metrics'],
        'Leadership Development': ['leadership', 'development', 'mentoring', 'coaching', 'training', 'capacity building'],
        'Supply Chain Optimization': ['supply chain', 'logistics', 'distribution', 'procurement', 'inventory'],
        'Cost Reduction': ['cost', 'budget', 'savings', 'reduction', 'expense', 'financial'],
        'Quality Enhancement': ['quality', 'improvement', 'standards', 'excellence', 'better'],
        'Customer Experience': ['customer', 'service', 'satisfaction', 'experience', 'client', 'user'],
        'Risk Management': ['risk', 'compliance', 'security', 'safety', 'mitigation'],
        'Growth Strategy': ['growth', 'expansion', 'strategy', 'development', 'scale', 'scaling']
    }
    
    text_lower = text.lower()
    
    # Score each problem type based on keyword matches
    problem_scores = {}
    for problem_type, keywords in problem_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            problem_scores[problem_type] = score
    
    # Return the problem type with the highest score
    if problem_scores:
        return max(problem_scores, key=problem_scores.get)
    
    return 'Business Challenge'

def create_summary(text: str, max_chars: int = 300) -> str:
    """Create a summary of the case study using GROQ AI."""
    try:
        # Initialize GROQ client
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            print("⚠️ GROQ_API_KEY not found, using fallback summary")
            return create_fallback_summary(text, max_chars)
        
        groq_client = Groq(api_key=groq_api_key)
        
        # Create focused prompt for keyword-rich summary
        prompt = f"""
Create a concise, keyword-rich summary of this digital transformation case study.

Requirements:
- Exactly 200-250 words
- Include specific technologies, platforms, tools (e.g., Salesforce, Microsoft Dynamics, AI tools)
- Include key metrics and outcomes (percentages, numbers)
- Include industry context and problem types
- Focus on technical implementation and measurable results
- Use searchable keywords someone would look for

Case Study:
{text[:2500]}

Generate a focused summary with important technical keywords:"""

        completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=350
        )
        
        summary = completion.choices[0].message.content.strip()
        
        if not summary or len(summary) < 50:
            print("⚠️ GROQ returned insufficient summary, using fallback")
            return create_fallback_summary(text, max_chars)
        
        # Ensure reasonable length
        words = summary.split()
        if len(words) > 300:
            summary = ' '.join(words[:300])
        
        print(f"✅ GROQ summary generated: {len(words)} words")
        return summary
        
    except Exception as e:
        print(f"⚠️ Error with GROQ summary generation: {e}")
        return create_fallback_summary(text, max_chars)

def create_fallback_summary(text: str, max_chars: int = 300) -> str:
    """Fallback method: Create a summary of the case study."""
    # Take first few sentences up to max_chars
    sentences = text.split('.')
    summary = ""
    
    for sentence in sentences:
        if len(summary + sentence) < max_chars:
            summary += sentence + ". "
        else:
            break
    
    return summary.strip()

def save_case_studies_metadata(case_studies: list[dict], output_path: str = "data/case_studies_metadata.json"):
    """Save case studies metadata to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(case_studies, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved case studies metadata to {output_path}")

def main():
    """Main function to process case studies."""
    docx_path = "data/Case-Studies.docx"
    
    try:
        case_studies = extract_case_studies_from_docx(docx_path)
        
        # Save metadata
        save_case_studies_metadata(case_studies)
        
        # Print summary
        print("\n📊 Case Studies Summary:")
        print("-" * 50)
        for cs in case_studies:
            print(f"ID: {cs['id']}")
            print(f"Title: {cs['title']}")
            print(f"Industry: {cs['industry']}")
            print(f"Problem Type: {cs['problem_type']}")
            print(f"Word Count: {cs['word_count']}")
            print(f"Summary: {cs['summary'][:100]}...")
            print("-" * 50)
        
        return case_studies
        
    except Exception as e:
        print(f"❌ Error processing case studies: {e}")
        return []

if __name__ == "__main__":
    main()
