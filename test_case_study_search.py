#!/usr/bin/env python3
"""
Test case study specific searches
"""

import sys
sys.path.append('.')

from app_main import SemanticSearcher

def test_case_study_searches():
    """Test searches specifically targeting case studies"""
    print("📋 Testing Case Study Specific Searches...")
    
    try:
        searcher = SemanticSearcher()
        
        # Test case study specific queries
        test_queries = [
            "digital transformation case study",
            "Salesforce implementation nonprofit", 
            "AI drone monitoring agriculture",
            "mobile app development education",
            "supply chain optimization ERP"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results, detected_category = searcher.search(query, k=5, min_score=0.2)
            
            print(f"🤖 Detected Category: {detected_category}")
            
            case_study_results = [r for r in results if 'case-studies' in r['metadata'].get('sheet', '')]
            print(f"📋 Case Study Results: {len(case_study_results)}/{len(results)}")
            
            for i, result in enumerate(case_study_results[:3], 1):
                title = result['metadata'].get('values', ['Unknown'])[0]
                score = result['score']
                print(f"  {i}. {title[:70]}... (Score: {score:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing case study searches: {e}")
        return False

if __name__ == "__main__":
    test_case_study_searches()
