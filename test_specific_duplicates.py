#!/usr/bin/env python3
"""
Test script to verify deduplication works with known duplicate tools.
"""

from app_main import SemanticSearcher

def test_specific_duplicates():
    try:
        searcher = SemanticSearcher()
        
        # Test searches for known duplicate tools
        test_queries = [
            "Runway",
            "Zapier", 
            "Power BI",
            "Tableau",
            "Hubspot",
            "Notion"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Testing query: '{query}'")
            print('='*60)
            
            results, category = searcher.search(query, k=10, min_score=0.30)
            
            print(f"Detected category: {category}")
            print(f"Results found: {len(results)}")
            
            # Check for the specific tool in results
            tool_found = False
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                values = metadata.get('values', [])
                sheet = metadata.get('sheet', '')
                
                if 'cleaned sheet' in sheet.lower() and len(values) >= 3:
                    tool_name = values[2]
                    if query.lower() in tool_name.lower() or tool_name.lower() in query.lower():
                        print(f"\nResult {i} - MATCH FOUND:")
                        print(f"  Tool Name: {tool_name}")
                        print(f"  Category: {values[0]}")
                        print(f"  Sub-Category: {values[1]}")
                        print(f"  Score: {result['score']:.4f}")
                        tool_found = True
                        break
            
            if not tool_found:
                print(f"\nNo exact match found for '{query}' in top results")
                # Show top result anyway
                if results:
                    result = results[0]
                    metadata = result['metadata']
                    values = metadata.get('values', [])
                    if len(values) >= 3:
                        print(f"Top result: {values[2]} (Score: {result['score']:.4f})")
                        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_specific_duplicates()
