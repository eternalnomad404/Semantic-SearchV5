#!/usr/bin/env python3
"""
Test script to verify that tool name deduplication is working correctly.
"""

from app_main import SemanticSearcher
import json

def test_deduplication():
    try:
        # Initialize the searcher
        searcher = SemanticSearcher()
        
        # Test with a search that might return duplicate tool names
        query = "design tool"
        results, category = searcher.search(query, k=50, min_score=0.20)
        
        print(f"Search query: '{query}'")
        print(f"Detected category: {category}")
        print(f"Total results: {len(results)}")
        print("\n" + "="*80)
        
        # Track tool names from tools sheet to check for duplicates
        tool_names_found = {}
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            values = metadata.get('values', [])
            sheet = metadata.get('sheet', '')
            score = result['score']
            
            print(f"\nResult {i}:")
            print(f"  Sheet: {sheet}")
            print(f"  Score: {score:.4f}")
            
            if len(values) >= 3:
                category = values[0]
                sub_category = values[1] 
                tool_name = values[2]
                print(f"  Category: {category}")
                print(f"  Sub-Category: {sub_category}")
                print(f"  Tool Name: {tool_name}")
                
                # Check for duplicates in tools sheet
                if 'cleaned sheet' in sheet.lower():
                    tool_key = tool_name.strip().lower()
                    if tool_key in tool_names_found:
                        print(f"  ❌ DUPLICATE FOUND! Previous score: {tool_names_found[tool_key]:.4f}")
                    else:
                        tool_names_found[tool_key] = score
                        print(f"  ✅ First occurrence of this tool")
            else:
                print(f"  Values: {values}")
        
        # Summary
        print(f"\n" + "="*80)
        print("DEDUPLICATION SUMMARY:")
        tools_count = sum(1 for r in results if 'cleaned sheet' in r['metadata'].get('sheet', '').lower())
        print(f"Total tool results: {tools_count}")
        print(f"Unique tool names: {len(tool_names_found)}")
        
        if tools_count == len(tool_names_found):
            print("✅ SUCCESS: No duplicate tool names found!")
        else:
            print("❌ FAILURE: Duplicate tool names detected!")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_deduplication()
