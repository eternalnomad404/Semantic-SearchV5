#!/usr/bin/env python3
"""
Test script to check for potential duplicates in the data.
"""

import json

def analyze_duplicates():
    try:
        # Load metadata to check for actual duplicates
        with open("vectorstore/metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        tool_names = {}
        total_tools = 0
        
        for entry in metadata:
            sheet = entry.get('sheet', '')
            values = entry.get('values', [])
            
            if 'cleaned sheet' in sheet.lower() and len(values) >= 3:
                total_tools += 1
                tool_name = str(values[2]).strip().lower()
                
                if tool_name in tool_names:
                    tool_names[tool_name].append({
                        'category': values[0],
                        'sub_category': values[1],
                        'name': values[2]
                    })
                else:
                    tool_names[tool_name] = [{
                        'category': values[0],
                        'sub_category': values[1], 
                        'name': values[2]
                    }]
        
        # Find duplicates
        duplicates = {name: entries for name, entries in tool_names.items() if len(entries) > 1}
        
        print(f"Total tools in dataset: {total_tools}")
        print(f"Unique tool names: {len(tool_names)}")
        print(f"Duplicate tool names: {len(duplicates)}")
        
        if duplicates:
            print("\nDUPLICATE TOOL NAMES FOUND:")
            print("="*60)
            for tool_name, entries in duplicates.items():
                print(f"\nTool Name: '{entries[0]['name']}'")
                print(f"Appears {len(entries)} times:")
                for i, entry in enumerate(entries, 1):
                    print(f"  {i}. Category: {entry['category']}, Sub-Category: {entry['sub_category']}")
        else:
            print("\n✅ No duplicate tool names found in the dataset!")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_duplicates()
