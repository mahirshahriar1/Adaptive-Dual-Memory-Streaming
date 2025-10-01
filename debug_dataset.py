#!/usr/bin/env python3
"""
Debug script to inspect the wikitext dataset structure
"""
from datasets import load_dataset

print("=" * 60)
print("WIKITEXT DATASET DEBUG")
print("=" * 60)

# Try different variants
for task in ["wikitext-103-raw-v1", "wikitext-2-raw-v1", "wikitext-103-v1", "wikitext-2-v1"]:
    print(f"\n--- Testing: {task} ---")
    try:
        data = load_dataset("wikitext", task)
        print(f"✓ Loaded successfully")
        print(f"  Available splits: {list(data.keys())}")
        
        for split_name in data.keys():
            split_data = data[split_name]
            print(f"\n  Split '{split_name}':")
            print(f"    Size: {len(split_data)}")
            print(f"    Fields: {split_data.column_names}")
            
            # Show first non-empty item
            for i in range(min(5, len(split_data))):
                item = split_data[i]
                text = item.get('text', '')
                if text and text.strip():
                    print(f"    Sample {i}: {len(text)} chars, starts with: {text[:100]!r}")
                    break
            else:
                print(f"    WARNING: First 5 items are empty!")
                
    except Exception as e:
        print(f"✗ Failed: {e}")

print("\n" + "=" * 60)
print("Testing with eval_adms_vs_streaming.py parameters...")
print("=" * 60)

# Test exact parameters from script
try:
    data = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    print(f"\n✓ Dataset loaded: {len(data)} examples")
    print(f"  Fields: {data.column_names}")
    
    # Check first few items
    non_empty = 0
    for i in range(min(10, len(data))):
        item = data[i]
        text = item.get('text', '')
        if text and text.strip():
            non_empty += 1
            if non_empty == 1:
                print(f"\n  First non-empty example (index {i}):")
                print(f"    Length: {len(text)} chars")
                print(f"    Preview: {text[:200]!r}")
    
    print(f"\n  Non-empty in first 10: {non_empty}")
    
except Exception as e:
    print(f"\n✗ Failed to load test split: {e}")
