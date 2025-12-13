```python
from datasets import load_dataset
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    print("Loading dataset (streaming)...")
    ds = load_dataset("narrativeqa", split="test", trust_remote_code=True, streaming=True)
    
    print("Scanning...")
    count = 0
    target_book = "Dracula"
    
    # We will check the first 2000 items to limit time if streaming
    limit = 2000
    for i, row in enumerate(ds):
        if i > limit:
            break
            
        doc = row['document']
        if doc['kind'] == 'gutenberg':
            title_info = doc.get('start', '')
            if target_book in title_info:
                count += 1
                if count == 1:
                    print(f"FOUND BOOK: {title_info[:50]}...")
                    print(f"ID: {doc['id']}")

    print(f"Total entries found for {target_book} in scanned range: {count}")

except Exception as e:
    print(f"Error: {e}")
