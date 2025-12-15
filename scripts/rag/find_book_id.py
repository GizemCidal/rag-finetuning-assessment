from datasets import load_dataset

print("Loading NarrativeQA (test split) to find Zuleika Dobson...")
ds = load_dataset("narrativeqa", split="test", trust_remote_code=True)

found = False
for row in ds:
    doc = row['document']
    # Check if title mentions Zuleika
    # The title might be in doc['text'] or inferred? 
    # Actually NarrativeQA 'document' content usually has a 'title' field or we check the start of text?
    # Let's check the URL or simply print items where the question contains "Zuleika"
    
    if "Zuleika" in row['question']['text']:
        print(f"--- Found Entry ---")
        print(f"Question: {row['question']['text']}")
        print(f"Doc Kind: {doc['kind']}")
        print(f"Doc URL: {doc.get('url', 'N/A')}")
        print(f"Doc ID: {doc.get('id', 'N/A')}")
        found = True
        break

if not found:
    print("Could not find Zuleika in the test split. Checking validation split...")
    ds_val = load_dataset("narrativeqa", split="validation", trust_remote_code=True)
    for row in ds_val:
        if "Zuleika" in row['question']['text']:
            doc = row['document']
            print(f"--- Found Entry (Validation) ---")
            print(f"Question: {row['question']['text']}")
            print(f"Doc Kind: {doc['kind']}")
            print(f"Doc URL: {doc.get('url', 'N/A')}")
            print(f"Doc ID: {doc.get('id', 'N/A')}")
            found = True
            break
