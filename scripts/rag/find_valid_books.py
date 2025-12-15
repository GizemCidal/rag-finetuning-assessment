from datasets import load_dataset
import collections

# Load NarrativeQA (titles only to be fast)
print("Loading NarrativeQA dataset...")
ds = load_dataset("narrativeqa", split="test", trust_remote_code=True)

# Count QA pairs per document kind
book_stats = collections.defaultdict(int)
book_titles = {}

print("\nScanning for Project Gutenberg books in NarrativeQA...")
for row in ds:
    doc = row['document']
    # Filter only for Gutenberg books
    if doc['kind'] == 'gutenberg':
        doc_id = doc['id']
        title = doc['start'] # NarrativeQA often puts title/metadata in start (or use keys if available)
        # Better extraction: usually the text is in 'text', but we want metadata.
        # Actually NarrativeQA has a 'document' dict with 'kind', 'url', 'file_size', etc.
        
        # Count questions per document ID
        book_stats[doc_id] += 1
        
        # Try to find a readable title if possible, or use the ID/URL
        book_titles[doc_id] = doc.get('url', 'No URL')

# Sort by number of questions
sorted_books = sorted(book_stats.items(), key=lambda x: x[1], reverse=True)

print(f"\nFound {len(sorted_books)} Gutenberg books in the Test split.")
print("Top 10 books with most questions:")
for doc_id, count in sorted_books[:10]:
    print(f"ID: {doc_id} | Questions: {count} | URL: {book_titles[doc_id]}")

