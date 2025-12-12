import requests
import os
from datasets import load_dataset
from .config import RAGConfig

class DataLoader:
    def __init__(self, config: RAGConfig):
        self.config = config

    def download_book(self):
        """Downloads the book text from Project Gutenberg."""
        file_path = os.path.join(self.config.DATA_DIR, self.config.BOOK_FILENAME)
        if os.path.exists(file_path):
            print(f"Book already exists at {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        print(f"Downloading book from {self.config.BOOK_URL}...")
        try:
            # Note: Gutenberg often redirects or requires User-Agent
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(self.config.BOOK_URL, headers=headers)
            response.raise_for_status()
            text = response.text
            
            # Basic cleaning: remove Byte Order Mark if present
            if text.startswith('\ufeff'):
                text = text[1:]
                
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
                
            print(f"Book saved to {file_path}")
            return text
        except Exception as e:
            print(f"Error downloading book: {e}")
            raise

    def load_qa_pairs(self):
        """Loads and filters QA pairs for the specific book from NarrativeQA."""
        # Using the exact ID logic found in debugging
        print(f"Loading NarrativeQA test split for ID {self.config.BOOK_ID}...")
        ds = load_dataset("narrativeqa", split="test", trust_remote_code=True)
        
        qa_pairs = []
        target_id_str = self.config.BOOK_ID
        
        for row in ds:
            doc = row['document']
            if doc['kind'] == 'gutenberg':
                url = doc.get('url', '')
                # Filter by ID in URL (e.g., .../1845.txt...)
                if target_id_str in url:
                    qa_pairs.append({
                        "question": row['question']['text'],
                        "answer1": row['answers'][0]['text'],
                        "answer2": row['answers'][1]['text'] if len(row['answers']) > 1 else "",
                        "doc_id": doc['id']
                    })
        
        print(f"Found {len(qa_pairs)} QA pairs for Book ID {target_id_str}.")
        return qa_pairs
