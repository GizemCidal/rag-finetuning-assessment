import requests
import os
import re
from datasets import load_dataset
from .config import RAGConfig

"""
Data loading and preprocessing module.

Handles downloading books from Project Gutenberg, cleaning text (header/footer removal,
normalization), and loading QA pairs from NarrativeQA for evaluation.
"""

class DataLoader:
    """
    Handles downloading and processing of book data.
    """
    def __init__(self, config: RAGConfig):
        self.config = config

    def download_book(self):
        """
        Downloads the book text from Project Gutenberg.

        Checks if the file already exists locally; if not, downloads and cleans it.

        Returns:
            str: The raw text content of the book.
        """
        file_path = os.path.join(self.config.DATA_DIR, self.config.BOOK_FILENAME)
        if os.path.exists(file_path):
            print(f"Book already exists at {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        print(f"Downloading book from {self.config.BOOK_URL}...")
        try:
            # Gutenberg often redirects or requires User-Agent
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(self.config.BOOK_URL, headers=headers)
            response.raise_for_status()
            text = response.text
            
            # Clean Byte Order Mark if present
            if text.startswith('\ufeff'):
                text = text[1:]

            # Gutenberg Header/Footer Removal
            text = self._clean_gutenberg_text(text)
            
            # Advanced Cleaning (Unwrap & Normalize)
            text = self._normalize_text(text)
                
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
                
            print(f"Book saved to {file_path}")
            return text
        except Exception as e:
            print(f"Error downloading book: {e}")
            raise

    def _clean_gutenberg_text(self, text: str) -> str:
        """
        Removes Project Gutenberg headers and footers.

        Args:
            text (str): The raw text with Gutenberg markers.

        Returns:
            str: The cleaned text within the markers.
        """
        # Markers
        start_markers = ["*** START OF THE PROJECT GUTENBERG EBOOK", "*** START OF THIS PROJECT GUTENBERG EBOOK"]
        end_markers = ["*** END OF THE PROJECT GUTENBERG EBOOK", "*** END OF THIS PROJECT GUTENBERG EBOOK"]
        
        start_idx = 0
        end_idx = len(text)
        
        for marker in start_markers:
            idx = text.find(marker)
            if idx != -1:
                # Move past the marker line (approx 80 chars or newline)
                # Ensure we skip the marker line itself
                start_idx = text.find('\n', idx) + 1
                break
                
        for marker in end_markers:
            idx = text.find(marker)
            if idx != -1:
                end_idx = idx
                break
                
        if start_idx == 0 and end_idx == len(text):
            print("Warning: Gutenberg markers not found. Skipping strip.")
            
        return text[start_idx:end_idx].strip()

    def _normalize_text(self, text: str) -> str:
        """
        Normalizes whitespace and unwraps hard-wrapped lines common in Gutenberg texts.
        
        Converts double newlines to paragraph markers to preserve structure,
        removes single newlines to unwrap lines, and then restores paragraphs.

        Args:
            text (str): Input text.

        Returns:
            str: Normalized text.
        """
        # Protect Paragraphs: Convert double newlines to a special marker
        # Look for 2 or more newlines and replace with a marker
        text = re.sub(r'\n{2,}', ' [[PARAGRAPH]] ', text)
        
        # Unwrap Lines: Convert remaining single newlines to spaces
        # This fixes: "broken\nlines" -> "broken lines"
        text = text.replace('\n', ' ')
        
        # Restore Paragraphs: Convert marker back to double newlines
        text = text.replace(' [[PARAGRAPH]] ', '\n\n')
        
        # Collapse Whitespace: '  ' -> ' '
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()

    def load_qa_pairs(self):
        """
        Loads and filters QA pairs for the specific book from NarrativeQA.

        Returns:
            list: A list of dicts, each containing 'question', 'answer1', 'answer2', and 'doc_id'.
        """
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
