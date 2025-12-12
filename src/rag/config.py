import os

class RAGConfig:
    # Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    QDRANT_PATH = os.path.join(DATA_DIR, 'qdrant_db')
    
    # Text Data
    # Book: The School for Scandal by Richard Brinsley Sheridan
    BOOK_ID = "1845" 
    BOOK_URL = "https://www.gutenberg.org/files/1845/1845-0.txt"
    BOOK_FILENAME = "school_for_scandal.txt"
    
    # Models
    LLM_MODEL_NAME = "google/gemma-3-1b-it"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Lightweight, good for Colab
    
    # Chunking
    PARENT_CHUNK_SIZE = 1000
    CHILD_CHUNK_SIZE = 250
    CHUNK_OVERLAP = 100
    
    # Retrieval
    TOP_K = 5
    
    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
