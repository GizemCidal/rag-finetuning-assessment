import os

class RAGConfig:
    # Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'rag', 'data')
    QDRANT_PATH = os.path.join(DATA_DIR, 'qdrant_db')
    
    # Text Data
    # Text Data
    # Book: Zuleika Dobson by Max Beerbohm
    BOOK_ID = "1845" 
    BOOK_URL = "https://www.gutenberg.org/files/1845/1845-0.txt"
    BOOK_FILENAME = "zuleika_dobson.txt"
    COLLECTION_NAME = f"gutenberg_{BOOK_ID}_children"
    
    # Models
    LLM_MODEL_NAME = "google/gemma-3-1b-it"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Lightweight, good for Colab
    EMBEDDING_DIM = 384 # Explicitly correctly set for MiniLM-L6-v2, though dynamic check is better. 
    # But since we use config for DB creation before loading model sometimes, keeping it or inferring later.
    # User requested explicit dynamic check, but config is static. I'll allow DB to use this or dynamic.
    
    # Chunking
    CHUNK_UNIT = "chars"
    PARENT_CHUNK_SIZE = 2000
    CHILD_CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    
    # Retrieval
    TOP_K = 10
    
    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
