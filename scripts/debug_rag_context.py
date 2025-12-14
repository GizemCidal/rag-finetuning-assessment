import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.config import RAGConfig
from rag.vector_db import VectorDBHandler
from rag.retriever import HierarchicalRetriever

def debug_context():
    config = RAGConfig()
    vdb = VectorDBHandler(config)
    
    # Load parents map
    with open(os.path.join(config.DATA_DIR, 'parents_map.json'), 'r') as f:
        parents_map = json.load(f)

    # Question to test
    question = "Why does Zuleika reject the Duke?"
    print(f"--- Debugging Context for Question: '{question}' ---")
    
    # Singleton Model
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    retriever = HierarchicalRetriever(config, vdb, parents_map, embedding_model)
    
    # Retrieve with Reranker
    context = retriever.retrieve_context(question, top_k=config.TOP_K, use_reranker=True)
    
    print("\n--- RETRIEVED CONTEXT (Top K) ---")
    print(context[:2000] + "... (truncated)")  # Print first 2000 chars

    print("\n--- ANALYSIS ---")
    print("Does the context contain the word 'love'? ", "love" in context.lower())
    print("Does the context contain the word 'suicide'? ", "suicide" in context.lower())
    print("Does the context contain the word 'reject'? ", "reject" in context.lower() or "refuse" in context.lower())

if __name__ == "__main__":
    debug_context()
