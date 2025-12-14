import pandas as pd
import sys
import os
import json

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.config import RAGConfig
from rag.vector_db import VectorDBHandler
from rag.retriever import HierarchicalRetriever
from rag.generator import RAGGenerator
from rag.evaluator import Evaluator
from rag.data_loader import DataLoader
from rag.chunking import HierarchicalChunker

def run_comparison():
    print("--- Initializing RAG Components for Comparison ---")
    config = RAGConfig()
    vdb = VectorDBHandler(config)
    
    # Try to load parents map from disk
    parents_map_path = os.path.join(config.DATA_DIR, 'parents_map.json')
    
    # SINGLETON EMBEDDING MODEL (Moved up for Indexing)
    from sentence_transformers import SentenceTransformer
    print(f"Loading Singleton Embedding Model: {config.EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    
    if os.path.exists(parents_map_path):
        print(f"Loading parents map from {parents_map_path}...")
        with open(parents_map_path, 'r') as f:
            parents_map = json.load(f)
    else:
        print("Parents map not found on disk. Re-creating from book text...")
        loader = DataLoader(config)
        book_text = loader.download_book()
        chunker = HierarchicalChunker(
            parent_chunk_size=config.PARENT_CHUNK_SIZE,
            child_chunk_size=config.CHILD_CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
        chunks = chunker.chunk_data(book_text)
        # parents is already a dict of {id: text}
        parents_map = chunks['parents']
        
        parents_map = chunks['parents']
        
        # Save for future
        with open(parents_map_path, 'w') as f:
            json.dump(parents_map, f)
            print("Saved generated parents map to disk.")

        # INDEXING (New Logic)
        print("Indexing chunks into Qdrant...")
        vdb.create_collection()
        vdb.index_chunks(chunks, embedding_model)
        print("Indexing complete.")
    
    # Load QA Pairs for testing
    # Bypass datasets library due to potential environment/version conflicts in script mode
    # Using known questions from the dataset (Book ID 1845)
    qa_pairs = [
        {
            "question": "Who are Zuleika's most prominent suitors?",
            "answer1": "The Duke of Dorset and Noaks."
        },
        {
            "question": "Why does Zuleika reject the Duke?",
            "answer1": "Because he is not 'hopeless' enough; she only loves those she cannot have."
        },
        {
            "question": "Who is the first person Zuleika falls in love with?",
            "answer1": "She has never loved anyone, but she thinks she might love the Duke if he didn't love her."
        }
    ]
    
    # QA Pairs loaded above
    
    # SINGLETON EMBEDDING MODEL
    # Already loaded above
    # embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    
    retriever = HierarchicalRetriever(config, vdb, parents_map, embedding_model=embedding_model)
    generator = RAGGenerator(config)
    evaluator = Evaluator()
    
    print("\n--- Running Comparison (Top 3 Questions) ---")
    test_pairs = qa_pairs[:3]
    results = []
    
    for qa in test_pairs:
        question = qa['question']
        reference = qa['answer1']
        print(f"\nQ: {question}")
        
        # 1. Base Retrieval
        print("  Running Base Retrieval...")
        ctx_base = retriever.retrieve_context(question, top_k=config.TOP_K, use_reranker=False)
        ans_base = generator.generate_answer(question, ctx_base, do_sample=False)
        
        # 2. Reranked Retrieval
        print("  Running Reranked Retrieval...")
        ctx_rerank = retriever.retrieve_context(question, top_k=config.TOP_K, use_reranker=True)
        # Note: We can also check deterministic vs creative. For Eval, prefer False.
        ans_rerank = generator.generate_answer(question, ctx_rerank, do_sample=False)
        
        # Eval
        score_base = evaluator.evaluate(ans_base, reference)
        score_rerank = evaluator.evaluate(ans_rerank, reference)
        
        results.append({
            "Question": question[:50] + "...",
            "Base ROUGE": round(score_base['rouge'], 4),
            "Rerank ROUGE": round(score_rerank['rouge'], 4),
            "Improvement": round(score_rerank['rouge'] - score_base['rouge'], 4)
        })
        
    df = pd.DataFrame(results)
    print("\n--- Results Summary ---")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    run_comparison()
