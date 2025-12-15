import os
import json
import pandas as pd
import torch
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rag.config import RAGConfig
from rag.data_loader import DataLoader
from rag.chunking import HierarchicalChunker
from rag.vector_db import VectorDBHandler
from rag.retriever import HierarchicalRetriever
from rag.generator import RAGGenerator
from rag.evaluator import Evaluator
"""
Full RAG Experiment Script.

Runs the complete pipeline:
1. Data loading.
2. Chunking and Indexing.
3. RAG Setup (Retriever, Generator).
4. Evaluation loop comparing Baseline vs. RAG.
"""

def main():
    """
    Main experiment function.
    """
    config = RAGConfig()
    
    # Data Loading
    print(">>> Stage 1: Data Loading")
    loader = DataLoader(config)
    book_text = loader.download_book()
    qa_pairs = loader.load_qa_pairs()
    
    # Limit QA pairs for quick testing/demo (remove limit for full run)
    # qa_pairs = qa_pairs[:10] 
    
    # Chunking & Indexing
    print(">>> Stage 2: Chunking & Indexing")
    chunker = HierarchicalChunker(
        parent_chunk_size=config.PARENT_CHUNK_SIZE,
        child_chunk_size=config.CHILD_CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP
    )
    
    chunks_data = chunker.chunk_data(book_text)
    
    # Initialize Vector DB
    vdb = VectorDBHandler(config)
    vdb.create_collection()
    
    # Check if already indexed (simple check: if collection count > 0)
    # Check index existence 
    # Checking if empty.
    collection_info = vdb.client.get_collection(vdb.collection_name)
    if collection_info.points_count == 0:
        print("Indexing data...")
        # Need embedding model for indexing
        # Use the retriever's encoder for consistency, even if instantiated temporarily
        # This ensures the same embedding model is used for indexing and retrieval.
        encoder = HierarchicalRetriever.get_encoder(config)
        child_texts = [c['text'] for c in chunks_data['children']]
        embeddings = encoder.encode(child_texts).tolist()
        
        vdb.upsert_chunks(chunks_data['children'], embeddings)
    else:
        print(f"Index already contains {collection_info.points_count} points. Skipping indexing.")

    # Setup Components
    print(">>> Stage 3: Setting up RAG Components")
    # Encoder needed for retriever
    encoder = HierarchicalRetriever.get_encoder(config)
    
    retriever = HierarchicalRetriever(config, vdb, chunks_data['parents'], embedding_model=encoder)
    generator = RAGGenerator(config)
    evaluator = Evaluator()
    
    # Evaluation Loop
    print(">>> Stage 4: Running Evaluation")
    results = []
    
    for qa in tqdm(qa_pairs, desc="Evaluating"):
        question = qa['question']
        ground_truth = qa['answer1']
        
        # A. Baseline (Zero-shot, No Context)
        # Provide empty context or instruction to answer from knowledge
        baseline_answer = generator.generate_answer(question, context="")
        baseline_metrics = evaluator.calculate_metrics(ground_truth, baseline_answer)
        
        # B. RAG
        context = retriever.retrieve_context(question)
        rag_answer = generator.generate_answer(question, context)
        rag_metrics = evaluator.calculate_metrics(ground_truth, rag_answer)
        
        results.append({
            "Question": question,
            "Ground_Truth": ground_truth,
            "Baseline_Answer": baseline_answer,
            "Baseline_BLEU": baseline_metrics['bleu4'],
            "Baseline_ROUGE": baseline_metrics['rouge_l'],
            "RAG_Answer": rag_answer,
            "RAG_Context": context[:200] + "...", # Truncate for display
            "RAG_BLEU": rag_metrics['bleu4'],
            "RAG_ROUGE": rag_metrics['rouge_l']
        })

    # Reporting
    df = pd.DataFrame(results)
    
    print("\n>>> Results Summary:")
    print(df[['Baseline_BLEU', 'RAG_BLEU', 'Baseline_ROUGE', 'RAG_ROUGE']].mean())
    
    output_path = "rag_evaluation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to {output_path}")

if __name__ == "__main__":
    main()
