import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from rag.config import RAGConfig
from rag.data_loader import DataLoader
from rag.chunking import HierarchicalChunker
from rag.vector_db import VectorDBHandler
from rag.retriever import HierarchicalRetriever
from rag.generator import RAGGenerator
from rag.evaluator import Evaluator

def main():
    config = RAGConfig()
    
    # 1. Data Loading
    print(">>> Stage 1: Data Loading")
    loader = DataLoader(config)
    book_text = loader.download_book()
    qa_pairs = loader.load_qa_pairs()
    
    # Limit QA pairs for quick testing/demo (remove limit for full run)
    # qa_pairs = qa_pairs[:10] 
    
    # 2. Chunking & Indexing
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
    # For this task, we can just re-index or check count. 
    # Let's assume re-indexing for safety or checking if empty.
    collection_info = vdb.client.get_collection(vdb.collection_name)
    if collection_info.points_count == 0:
        print("Indexing data...")
        # Need embedding model for indexing
        # We can reuse the retriever's encoder logic or instantiate here.
        # Let's instantiate a temporary retriever to access encoder or standalone.
        # Better: use Retriever class just for encoding helper if possible or separate.
        # We'll just instantiate SentenceTransformer here to keep it simple.
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        
        child_texts = [c['text'] for c in chunks_data['children']]
        embeddings = encoder.encode(child_texts).tolist()
        
        vdb.upsert_chunks(chunks_data['children'], embeddings)
    else:
        print(f"Index already contains {collection_info.points_count} points. Skipping indexing.")

    # 3. Setup Components
    print(">>> Stage 3: Setting up RAG Components")
    retriever = HierarchicalRetriever(config, vdb, chunks_data['parents'])
    generator = RAGGenerator(config)
    evaluator = Evaluator()
    
    # 4. Evaluation Loop
    print(">>> Stage 4: Running Evaluation")
    results = []
    
    for qa in tqdm(qa_pairs, desc="Evaluating"):
        question = qa['question']
        ground_truth = qa['answer1']
        
        # A. Baseline (Zero-shot, No Context)
        # Note: We provide empty context or instruction to answer from knowledge.
        baseline_answer = generator.generate_answer(question, context="")
        baseline_metrics = evaluator.calculate_metrics(ground_truth, baseline_answer)
        
        # B. RAG
        context = retriever.retrieve(question)
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

    # 5. Reporting
    df = pd.DataFrame(results)
    
    print("\n>>> Results Summary:")
    print(df[['Baseline_BLEU', 'RAG_BLEU', 'Baseline_ROUGE', 'RAG_ROUGE']].mean())
    
    output_path = "rag_evaluation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to {output_path}")

if __name__ == "__main__":
    main()
