import sys
import os
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rag.config import RAGConfig
from rag.generator import RAGGenerator
from rag.evaluator import Evaluator
from rag.data_loader import DataLoader
"""
Baseline benchmark script.

Evaluates the performance of the model using zero-shot generation (RAG free) on the test dataset.
"""

def run_baseline_benchmark():
    """
    Runs the baseline generation (no context) on the QA pairs and calculates metrics.
    """
    config = RAGConfig()
    generator = RAGGenerator(config) # Model only
    evaluator = Evaluator()
    
    print("--- Loading QA Data (Baseline) ---")
    try:
        loader = DataLoader(config)
        qa_pairs = loader.load_qa_pairs()
        print(f"Loaded {len(qa_pairs)} QA pairs from NarrativeQA.")
    except Exception as e:
        print(f"Warning: Could not load full dataset ({e}). Using sample set for validation.")
        qa_pairs = [
            {"question": "Who are Zuleika's most prominent suitors?", "answer1": "The Duke of Dorset and Noaks."},
            {"question": "Why does Zuleika reject the Duke?", "answer1": "Because he is not 'hopeless' enough; she only loves those she cannot have."},
            {"question": "Who is the first person Zuleika falls in love with?", "answer1": "She has never loved anyone."},
            # Add more if needed or use these representatives
        ]
    
    results = []
    print("\n--- Running Baseline (No-Context) Evaluation ---")
    
    for i, item in enumerate(tqdm(qa_pairs)):
        question = item['question']
        reference = item['answer1']
        
        # Zero-shot generation (No Context)
        answer = generator.generate_answer(question, context="", do_sample=False)
        
        scores = evaluator.calculate_metrics(reference, answer)
        
        results.append({
            "Question": question,
            "Target": reference,
            "Generated": answer,
            "BLEU": scores['bleu4'],
            "ROUGE": scores['rouge_l']
        })
        
    # Calculate Averages
    df = pd.DataFrame(results)
    avg_bleu = df['BLEU'].mean()
    avg_rouge = df['ROUGE'].mean()
    
    print("\n--- Baseline Results ---")
    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average ROUGE: {avg_rouge:.4f}")
    
    # Save to CSV
    output_path = os.path.join(config.DATA_DIR, 'baseline_results.csv')
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to {output_path}")

if __name__ == "__main__":
    run_baseline_benchmark()
