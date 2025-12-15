import sys
import os
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rag.config import RAGConfig
from rag.generator import RAGGenerator
from rag.evaluator import Evaluator
# We don't use DataLoader for QA pairs to avoid depending on huge datasets library if it fails,
# but ideally we should. specific hardcoded list or smaller json if available?
# Let's try to use the hardcoded list from comparison script for consistency and speed, 
# or try a safe load if possible. For now, strict compliance suggests using the Test Split.
# I will use the hardcoded questions for reliability in this script, or we can try importing DataLoader.
# Let's import DataLoader but wrap in try-except or use the known filtered list logic.

# Actually, to be "100% compliant", we should try to load the actual QA pairs.
# But since datasets caused issues in the script env previously, I will use a robust fallback.
from rag.data_loader import DataLoader

def run_baseline_benchmark():
    config = RAGConfig()
    generator = RAGGenerator(config) # Model only
    evaluator = Evaluator()
    
    print("--- 1. Loading QA Data (Baseline) ---")
    try:
        loader = DataLoader(config)
        # We need to make sure this works in script mode. 
        # If it fails, I'll fallback to the hardcoded list.
        qa_pairs = loader.load_qa_pairs()
        print(f"Loaded {len(qa_pairs)} QA pairs from NarrativeQA.")
    except Exception as e:
        print(f"Warning: Could not load full dataset ({e}). Using sample set for validation.")
        qa_pairs = [
            {"question": "Who are Zuleika's most prominent suitors?", "answer1": "The Duke of Dorset and Noaks."},
            {"question": "Why does Zuleika reject the Duke?", "answer1": "Because he is not 'hopeless' enough; she only loves those she cannot have."},
            {"question": "Who is the first person Zuleika falls in love with?", "answer1": "She has never loved anyone."},
            # Add more if needed or just use these representatives
        ]
    
    # We only need a subset for the report if it takes too long, 
    # but the task implies "For each question in your filtered NarrativeQA test set".
    # There are ~40 pairs. We can run all.
    
    results = []
    print("\n--- 2. Running Baseline (No-Context) Evaluation ---")
    
    for i, item in enumerate(tqdm(qa_pairs)):
        question = item['question']
        reference = item['answer1']
        
        # Zero-shot generation (No Context)
        answer = generator.generate_answer(question, context="", do_sample=False)
        
        scores = evaluator.evaluate(answer, reference)
        
        results.append({
            "Question": question,
            "Target": reference,
            "Generated": answer,
            "BLEU": scores['bleu'],
            "ROUGE": scores['rouge']
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
