import os
import torch
import json
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

"""
Evaluation script for finetuned Gemma models.

This script loads a base, QLoRA, or GaLore model and evaluates it on a test dataset
using BLEU and ROUGE metrics.
"""

# Configuration
MODEL_ID = "google/gemma-3-1b-it"
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_FILE = os.path.join(DATA_DIR, 'finetune_test.jsonl')

def load_model(model_type, model_path=None):
    """
    Loads the requested model type (base, qlora, galore).

    Args:
        model_type (str): Type of model to load ("base", "qlora", "galore").
        model_path (str, optional): Path to adapters or full model directory.

    Returns:
        tuple: A tuple containing (model, tokenizer).
    """
    print(f"Loading {model_type} model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Generation
    
    if model_type == "base":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    elif model_type == "qlora":
        # Load Base in 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        # Load Adapter
        if model_path:
            print(f"Loading adapter from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
        else:
            print("Warning: No adapter path provided for QLoRA, using base model (quantized).")
            
    elif model_type == "galore":
        # Load full finetuned model
        if not model_path:
            raise ValueError("GaLore requires model_path")
        print(f"Loading finetuned model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16, # matching training
            trust_remote_code=True
        )

    model.eval()
    return model, tokenizer

def format_prompt(sample):
    """
    Formats the sample into a prompt for generation (excluding output).

    Args:
        sample (dict): A dictionary containing 'instruction' and optionally 'input'.

    Returns:
        str: The formatted prompt string.
    """
    instruction = sample['instruction']
    input_text = sample.get('input', '')
    
    if input_text:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return text

def calculate_metrics(predictions, references):
    """
    Calculates BLEU-4 and ROUGE-L metrics.

    Args:
        predictions (list): List of generated response strings.
        references (list): List of ground truth response strings.

    Returns:
        tuple: (avg_bleu, avg_rouge) scores.
    """
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleu_scores = []
    rouge_scores = []
    
    smooth = SmoothingFunction().method1
    
    for pred, ref in zip(predictions, references):
        # BLEU
        pred_tokens = pred.split()
        ref_tokens = [ref.split()]
        if len(pred_tokens) == 0:
            bleu_scores.append(0)
        else:
            bleu_scores.append(sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth))
            
        # ROUGE
        scores = rouge.score(ref, pred)
        rouge_scores.append(scores['rougeL'].fmeasure)
        
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    
    return avg_bleu, avg_rouge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["base", "qlora", "galore"], required=True)
    parser.add_argument("--path", help="Path to adapter (QLoRA) or model dir (GaLore)")
    parser.add_argument("--limit", type=int, default=100, help="Number of test samples to eval (default 100 for speed)")
    args = parser.parse_args()
    
    # Load Data
    print(f"Loading test data from {TEST_FILE}")
    dataset = load_dataset('json', data_files=TEST_FILE, split='train')
    
    if args.limit > 0:
        dataset = dataset.select(range(min(len(dataset), args.limit)))
    
    # Load Model
    model, tokenizer = load_model(args.type, args.path)
    
    predictions = []
    references = []
    
    print("Generating responses...")
    for sample in tqdm(dataset):
        prompt = format_prompt(sample)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=False, # Deterministic for Eval
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode: cut off the prompt
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract raw response part:
        # Prompt ends with "### Response:\n"
        # Splitting by marker
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text # Fallback
            
        predictions.append(response)
        references.append(sample['output'])
        
    # Calculate Metrics
    bleu, rouge = calculate_metrics(predictions, references)
    
    print("\n--- Results ---")
    print(f"Model: {args.type}")
    print(f"BLEU-4: {bleu:.4f}")
    print(f"ROUGE-L: {rouge:.4f}")
    
    # Save results to file for report
    results = {
        "model_type": args.type,
        "bleu": bleu,
        "rouge": rouge
    }
    with open(f"eval_results_{args.type}.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
