import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
Shared configuration and utility functions for finetuning scripts.

This module contains common constants, data paths, and helper functions used across
different finetuning strategies (QLoRA, GaLore) for the Gemma model.
"""

# Configuration
MODEL_ID = "google/gemma-3-1b-it"
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'finetune_train.jsonl')
OUTPUT_DIR_QLORA = os.path.join(os.path.dirname(__file__), 'output_qlora')
OUTPUT_DIR_GALORE = os.path.join(os.path.dirname(__file__), 'output_galore')

# Shared Hyperparameters
MAX_SEQ_LENGTH = 1024

def format_instruction(sample):
    """
    Formats the instruction for Gemma.

    Handles both single example (dict) and batch (dict of lists) processing.
    Constructs the prompt using the "### Instruction", "### Input" (optional),
    and "### Response" format.

    Args:
        sample (dict): A dictionary containing 'instruction', 'output', and optionally 'input'.
            Can be a single example or a batch of examples.

    Returns:
        list: A list of formatted prompt strings.
    """
    if isinstance(sample['instruction'], list):
        output_texts = []
        for i in range(len(sample['instruction'])):
            instruction = sample['instruction'][i]
            input_text = sample['input'][i] if 'input' in sample and sample['input'][i] else ""
            output_text = sample['output'][i]
            
            if input_text:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
            output_texts.append(text)
        return output_texts
    else:
        # Single example
        instruction = sample['instruction']
        input_text = sample.get('input', '')
        output_text = sample['output']
        
        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
        return [text]

def load_tokenizer_and_model(model_id, quantization_config=None, use_cache=False):
    """
    Loads the Tokenizer and Model with standard configurations.

    Args:
        model_id (str): The Hugging Face model identifier.
        quantization_config (BitsAndBytesConfig, optional): Configuration for quantization.
            Defaults to None.
        use_cache (bool, optional): Whether to use KV cache. Defaults to False.

    Returns:
        tuple: A tuple containing the (tokenizer, model).
    """
    print(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not quantization_config else None
    )
    
    model.config.use_cache = use_cache
    if not quantization_config:
        model.config.pretraining_tp = 1 
        
    return tokenizer, model
