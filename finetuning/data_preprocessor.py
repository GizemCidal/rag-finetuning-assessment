import os
import json
import random
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset

# Configuration
DATA_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
TRAIN_SAMPLES_PER_SOURCE = 5000
TEST_SAMPLES_PER_SOURCE = 2000
SEED = 42

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def format_alpaca(example):
    return {
        "instruction": example['instruction'],
        "input": example['input'],
        "output": example['output']
    }

def format_tulu(example):
    # Tulu structure: 'messages': [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
    messages = example['messages']
    instruction = ""
    input_text = ""
    output_text = ""
    
    # Simple extraction: specific to Tulu's structure which is usually user->assistant
    # We will flatten multi-turn into single turn for simplicity if needed, or better, 
    # take the last user message as instruction if context isn't strictly required, 
    # but Tulu is SFT mixture. Let's try to capture the full conversation context in instruction 
    # or just the first user prompt. 
    # User plan says: "system + user -> Instruction", "assistant -> Response".
    
    # Finding first user and first assistant (or last assistant)
    user_content = []
    assistant_content = []
    
    for msg in messages:
        if msg['role'] == 'user':
            user_content.append(msg['content'])
        elif msg['role'] == 'assistant':
            assistant_content.append(msg['content'])
        elif msg['role'] == 'system':
             # Prepend system prompt to user instruction
             user_content.insert(0, f"System: {msg['content']}")
    
    # Join all user/system parts for instruction context
    if user_content:
        instruction = "\n\n".join(user_content)
    
    # Output is the assistant response. If multiple, we target the last one or join?
    # Usually SFT targets the final response. Let's take the first assistant response to match the immediate instruction
    # or the last one? UltraChat logic is "First user -> Instruction, Final assistant -> Response".
    # Let's align Tulu to "Instruction = Cumulative Context", "Output = Last Response" 
    # BUT user spec for Tulu: "system + user -> Instruction", "assistant -> Response".
    
    if assistant_content:
        output_text = assistant_content[-1] # Target the final answer
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }

def format_ultrachat(example):
    # UltraChat: 'messages': list of role/content. 
    # User spec: "First user message -> Instruction", "Final assistant message -> Response", "Intermediate turns discarded".
    messages = example['messages']
    
    instruction = ""
    output_text = ""
    
    # Find first user message
    for msg in messages:
        if msg['role'] == 'user':
            instruction = msg['content']
            break
            
    # Find last assistant message
    for msg in reversed(messages):
        if msg['role'] == 'assistant':
            output_text = msg['content']
            break
            
    return {
        "instruction": instruction,
        "input": "",
        "output": output_text
    }

def process_dataset(source_name, dataset_path, split, format_func, samples_train, samples_test):
    print(f"Loading {source_name} ({dataset_path})...")
    try:
        # Load dataset. Some might have different split names.
        ds = load_dataset(dataset_path, split=split)
    except Exception as e:
        print(f"Error loading {source_name}: {e}")
        return [], []

    print(f"  - Original size: {len(ds)}")
    
    # Shuffle
    ds = ds.shuffle(seed=SEED)
    
    # Split indices
    total_needed = samples_train + samples_test
    if len(ds) < total_needed:
        print(f"  - WARNING: Not enough samples in {source_name}. Needed {total_needed}, got {len(ds)}.")
        # Proceed with what we have? Or fail? Let's take max possible maintaining ratio? 
        # Or just take as many as possible for train, remainder test.
        # For this assessment, assuming datasets are large enough (they are).
    
    # Select indices
    # We convert to valid list of dicts first to avoid index issues with some dataset types
    ds_list = [format_func(ex) for ex in ds.select(range(total_needed))]
    
    train_data = ds_list[:samples_train]
    test_data = ds_list[samples_train:samples_train+samples_test]
    
    print(f"  - Selected: {len(train_data)} Train, {len(test_data)} Test")
    return train_data, test_data

def main():
    ensure_dir(DATA_OUTPUT_DIR)
    
    all_train = []
    all_test = []
    
    # 1. Alpaca
    # Alpaca 'train' split usually exists. 
    alp_train, alp_test = process_dataset(
        "Alpaca", "tatsu-lab/alpaca", "train", format_alpaca, 
        TRAIN_SAMPLES_PER_SOURCE, TEST_SAMPLES_PER_SOURCE
    )
    all_train.extend(alp_train)
    all_test.extend(alp_test)
    
    # 2. Tulu V2 SFT Mixture
    # This dataset is large. 'train' split.
    tulu_train, tulu_test = process_dataset(
        "Tulu V2", "allenai/tulu-v2-sft-mixture", "train", format_tulu,
        TRAIN_SAMPLES_PER_SOURCE, TEST_SAMPLES_PER_SOURCE
    )
    all_train.extend(tulu_train)
    all_test.extend(tulu_test)
    
    # 3. UltraChat 200k
    # 'train_sft' or 'train_gen' splits usually. HuggingFaceH4/ultrachat_200k has 'train_sft' and 'test_sft'.
    # We can pull from train_sft for both if it's large enough, or use valid splits.
    # Let's use 'train_sft' for training source and shuffle.
    uc_train, uc_test = process_dataset(
        "UltraChat", "HuggingFaceH4/ultrachat_200k", "train_sft", format_ultrachat,
        TRAIN_SAMPLES_PER_SOURCE, TEST_SAMPLES_PER_SOURCE
    )
    all_train.extend(uc_train)
    all_test.extend(uc_test)
    
    # Check totals
    print(f"\nTotal Train Samples: {len(all_train)}")
    print(f"Total Test Samples: {len(all_test)}")
    
    # Shuffle combined data
    random.shuffle(all_train)
    random.shuffle(all_test)
    
    # Save to JSONL
    train_path = os.path.join(DATA_OUTPUT_DIR, 'finetune_train.jsonl')
    test_path = os.path.join(DATA_OUTPUT_DIR, 'finetune_test.jsonl')
    
    print(f"Saving to {train_path}...")
    with open(train_path, 'w') as f:
        for item in all_train:
            f.write(json.dumps(item) + '\n')
            
    print(f"Saving to {test_path}...")
    with open(test_path, 'w') as f:
        for item in all_test:
            f.write(json.dumps(item) + '\n')
            
    print("Done!")

if __name__ == "__main__":
    main()
