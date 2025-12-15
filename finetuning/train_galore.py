import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

try:
    from galore_torch import GaLoreAdamW, GaLoreAdamW8bit
except ImportError:
    print("Error: galore_torch not installed.")
    sys.exit(1)

# Configuration
MODEL_ID = "google/gemma-3-1b-it"
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'finetune_train.jsonl')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output_galore')

# Hyperparameters
RANK = 128
UPDATE_PROJ_GAP = 200
SCALE = 2
TARGET_MODULES = ["attn", "mlp"]

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
MAX_SEQ_LENGTH = 1024

def format_instruction(sample):
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
        instruction = sample['instruction']
        input_text = sample.get('input', '')
        output_text = sample['output']
        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
        return [text]

def main():
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading dataset from {TRAIN_FILE}...")
    dataset = load_dataset('json', data_files=TRAIN_FILE, split='train')

    # GaLore Config
    galore_params = []
    target_modules_list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    id_galore_params = set()
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(t in name for t in target_modules_list):
                galore_params.append({
                    "params": module.parameters(),
                    "rank": RANK,
                    "update_proj_gap": UPDATE_PROJ_GAP,
                    "scale": SCALE,
                    "proj_type": "std"
                })
                for p in module.parameters():
                    id_galore_params.add(id(p))
                    
    non_galore_params = [p for p in model.parameters() if id(p) not in id_galore_params]
    param_groups = galore_params + [{"params": non_galore_params}]
    
    optimizer = GaLoreAdamW8bit(param_groups, lr=LEARNING_RATE)
    print("GaLore Optimizer configured.")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        formatting_func=lambda x: [format_instruction(s) for s in x] if isinstance(x, list) else format_instruction(x),
        optimizers=(optimizer, None)
    )
    
    print("Starting GaLore training...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
