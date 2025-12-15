import os
import sys

# Add project root to sys.path to access the finetuning package
# This allows running `python finetuning/train_galore.py` from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Finetunes Gemma 2 9B using GaLore (Gradient Low-Rank Projection).
Uses 8-bit optimizer and gradient projection for memory-efficient full-parameter training.
"""

import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
)
from trl import SFTTrainer
from finetuning.common import (
    MODEL_ID, TRAIN_FILE, OUTPUT_DIR_GALORE, MAX_SEQ_LENGTH,
    format_instruction, load_tokenizer_and_model
)

try:
    from galore_torch import GaLoreAdamW, GaLoreAdamW8bit
except ImportError:
    print("Error: galore_torch not installed.")
    sys.exit(1)

# Configuration
OUTPUT_DIR = OUTPUT_DIR_GALORE

# Hyperparameters
RANK = 128
UPDATE_PROJ_GAP = 200
SCALE = 2
TARGET_MODULES = ["attn", "mlp"]

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8 # Increased to match effective batch size
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1

def main():
    """
    Executes the GaLore training pipeline.
    
    Steps:
    1. Loads tokenizer and model (no quantization).
    2. Loads the dataset.
    3. Configures GaLore optimizer for specific target modules.
    4. Initializes SFTTrainer with training arguments.
    5. Runs training and saves the final model.
    """
    torch.cuda.empty_cache() # Clear any residual memory
    
    # Load Tokenizer & Model
    tokenizer, model = load_tokenizer_and_model(MODEL_ID, use_cache=False)
    
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
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        eval_strategy="no",
        gradient_checkpointing=True, # Critical for saving memory
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
