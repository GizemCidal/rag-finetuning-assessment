import sys
import os

# Add project root to sys.path to access the finetuning package
# This allows running `python finetuning/train_qlora.py` from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Finetunes Gemma 2 9B using QLoRA (Quantized Low-Rank Adaptation).
Uses 4-bit quantization and LoRA adapters for memory efficiency.
"""

import torch
import json
from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from finetuning.common import (
    MODEL_ID, TRAIN_FILE, OUTPUT_DIR_QLORA, MAX_SEQ_LENGTH,
    format_instruction, load_tokenizer_and_model
)

# Configuration
OUTPUT_DIR = OUTPUT_DIR_QLORA

# Hyperparameters
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

def main():
    """
    Executes the QLoRA training pipeline.
    
    Steps:
    1. Loads tokenizer and 4-bit quantized model.
    2. Prepares model for k-bit training and attaches LoRA adapters.
    3. Loads the dataset.
    4. Initializes SFTTrainer with training arguments.
    5. Runs training and saves the adapter weights.
    """
    # Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load Tokenizer & Model
    tokenizer, model = load_tokenizer_and_model(MODEL_ID, quantization_config=bnb_config)
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES
    )
    
    # Load Dataset
    print(f"Loading dataset from {TRAIN_FILE}...")
    dataset = load_dataset('json', data_files=TRAIN_FILE, split='train')
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="none"
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        formatting_func=lambda x: [format_instruction(s) for s in x] if isinstance(x, list) else format_instruction(x),
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
