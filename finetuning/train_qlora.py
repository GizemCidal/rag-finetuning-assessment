import os
import sys
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration
MODEL_ID = "google/gemma-3-1b-it"
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'finetune_train.jsonl')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output_qlora')

# Hyperparameters
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
MAX_SEQ_LENGTH = 1024

def format_instruction(sample):
    """
    Format the instruction for Gemma.
    Gemma uses specific chat templates, but for SFT we can mostly stick to the prompt format
    or use the chat template if we want to retain the 'instruction-tuned' nature appropriately.
    For simplicity and consistency with the task 'Instruction/Input/Response' format:
    """
    instruction = sample['instruction']
    input_text = sample.get('input', '')
    output_text = sample['output']
    
    # Standard Alpaca-style prompt, or we can use Gemma's chat template structure `<start_of_turn>...`
    # Let's use a clear Instruction format that the model can learn.
    if input_text:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
    
    return text

def main():
    print(f"Loading model: {MODEL_ID}")
    
    # 1. Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False # Silence warnings during training
    model.config.pretraining_tp = 1 
    
    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix for fp16

    # 4. Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES
    )
    
    # 5. Load Dataset
    print(f"Loading dataset from {TRAIN_FILE}...")
    dataset = load_dataset('json', data_files=TRAIN_FILE, split='train')
    
    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no", # We eval separately
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="none" # Disable wandb for local test to avoid auth issues if not set up
    )

    # 7. Trainer
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
