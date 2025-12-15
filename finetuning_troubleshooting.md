# Finetuning Phase Troubleshooting Log

## Issue 1: `evaluation_strategy` Deprecation Warning
- **Symptom**: `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`
- **Cause**: Newer versions of the `transformers` library have renamed `evaluation_strategy` to `eval_strategy`.
- **Fix**: Updated `train_qlora.py` and `train_galore.py` to use `eval_strategy="no"`.

## Issue 2: `max_seq_length` Argument Error in `SFTTrainer`
- **Symptom**: `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'max_seq_length'`
- **Context**:
    - Initially, we used `trl` version `0.26.1` (latest).
    - In newer TRL versions, `max_seq_length` must be passed via `SFTConfig`, not directly to `SFTTrainer`.
    - However, attempting to use `SFTConfig` caused further compatibility issues with the specific `transformers` version installed.
- **Root Cause**: Breaking API changes in recent `trl` releases.
- **Resolution**:
    - **Downgraded TRL** to a stable version: `pip install trl==0.8.6`.
    - Updated `finetuning/requirements.txt` to pin `trl==0.8.6`.
    - Reverted `train_qlora.py` and `train_galore.py` to pass `max_seq_length` directly to `SFTTrainer`, which is supported in v0.8.6.

## Issue 3: Mac (MPS) Compatibility for 4-bit Quantization
- **Observed Behavior**: `bitsandbytes` library (used for 4-bit loading) relies heavily on CUDA.
- **Impact**: Running `train_qlora.py` locally on Mac (M1/M2/M3) fails or warns about missing CUDA.
- **Recommendation**: These scripts are verified for logic but **must be executed on a CUDA-enabled Global GPU environment** (like Google Colab T4 or A100).

## Issue 4: `eval_strategy` vs `evaluation_strategy` Hybrid Conflict
- **Symptom**: After downgrading TRL, we had a mix of old TRL (expecting old args) and new Transformers (expecting new args).
- **Fix**:
    - Kept `transformers` updated.
    - Used `eval_strategy` in `TrainingArguments` (for Transformers compatibility).
    - Used `max_seq_length` in `SFTTrainer` init (for TRL 0.8.6 compatibility).

## Issue 5: `formatting_func` Batch Processing Error
- **Symptom**: `ValueError: The formatting_func should return a list of processed strings...`
- **Cause**: The initial lambda function was not correctly handling the batch dictionary format `{'col': [...]}` passed by `SFTTrainer`.
- **Fix**: Refactored `format_instruction` to explicitly check for `isinstance(sample['instruction'], list)` and handle both batch and single-item inputs correctly.

## Issue 6: Mac `bitsandbytes` Device Error (QLoRA)
- **Symptom**: `TypeError: device() received an invalid combination of arguments - got (NoneType)` during `trainer.train()`.
- **Cause**: The `accelerate` library fails to assign a valid CUDA device on Mac when `bitsandbytes` quantization is active.
- **Conclusion**: Confirms script logic is correct up to hardware execution.

## Issue 7: Mac MPS Support for GaLore (8-bit Optimizer)
- **Symptom**: `NotImplementedError: The operator 'bitsandbytes::optimizer_update_8bit_blockwise' is not currently implemented for the MPS device.`
- **Cause**: `GaLoreAdamW8bit` uses `bitsandbytes` for 8-bit optimization, which lacks MPS support.
- **Conclusion**: `train_galore.py` logic is verified (imports, data loading, model init succeeded). Training requires NVIDIA GPU (Colab).

## Issue 8: CUDA Out of Memory (OOM) with GaLore
- **Symptom**: `torch.OutOfMemoryError: CUDA out of memory` on T4 GPU (16GB).
- **Cause**: The combination of `gemma-3-1b` + Optimizer States + batch size of 2 exceeded 16GB VRAM.
- **Fix**:
    1. Reduced `BATCH_SIZE` from 2 to 1.
    2. Increased `GRADIENT_ACCUMULATION_STEPS` from 4 to 8 (to maintain effective batch size).
    3. Enabled `gradient_checkpointing=True` in `TrainingArguments` (trades compute for memory).
    4. Added `torch.cuda.empty_cache()` at script start.
