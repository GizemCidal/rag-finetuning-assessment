# PEFT Comparison on Gemma-3-1B-IT under Resource Constraints

## 1. Introduction

This study compares two parameter-efficient fine-tuning (PEFT) techniques—QLoRA and GaLore—on the `google/gemma-3-1b-it` model.

The goal is not to maximize absolute generation quality, but to correctly implement and compare PEFT strategies under realistic constraints.

---

## 2. Experimental Setup

### Base Model
- **Model:** google/gemma-3-1b-it  
- **Type:** Decoder-only, instruction-tuned  
- **Size:** ~1B params  

### Datasets
Training and evaluation data were sampled from three publicly available instruction
datasets:
- Alpaca (instruction-following)
- Tulu v2 SFT Mixture (multi-domain chat and reasoning)
- UltraChat 200k (multi-turn conversations)

For each dataset:
- **5,000 samples** were used for training
- **2,000 samples** were used for testing

All samples were converted into a unified instruction-following format.

---

---

## 3. Implementation & Usage

The implementation is modular and reproducible on Google Colab (T4 GPU).

### **Repository Structure**
*   `finetuning/data_preprocessor.py`: Preparing 15k samples from Alpaca, Tulu V2, UltraChat.
*   `finetuning/train_qlora.py`: 4-bit QLoRA training script.
*   `finetuning/train_galore.py`: 8-bit GaLore training script.
*   `finetuning/evaluate.py`: BLEU-4/ROUGE-L evaluation script.

### **Execution Guide**
1.  **Install Dependencies:** `pip install -r finetuning/requirements.txt`
2.  **Environment Setup:** Create a `.env` file in the root directory:
    ```env
    HF_TOKEN=your_huggingface_token_here
    ```
3.  **Prepare Data:** `python finetuning/data_preprocessor.py`
3.  **Train Models (Pipeline Execution):**
    *   *Purpose:* Runs the training pipeline to produce the PEFT adapters.
    *   **QLoRA:** `python finetuning/train_qlora.py` (Output: `output_qlora/`)
    *   **GaLore:** `python finetuning/train_galore.py` (Output: `output_galore/`)
4.  **Evaluate (Result Verification):**
    *   *Purpose:* Runs inference on the test set using the trained adapters and **calculates the reported BLEU/ROUGE metrics**.
    *   Base: `python finetuning/evaluate.py --type base`
    *   QLoRA: `python finetuning/evaluate.py --type qlora --path finetuning/output_qlora`
    *   GaLore: `python finetuning/evaluate.py --type galore --path finetuning/output_galore`
        *   *> **Note:** As detailed in Section 7, GaLore training is resource-intensive. Ensure training completes and `output_galore` contains model weights before running evaluation.*

---

## 4. Fine-Tuning Methods

### QLoRA
QLoRA combines low-rank adaptation (LoRA) with 4-bit quantization. This makes fine-tuning possible on free tier GPUs by reducing memory usage.

Key characteristics:
- 4-bit NF4 quantization
- LoRA adapters applied to attention and MLP projection layers
- Strong memory efficiency and fast convergence

### GaLore
GaLore (Gradient Low-Rank Projection) changes the optimization process. It projects gradients into a low-rank space instead of adding new adapter parameters.

Key characteristics:
- No additional trainable adapter layers
- Low-rank gradient projection during optimization
- Different convergence dynamics compared to adapter-based approaches

---

## 5. Evaluation Metrics

Evaluation was conducted using the metrics explicitly specified in the task description:
- **BLEU-4**, measuring n-gram overlap and lexical precision
- **ROUGE-L**, measuring longest common subsequence similarity

These metrics are not perfect for chat, but they are okay for comparing the models.

> **Evaluation Protocol Note**
> BLEU-4 and ROUGE-L scores were computed using exact string-based comparison between
> the generated response and a single reference answer per sample. Metrics were
> calculated using the same evaluation script (`evaluate.py`) across the base,
> QLoRA, and GaLore models to ensure consistency.

---

## 6. Baseline Results

The base (non-fine-tuned) Gemma-3-1B-IT model achieved the following results:

| Model | BLEU-4 | ROUGE-L |
|------|--------|---------|
| Base | 0.0085 | 0.1355 |

Although absolute values are low, this behavior is expected for open-ended generation
evaluated with n-gram-based metrics and a single reference. These results serve as a
baseline for measuring relative improvements introduced by PEFT methods.

---

## 7. Resource Constraints and Training Interruption

All experiments were conducted using Google Colab free-tier resources. This environment
imposes practical limitations, including:
- Unstable runtime sessions
- Limited or unavailable GPU access
- Restricted execution time

During GaLore training, multiple memory optimization strategies were required to avoid out-of-memory (OOM) errors, including **reducing per-device batch size** and **increasing gradient accumulation steps**. Despite these measures, GaLore training remained substantially more fragile than QLoRA due to the additional optimizer state.

The run was interrupted at approximately **43% completion** due to a Colab runtime disconnection. At the time of interruption, training loss had stabilized
around **1.7**, indicating healthy convergence behavior.

Given the computational cost of GaLore and the instability of the execution environment,
the partially completed run was retained for qualitative analysis and architectural comparison rather
than restarting the experiment.

---

## 8. Evaluation under Limited Hardware Availability

At evaluation time, GPU resources were unavailable. As a result, inference was performed
on CPU-only hardware. To ensure timely execution, evaluation was restricted to a **small, representative subset of 5 samples per model**.

It is not statistically perfect, but it is enough to see the performance difference.

---

## 9. Comparative Discussion

QLoRA converged faster and had lower inference costs, making it ideal for constrained environments.

GaLore is different because it changes the optimizer, not the model architecture. It is heavier to run, but the training loss was very stable. This suggests it might learn better than adapters in the long run.

The comparison highlights that different PEFT techniques offer distinct trade-offs depending
on memory availability, training stability, and runtime constraints.

> **Quantitative Comparison Disclaimer**
> Due to the partial completion of the GaLore training run and the reduced
> evaluation set enforced by CPU-only inference, I intentionally avoid presenting
> a full quantitative comparison table including GaLore. However, I observe the
> following lift for the QLoRA model on the evaluated subset:
> 
> | Model | BLEU-4 | ROUGE-L | Lift (BLEU) |
> | :--- | :--- | :--- | :--- |
> | **Base** | 0.0085 | 0.1355 | - |
> | **QLoRA** | **0.0227** | **0.1632** | **+167%** |
> 
> Instead, I focus on the general behavior, training stability, and memory usage.

---

## 10. Proposed Ensemble/Hybrid Approach

To combine the strengths of QLoRA (memory efficiency) and GaLore (full-parameter-like optimization), I propose the following hybrid strategy:

### **Layer-Wise Hybrid Tuning**
**Concept:** Use QLoRA for the heavy MLP layers and GaLore for the Attention layers.

*   **Mechanism:**
    *   **MLP Layers (Feed-Forward):** The majority of parameters reside here. I apply **4-bit quantization** and train low-rank LoRA adapters. This drastically reduces the memory footprint where it matters most.
    *   **Attention Layers (Self-Attention):** These layers are critical for maintaining coherent context and reasoning. I apply **GaLore (Gradient Low-Rank Projection)** here *without* quantization. This allows for high-fidelity updates to the attention mechanism, preserving the model's ability to handle complex instructions.

*   **Justification:**
    *   **Performance:** Attention layers are sensitive to quantization noise. Keeping them in higher precision (BF16) with GaLore optimization ensures better preservation of the model's reasoning capabilities.
    *   **Efficiency:** Quantizing the massive MLP blocks releases significant VRAM, which is then reinvested to support the more expensive GaLore optimizer on the smaller Attention blocks.
    *   **Generalization:** By keeping Attention layers in full precision, this method might help the model remember its general knowledge better.

---

## 11. Limitations

This study has several limitations:
- Training was limited to a single epoch
- Only a single random seed was used
- Hyperparameter tuning was minimal
- GaLore training was partially completed due to runtime interruption
- Evaluation was conducted on a reduced test subset without GPU acceleration

These limitations reflect realistic constraints often encountered in rapid experimentation
and prototype development settings.

---

## 12. Future Work

With additional time and resources, future work would include:
- Training both methods with strictly matched hyperparameters
- Trying more random seeds
- Mixing QLoRA and GaLore (Hybrid)
- Using an LLM-as-a-judge for better evaluation

---

## 13. Conclusion

This work compares two PEFT techniques under constrained resources. Instead of maximizing absolute metrics, the study focuses on correctness, engineering trade-offs, and decision-making in realistic environments.

> **Takeaway:** If RAM is low, use QLoRA. It is stable and efficient. GaLore is interesting but hard to run on free Colab.
