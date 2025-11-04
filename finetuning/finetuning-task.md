# LLM Finetuning Task: Comparing PEFT Techniques on Gemma-3-1b-it

## 1. Objective

The primary goal of this task is to finetune the `google/gemma-3-1b-it` large language model using two distinct Parameter-Efficient Fine-Tuning (PEFT) techniques. You will compare their effectiveness based on performance metrics, resource consumption (memory and time), and discuss potential generalization behaviors.

## 2. Base Language Model

*   **Model:** `google/gemma-3-1b-it`
*   **Link:** [https://huggingface.co/google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
*   **Rationale:** This model provides a strong instruction-following baseline while being relatively small, making it feasible for experimentation within typical resource constraints (e.g., Google Colab).

## 3. Datasets & Sampling

You will use a combined dataset sampled from the following sources:

*   **Alpaca:** `tatsu-lab/alpaca` ([https://huggingface.co/datasets/tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)) - General instruction following.
*   **Tulu v2 SFT:** `allenai/tulu-v2-sft-mixture` ([https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture)) - Diverse mixture including chat, coding, reasoning.
*   **Ultrachat 200k:** `HuggingFaceH4/ultrachat_200k` ([https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)) - High-quality, multi-turn chat data.

**Sampling Strategy:**

1.  Randomly select **5,000 training samples** from *each* dataset.
2.  Randomly select **2,000 test samples** from *each* dataset.
3.  Combine these samples to create a final training set (15,000 samples) and a final test set (6,000 samples). Ensure no overlap between train and test sets.

## 4. Data Preprocessing

*   **Goal:** Convert the sampled data into a consistent instruction-following format suitable for the `gemma-3-1b-it` model and standard training pipelines (e.g., Hugging Face `transformers.Trainer` or `trl.SFTTrainer`).
*   **Required Format:** Implement a clear template. A recommended starting point (adapt as needed based on dataset structure and model requirements):

    ```
    ### Instruction:
    {Instruction text from the dataset}

    ### Input:
    {Input context from the dataset, if available, otherwise leave blank or integrate into Instruction}

    ### Response:
    {Expected response/output from the dataset}
    ```
    *Note: Carefully examine each dataset's structure (`prompt`/`completion`, `instruction`/`input`/`output`, `messages` format for chat) and map it consistently to your chosen template.*

*   **Tools:** Use libraries like Hugging Face `datasets` for loading, sampling, and mapping/formatting.
*   **Documentation:** Clearly document the specific preprocessing steps, including how fields from each original dataset were mapped to your template, in your `README.md` and report.

## 5. Finetuning Techniques

Select **two** distinct finetuning techniques from the list below. Your chosen techniques should ideally represent different approaches to parameter efficiency.

ORPO (Odds Ratio Preference Optimization)
PEFT (Parameter-Efficient Fine-Tuning)
LoRA (Low-Rank Adaptation)
GaLore (Gradient Low-Rank Projection)
GRPO (Group Relative Policy Optimization)
QLoRA (Quantized Low-Rank Adaptation)
LoRA-GGPO (Gradient-Guided Perturbation Optimization)
DoRA (Dropout Regularized Adapter)
LoftQ (Low-rank Optimization with Fine-Tuned Quantization)
ReFT (Representation Fine-Tuning)
SMT (Sparse Matrix Tuning)

**Proposed Selection:**

1.  **QLoRA (Quantized Low-Rank Adaptation):** Builds on LoRA by using a quantized (e.g., 4-bit) base model and training LoRA adapters on top. Focuses on high memory efficiency.
2.  **GaLore (Gradient Low-Rank Projection):** A more recent technique focusing on projecting gradients into a low-rank subspace during optimization, potentially offering different efficiency/performance trade-offs compared to adapter-based methods.

*(Alternative options if you wish to deviate, provide strong justification: ORPO, DoRA, LoftQ, ReFT, etc.)*

**Implementation:**

*   Use relevant libraries like Hugging Face `transformers`, `peft`, `bitsandbytes` (for QLoRA), and potentially `trl`.
*   Finetune the *same* `gemma-3-1b-it` base model separately using each of the two chosen techniques on the prepared training dataset.
*   Train for a sufficient number of steps/epochs to observe meaningful improvement but remain mindful of resource limits (e.g., 1-3 epochs). Use the same hyperparameters (learning rate, batch size, epochs if possible) across techniques for a fair comparison, adjusting only where inherently required by the technique itself (e.g., quantization parameters for QLoRA).

**Justification (Required in Report):**

For *each* of the two chosen techniques, provide a justification based on:

*   **Potential Performance Improvement:** Why might this technique yield good results on instruction-following tasks?
*   **Memory/Resource Efficiency:** How does this technique save memory or computational resources compared to full finetuning or other PEFT methods?
*   **Generalization Behavior (Hypothesized):** How might this technique affect the model's ability to generalize to unseen instructions compared to others?

## 6. Evaluation

*   **Metrics:**
    *   **BLEU-4:** Measures n-gram overlap, useful for generation precision.
    *   **ROUGE-L:** Measures longest common subsequence, useful for recall and fluency.
*   **Procedure:**
    1.  Evaluate the **base `gemma-3-1b-it` model** (before finetuning) on the prepared test set using the chosen metrics. This is your baseline.
    2.  Evaluate **each of the two finetuned models** on the same test set using the same metrics.
    3.  Record **peak GPU memory usage** and **total training time** for each finetuning run. Plot **BLEU/ROUGE vs. memory use** for each technique. 
*   **Reporting Table:** Present results clearly, similar to this structure:

    | Technique | BLEU-4 (Before) | BLEU-4 (After) | ROUGE-L (Before) | ROUGE-L (After) | Peak Memory (GB) | Training Time (Hrs) |
    | :-------- | :-------------- | :------------- | :--------------- | :-------------- | :--------------- | :------------------ |
    | QLoRA     | [Baseline]      | [Result]       | [Baseline]       | [Result]        | [Result]         | [Result]            |
    | GaLore    | [Baseline]      | [Result]       | [Baseline]       | [Result]        | [Result]         | [Result]            |
    *(Replace [Baseline] and [Result] with actual values)*

## 7. Proposed Ensemble/Hybrid Approach

*   **Task:** Propose and explain *one* potential ensemble or hybrid setup combining *at least two* of your chosen fine-tuning techniques (LoRA, QLoRA, GaLore).
*   **Example Structure:**
    *   **Chosen Combination:** e.g., "Hybrid: QLoRA base model with standard LoRA adapters applied post-quantization" or "Ensemble: Inference-time averaging of LoRA and GaLore model outputs".
    *   **Rationale:** Justify why this combination might be beneficial. Consider aspects like:
        *   **Performance:** Could it capture the strengths of both methods?
        *   **Resource Use:** Does it offer a better trade-off than either method alone? (e.g., memory savings from QLoRA + adaptation from LoRA).
        *   **Generalization:** Could the combination lead to more robust outputs?
*   **Note:** You only need to *propose and justify* this; implementation is optional unless explicitly requested later.

## 8. Deliverables

1.  **Code Repository:**
    *   A publicly accessible repository (e.g., GitHub, GitLab).
    *   Well-structured and commented Python code for:
        *   Data sampling and preprocessing.
        *   Finetuning script(s) demonstrating the implementation for each of the two chosen techniques.
        *   Evaluation script(s).
    *   A `requirements.txt` file listing all necessary libraries.
    *   A detailed `README.md` file including:
        *   Project overview and objectives.
        *   Setup instructions (environment, dependencies).
        *   Step-by-step guide to run the preprocessing, training, and evaluation scripts.
        *   Summary of the results (can reference the report).
        *   Link to the final report.

2.  **Report (Max 2 pages):**
    *   Format: PDF or Markdown.
    *   Content:
        *   **Introduction:** Briefly state the goal and scope.
        *   **Approach & Methodology:** Describe the base model, datasets, preprocessing steps, chosen finetuning techniques, and evaluation metrics.
        *   **Implementation Details:** Briefly mention key libraries/tools used. Highlight any significant challenges encountered (e.g., library compatibility, resource constraints) and how they were addressed.
        *   **Results & Discussion:** Present the evaluation results table. Interpret the findings: compare the performance of the techniques, discuss the observed resource usage (memory/time). Show 5-10 example generations for each model side-by-side.
        *   **Ensemble/Hybrid Proposal:** Detail your proposed combination and its justification.
        *   **Conclusion:** Summarize the key findings and discuss the trade-offs between the compared techniques based on your results.

## 9. Execution Notes & Tips

*   **Resource Management:** Finetuning LLMs can be resource-intensive. Use tools like `nvidia-smi` to monitor GPU usage. Leverage techniques like gradient accumulation to manage memory if needed.
*   **Colab Pro-Tip:** If using Google Colab, save model checkpoints, datasets, and results frequently to Google Drive to prevent data loss due to session timeouts. Mount your Drive at the beginning of your notebook.
*   **Runtime:** Be mindful of training times. Aim for runs long enough to show learning (e.g., decreasing loss, improving metrics over a few epochs/many steps) but short enough to be practical. A few hours per technique might be a reasonable target depending on the hardware.
*   **Library Versions:** Pay attention to library versions (`transformers`, `peft`, `accelerate`, `bitsandbytes`, etc.) as compatibility issues can arise. Document the versions used in your `requirements.txt`.
*   **Reproducibility:** Set random seeds where possible (e.g., for sampling, model initialization, data shuffling) to improve the reproducibility of your results.