# Advanced RAG Task Report

## 1. Introduction
This project builds a resource-efficient RAG system to answer questions about "Zuleika Dobson" (Project Gutenberg ID 1845) using NarrativeQA. The system targets constrained environments (Google Colab Free Tier) using hierarchical chunking and an on-disk vector database.

## 2. Approach & Methodology

### 1. Data Selection
- **Document:** "Zuleika Dobson; or, An Oxford Love Story" by Max Beerbohm (Project Gutenberg ID 1845).
    *   *Correction Reference:* Initially, the config file mislabeled ID 1845 as "The School for Scandal". Manual verification (`head` command) confirmed the content is indeed "Zuleika Dobson", matching the QA pairs from NarrativeQA.
- **QA Pairs:** Filtered from the `NarrativeQA` dataset. Total pairs found: 40.
    *   *Note:* A broader search included popular titles like *Dracula*, but they were rejected due to zero usable QA pairs in the test split. Selection was driven empirically by QA availability rather than literary preference.

### Key Components
*   **Chunking Strategy**: **Hierarchical Chunking**.
    *   *Parent Chunks*: 2000 characters. Providing broader context for the LLM.
    *   *Child Chunks*: 500 characters. Specific segments for dense retrieval.
    *   *Logic*: Retrieved child chunks map back to their parent chunks for context expansion.

## 3. Implementation Details
The solution is implemented in modular Python:

### 3.1 Data Cleaning & Normalization
To improve retrieval quality, I implemented a rigorous cleaning pipeline in `DataLoader`:
*   **Header/Footer Removal:** Stripped Project Gutenberg license text and introductions.
*   **Text Unwrapping (New):** Converted hard-wrapped lines (common in Gutenberg texts) into continuous paragraphs to preserve sentence structures.
*   **Whitespace Normalization:** Collapsed multiple spaces and standardized paragraph breaks (`\n\n`).

### 3.2 Retrieval Setup
*   **Vector DB:** Qdrant (Local Mode). Chosen for its ability to persist data to disk (`./data/qdrant_db`), minimizing RAM usage.
*   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dim), optimized for speed.
*   **Generator**: `google/gemma-3-1b-it`. A compact 1B instruction-tuned model.

*   `src/rag/chunking.py`: Custom sliding window splitter with parent-child ID tracking.
*   `src/rag/vector_db.py`: Qdrant wrapper handling upserts and search.
*   `src/rag/retriever.py`: Logic to deduplicate parent contexts from multiple retrieved children.
*   `src/rag/reranker.py`: Cross-Encoder implementation for re-ranking retrieved results.
*   `src/rag/config.py`: Centralized configuration.

### 3.3 Usage Guide
**Setup:**
1.  **Install Dependencies:** `pip install -r rag/requirements.txt`
2.  **Environment Setup:** Create a `.env` file in the root directory and add your Hugging Face token (required for accessing Gemma):
    ```env
    HF_TOKEN=your_huggingface_token_here
    ```

**Execution & Reproducibility:**

1.  **Run Full System Verification (End-to-End):**
    *   *Purpose:* Runs the complete pipeline (Data Download -> Chunking -> Indexing -> Retrieval -> Generation) and **calculates the BLEU/ROUGE scores** reported below.
    ```bash
    python scripts/rag/run_experiment.py
    ```

2.  **Run Baseline Comparison:**
    *   *Purpose:* Generates answers without RAG (Zero-shot) to establish the baseline metrics.
    ```bash
    python scripts/rag/benchmark_baseline.py
    ```


## 4. Parameter Evolution & Optimization
System performance was iteratively improved by tuning key parameters.

| Parameter | Initial Value | Final Value | Reason for Change |
| :--- | :--- | :--- | :--- |
| **Parent Chunk Size** | 1000 chars | **2000 chars** | Initial context was too fragmented. Doubling size provided the LLM with complete narrative arcs. |
| **Child Chunk Size** | 250 chars | **500 chars** | 250 chars often split sentences or lost semantic meaning necessary for vector matching. |
| **Top-K Retrieval** | 5 | **10** | Searching only 5 chunks resulted in high "Context Missing" errors. Increasing to 10 improved recall. |
| **Retrieval Strategy** | Vector Search Only | **Vector + Re-ranking** | Vector search alone returned "semantically close but irrelevant" matches. Re-ranking (fetching 30, selecting top 10) significantly improved precision. |

> **Rationale:** I changed these settings because I saw errors in the logs (like incomplete sentences), not just by guessing.

## 5. Results & Discussion

**Baseline RAG Performance (Hierarchical, Top-K=10 + Re-ranking):**
| Approach | BLEU-4 | ROUGE-L | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline (No RAG)** | **0.0033** | **0.0554** | Model has near-zero knowledge of the specific book. Fast inference (<1s). |
| **RAG (Vector Only)** | ~0.02 | ~0.07 | Suffer from retrieval noise. |
| **RAG (Re-ranking) + Cleaning** | **~0.11** | **~0.14** | **Best Case.** Data cleaning doubled the performance (0.08 -> 0.14). |

*> Note: Scores are improved but still limited by the archaic language gap.*

> **Note:** BLEU and ROUGE are strict metrics for this task. The absolute numbers are low, but the improvement over baseline is important.

### Resource Monitoring (Qualitative)
*   **Memory (RAM):** 
    *   **Indexing:** Peak usage was ~2GB during chunk embedding (batch processing effective).
    *   **Inference:** Using `gemma-3-1b-it` (4-bit/half precision) required ~2.5GB VRAM/RAM, well within Colab free tier limits (12GB+).
*   **Disk Usage:** 
    *   **Qdrant:** The vector database (`data/qdrant_db`) size is compact (~500MB) for the single book, proving local storage is viable.
*   **Latency:** 
    *   **Vector Search:** Milliseconds (<50ms).
    *   **Re-ranking:** Adds ~0.5s latency per query, acceptable for the precision gain.
    *   **Generation:** ~2-5s per answer on CPU/Standard GPU.

## 6. Failure Analysis & Diagnostics
To understand the persistent low scores despite optimization (Reranking + Top-K=10), I performed a deep-dive diagnostic using a custom script (`debug_rag_context.py`).

**Methodology:**
*   Selected a specific failure case: *"Why does Zuleika reject the Duke?"*
*   Retrieved the raw text context that the system passed to the LLM.
*   Performed keyword analysis on the retrieved text.

**Findings:**
*   **Missing Keywords:** The retrieved context **did not contain** critical terms such as "love", "suicide", "reject", or "refuse".
*   **Irrelevant Content:** The retrieved chunks were semantically distant from the query (e.g., discussing dinner settings or unrelated character dialogue) despite high vector similarity scores.

**Root Cause:**
*   **Semantic Gap:** The embedding model is trained on modern web text, but the book is from 1911. The model does not understand the satirical style or old words.
*   **Conclusion:** The bottleneck is **Dual-Stage**:
    1.  **Retrieval:** The embeddings fail to capture the specific literary nuance, leading to "Garbage In".
    2.  **Generation:** Even if context were perfect, the 1B model struggles with reasoning. But currently, it doesn't even get the chance to reason because the input is irrelevant.

> **Implication:** The problem is the retrieval part, not the generation. Better embeddings would help more than a larger LLM.

## 7. Conclusion
This project delivered a **Production-aligned RAG prototype** and a **Robust Finetuning Infrastructure**. The RAG system highlighted the importance of data quality, while the Finetuning phase stress-tested MLOps on limited hardware. The final codebase supports sustainable training of 1B+ models on commodity hardware.

> **Insight:** Clean data is more important than a big model. If I had more time, I would focus on improving the embeddings first.
