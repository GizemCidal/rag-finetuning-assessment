# Advanced RAG Task Report

## 1. Introduction
The goal of this project was to build a resource-efficient RAG system to answer questions about the book "Zuleika Dobson" (Project Gutenberg ID 1845) using the NarrativeQA dataset. The system targets constrained environments (Google Colab Free Tier) by employing hierarchical chunking and an on-disk vector database.

## 2. Approach & Methodology

### 1. Data Selection
- **Document:** "Zuleika Dobson; or, An Oxford Love Story" by Max Beerbohm (Project Gutenberg ID 1845).
    *   *Correction Reference:* Initially, the config file mislabeled ID 1845 as "The School for Scandal". Manual verification (`head` command) confirmed the content is indeed "Zuleika Dobson", matching the QA pairs from NarrativeQA.
- **QA Pairs:** Filtered from the `NarrativeQA` dataset. Total pairs found: 40.

### Key Components
*   **Chunking Strategy**: **Hierarchical Chunking**.
    *   *Parent Chunks*: 2000 characters. Providing broader context for the LLM.
    *   *Child Chunks*: 500 characters. Specific segments for dense retrieval.
    *   *Logic*: Retrieved child chunks map back to their parent chunks for context expansion.

## 3. Implementation Details
The solution is implemented in modular Python:

### 3.1 Data Cleaning & Normalization
To ensure high-quality retrieval, we implemented a rigorous cleaning pipeline in `DataLoader`:
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

## 4. Parameter Evolution & Optimization
We iteratively improved the system performance by tuning key parameters.

| Parameter | Initial Value | Final Value | Reason for Change |
| :--- | :--- | :--- | :--- |
| **Parent Chunk Size** | 1000 chars | **2000 chars** | Initial context was too fragmented. Doubling size provided the LLM with complete narrative arcs. |
| **Child Chunk Size** | 250 chars | **500 chars** | 250 chars often split sentences or lost semantic meaning necessary for vector matching. |
| **Top-K Retrieval** | 5 | **10** | Searching only 5 chunks resulted in high "Context Missing" errors. Increasing to 10 improved recall. |
| **Retrieval Strategy** | Vector Search Only | **Vector + Re-ranking** | Vector search alone returned "semantically close but irrelevant" matches. Re-ranking (fetching 30, selecting top 10) significantly improved precision. |

## 5. Results & Discussion

**Baseline RAG Performance (Hierarchical, Top-K=10 + Re-ranking):**
| Approach | BLEU-4 | ROUGE-L | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline (No RAG)** | **0.0033** | **0.0554** | Model has near-zero knowledge of the specific book. Fast inference (<1s). |
| **RAG (Vector Only)** | ~0.02 | ~0.07 | Suffer from retrieval noise. |
| **RAG (Re-ranking) + Cleaning** | **~0.11** | **~0.14** | **Best Case.** Data cleaning doubled the performance (0.08 -> 0.14). |

*> Note: Scores are improved but still limited by the archaic language gap.*

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
To understand the persistent low scores despite optimization (Reranking + Top-K=10), we performed a deep-dive diagnostic using a custom script (`debug_rag_context.py`).

**Methodology:**
*   Selected a specific failure case: *"Why does Zuleika reject the Duke?"*
*   Retrieved the raw text context that the system passed to the LLM.
*   Performed keyword analysis on the retrieved text.

**Findings:**
*   **Missing Keywords:** The retrieved context **did not contain** critical terms such as "love", "suicide", "reject", or "refuse".
*   **Irrelevant Content:** The retrieved chunks were semantically distant from the query (e.g., discussing dinner settings or unrelated character dialogue) despite high vector similarity scores.

**Root Cause:**
*   **Semantic Gap:** The embedding model (`all-MiniLM-L6-v2`) is trained on modern web text/QA pairs. It struggles to map the user's direct question style to the **archaic, satirical, and flowery language** of the 1911 novel "Zuleika Dobson".
*   **Conclusion:** The bottleneck is **Dual-Stage**:
    1.  **Retrieval:** The embeddings fail to capture the specific literary nuance, leading to "Garbage In".
    2.  **Generation:** Even if context were perfect, the 1B model struggles with reasoning. But currently, it doesn't even get the chance to reason because the input is irrelevant.

## 7. Next Steps: PEFT Comparison Study
While domain adaptation for this specific book remains a valid path, the project will now shift focus to a broader study of **Parameter-Efficient Fine-Tuning (PEFT)** techniques.
1.  **Objective:** Compare **QLoRA** vs **GaLore** on the `gemma-3-1b-it` model.
2.  **Dataset:** A mixed instruction-tuning dataset (Alpaca + Tulu + Ultrachat) to improve general instruction following capabilities.
3.  **Challenges:** Training large models on Google Colab Free Tier (T4 GPU) requires aggressive memory optimization (Batch Size=1, Gradient Checkpointing) and frequent checkpointing strategies to handle runtime disconnections (~2.5h limit).

## 8. Conclusion
The implemented system demonstrates that while advanced RAG techniques (Hierarchical Chunking, Qdrant Persistence, Re-ranking) are successfully deployed, they hit a hard ceiling when the *semantic understanding* of the underlying models does not match the domain (Historical Literature). The path forward is now to explore efficient training methods (PEFT) to fundamentally enhance the model's capabilities.
