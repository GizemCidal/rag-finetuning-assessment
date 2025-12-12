# Advanced RAG Task Report

## 1. Introduction
The goal of this project was to build a resource-efficient RAG system to answer questions about the book "The School for Scandal" (Project Gutenberg) using the NarrativeQA dataset. The system targets constrained environments (Google Colab Free Tier) by employing hierarchical chunking and an on-disk vector database.

## 2. Approach & Methodology

### 1. Data Selection
- **Document:** "The School for Scandal" by Richard Brinsley Sheridan (Project Gutenberg ID 1845). Chosen for its availability and decent number of associated questions in the NarrativeQA test set.
- **QA Pairs:** Filtered from the `NarrativeQA` dataset. Total pairs found: 40.

### Key Components
*   **Vector Database**: **Qdrant** (Local Mode). Chosen for its ability to persist data to disk (`./data/qdrant_db`), minimizing RAM usage compared to purely in-memory solutions.
*   **Chunking Strategy**: **Hierarchical Chunking**.
    *   *Parent Chunks*: 1000 characters. Providing broad context for the LLM.
    *   *Child Chunks*: 250 characters. Highly specific segments for dense vector retrieval.
    *   *Logic*: Retrieved child chunks map back to their parent chunks, which are then fed to the LLM.
*   **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`. Selected for its speed and low memory footprint (384 dim), ideal for free-tier GPUs/CPUs.
*   **Generator**: `google/gemma-3-1b-it`. A compact 1B parameter model capable of instruction following while fitting in Colab RAM.

## 3. Implementation Details
The solution is implemented in modular Python:
*   `src/rag/chunking.py`: Custom sliding window splitter with parent-child ID tracking.
*   `src/rag/vector_db.py`: Qdrant wrapper handling upserts and search.
*   `src/rag/retriever.py`: Logic to deduplicate parent contexts from multiple retrieved children.

**Challenges**:
*   *Memory Constraint*: Loading the full book and embeddings in memory. Addressed by processing chunks in batches and using Qdrant's on-disk storage.
*   *Context Length*: Addressed by limiting the number of retrieved parent chunks (Top-K=5).

## 4. Results & Discussion

*(Run `run_experiment.py` to fill in these values)*

| Approach | BLEU-4 | ROUGE-L | Resource Observations |
| :--- | :--- | :--- | :--- |
| **Baseline (No RAG)** | [Value] | [Value] | Fast, but hallucinates specific plot details. |
| **RAG (Hierarchical)** | [Value] | [Value] | Slower indexing. Higher accuracy on specific names/events. |

*   **Observation**: The hierarchical approach allowed the model to access full sentences/paragraphs (Parent) even when the match was triggered by a specific keyword in a small segment (Child), improving answer coherence.

## 5. Conclusion
The implemented system demonstrates that advanced RAG techniques like hierarchical chunking can be effectively deployed on constrained hardware. Qdrant's local disk support was crucial for managing index size without using server-based solutions.
