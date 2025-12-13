# Advanced RAG System for Gutenberg Books (Zuleika Dobson)

This repository contains an Advanced Retrieval-Augmented Generation (RAG) system designed for resource-constrained environments (e.g., Google Colab Free Tier). It uses hierarchical chunking, a local on-disk vector database (Qdrant), a Re-ranking step, and the `gemma-3-1b-it` LLM.

## Features

*   **Hierarchical Chunking**: Splits text into large "parent" chunks (2000 chars) for context and smaller "child" chunks (500 chars) for precise retrieval.
*   **Re-ranking**: Uses a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to refine retrieval results.
*   **Persistent Vector DB**: Uses **Qdrant** in local/disk mode to simulate persistence.
*   **Local LLM**: Uses `google/gemma-3-1b-it` optimized for consumer hardware.

## Project Structure

*   `rag/`: Core implementation modules.
    *   `chunking.py`: Hierarchical logic.
    *   `vector_db.py`: Qdrant wrapper.
    *   `retriever.py`: Embedding, parent resolution, and Re-ranking logic.
    *   `generator.py`: LLM inference.
    *   `config.py`: Configuration parameters.
*   `scripts/`: Executable scripts.
    *   `test_rag_comparison.py`: Main evaluation script comparing Base vs. Reranked RAG.
    *   `benchmark_baseline.py`: script for zero-shot baseline evaluation.
*   `requirements.txt`: Python dependencies.

## Design Choices & Justifications

### 1. Hierarchical Chunking
**Why?** Standard chunking often cuts off context or retrieves too much noise.
**Strategy:** We index small "child" chunks (granular) for vector search but return their "parent" chunks (broad) to the LLM.
*   **Parent Size:** 2000 characters (capture full paragraphs/dialogue).
*   **Child Size:** 500 characters (dense semantic meaning).

### 2. Embedding Model (`all-MiniLM-L6-v2`)
**Why?** The task requires running on Colab Free Tier.
*   **Size:** Extremely lightweight (~80MB).
*   **Performance:** Consistently ranks high on MTEB benchmarks for its size class.
*   **Efficiency:** Allows fast batch encoding on CPU if GPU is busy with the LLM.

### 3. Vector Database (Qdrant)
**Why?**
*   **Local Persistence:** Unlike in-memory FAISS, Qdrant allows saving/loading from disk (`./data/qdrant_db`), simulating a production environment.
*   **API:** Developer-friendly Python client.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run RAG Comparison (Main Demo)
This script chunks the book, indexes it, and runs a comparison between Standard Vector Search and Re-ranked Search.
```bash
python scripts/test_rag_comparison.py
```

### 2. Run Baseline Benchmark
To evaluate the model's performance without any RAG context:
```bash
python scripts/benchmark_baseline.py
```
