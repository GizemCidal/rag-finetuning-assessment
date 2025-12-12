# Advanced RAG System for Gutenberg Books

This repository contains an Advanced Retrieval-Augmented Generation (RAG) system designed for resource-constrained environments (e.g., Google Colab Free Tier). It uses hierarchical chunking, a local on-disk vector database (Qdrant), and the `gemma-3-1b-it` LLM.

## Features

*   **Hierarchical Chunking**: Splits text into large "parent" chunks for context and smaller "child" chunks for precise retrieval.
*   **Persistent Vector DB**: Uses **Qdrant** in local/disk mode to simulate persistence and handle larger-than-memory indices.
*   **Local LLM**: Uses `google/gemma-3-1b-it` optimized for consumer hardware.
*   **Dataset**: Automated pipeline for downloading "Dracula" (Project Gutenberg) and evaluating on NarrativeQA.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main experiment script:

```bash
python run_experiment.py
```

This script will automatically:
- **Dataset:**
  - **Source Text:** "The School for Scandal" by Richard Brinsley Sheridan (Project Gutenberg ID 1845).
  - **QA Pairs:** Filtered from the `NarrativeQA` dataset (Test split) specifically for this book (40 pairs found).
3.  Chunk and index the book into Qdrant (`./data/qdrant_db`).
4.  Run the RAG evaluation loop.
5.  Save results to `rag_evaluation_results.csv`.

## Configuration

You can adjust parameters in `src/rag/config.py`:
*   `PARENT_CHUNK_SIZE` / `CHILD_CHUNK_SIZE`: Adjust for granularity.
*   `TOP_K`: Number of chunks to retrieve.

## Project Structure

*   `src/rag/`: Core implementation modules.
    *   `chunking.py`: Hierarchical logic.
    *   `vector_db.py`: Qdrant wrapper.
    *   `retriever.py`: Embedding and parent resolution.
    *   `generator.py`: LLM inference.
*   `run_experiment.py`: Main entry point.
*   `requirements.txt`: Python dependencies.
