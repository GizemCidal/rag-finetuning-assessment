# Advanced RAG Task: Hierarchical Retrieval from Literature on Constrained Resources

## 1. Objective

The goal of this task is to build and evaluate a Retrieval-Augmented Generation (RAG) system. You will implement a pipeline that uses a chosen book from Project Gutenberg as the knowledge source, indexed using hierarchical chunking into an on-disk vector database (Milvus Lite or Qdrant), and answers questions from the NarrativeQA dataset using the `google/gemma-3-1b-it` LLM. The evaluation will focus on retrieval quality and generation accuracy under simulated resource constraints (Google Colab free tier).

## 2. Components

*   **Language Model (Generator):** `google/gemma-3-1b-it`
    *   **Link:** [https://huggingface.co/google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
*   **Document Source:** Project Gutenberg ([https://www.gutenberg.org/](https://www.gutenberg.org/))
*   **QA Dataset Source:** NarrativeQA dataset ([https://huggingface.co/datasets/narrativeqa](https://huggingface.co/datasets/narrativeqa))
*   **Vector Database:** Milvus Lite *or* Qdrant (running locally/on-disk)
*   **Embedding Model:** Choose a suitable sentence-transformer model appropriate for the Colab free tier (e.g., `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, or similar). Justify your choice.

## 3. Data Selection & Preparation

1.  **Identify Candidate Books:** Examine the NarrativeQA dataset description or associated files (e.g., often found in their GitHub repository or paper appendices) to identify the list of books/stories it uses for its questions.
2.  **Select ONE Book:** Choose **one** book from the NarrativeQA list that is also available in plain text format (`.txt`) on Project Gutenberg. Document your chosen book and its Project Gutenberg ID/URL.
3.  **Download & Clean:** Download the plain text version of the selected book from Project Gutenberg. Perform necessary cleaning (e.g., remove Gutenberg headers/footers, potentially excessive whitespace or irrelevant metadata).
4.  **Align QA:** Filter the NarrativeQA dataset to retain only the question-answer pairs corresponding to your selected book. Use the **test split** of these filtered pairs for your final evaluation. Document the number of QA pairs available for your chosen book in the test set.

## 4. Indexing Strategy: Hierarchical Chunking

*   **Goal:** Implement a parent/child hierarchical chunking strategy for the selected book's text. This involves creating smaller "child" chunks suitable for precise retrieval and larger "parent" chunks that contain them, providing broader context.
*   **Implementation:**
    *   Define and justify your chosen chunk sizes (e.g., parent chunk size, child chunk size) and any overlap used.
    *   Process the cleaned book text into these hierarchical chunks.
    *   Choose an appropriate embedding model (see Section 2) and generate embeddings for the **child chunks**.
    *   Store the parent chunks, child chunks, their relationships (which child belongs to which parent), and the child chunk embeddings.
*   **Vector Database Setup:**
    *   Set up either Milvus Lite or Qdrant to run locally, storing its data **on disk** (not just in memory) to simulate persistence and potential disk I/O limitations.
    *   Index the **child chunk embeddings** into the vector database, ensuring you can link retrieved child chunks back to their corresponding parent chunk content.
*   **Documentation:** Clearly document your chunking parameters, embedding model choice, and vector database setup in your `README.md` and report.

## 5. RAG Pipeline Implementation

*   **Retrieval Strategy:**
    1.  Given a question from the filtered NarrativeQA test set, generate its embedding using the same embedding model.
    2.  Query the vector database to retrieve the top-k most relevant **child chunks**. Define and justify your choice of `k`.
    3.  Implement logic to potentially retrieve the corresponding **parent chunks** for the top retrieved child chunks to provide expanded context. Describe your specific strategy (e.g., retrieve parent for top-1 child, retrieve parents for top-n children, etc.).
*   **Context Formulation:** Combine the retrieved chunk information (either child chunks directly, parent chunks, or a mix) into a context string.
*   **Prompt Engineering:** Design a prompt template for `gemma-3-1b-it` that incorporates the retrieved context and the original question, instructing the model to answer based *primarily* on the provided context.
*   **Generation:** Use the `gemma-3-1b-it` model to generate an answer based on the formulated prompt and context.

## 6. Evaluation

*   **Metrics:**
    *   **BLEU-4:** Measures n-gram overlap with reference answers.
    *   **ROUGE-L:** Measures longest common subsequence with reference answers.
*   **Procedure:**
    1.  **Baseline (No RAG):** For each question in your filtered NarrativeQA test set, generate an answer using `gemma-3-1b-it` *without* any retrieved context (zero-shot). Evaluate these answers against the ground truth using BLEU-4 and ROUGE-L.
    2.  **RAG System:** For each question, use your implemented RAG pipeline (retrieval + generation) to generate an answer. Evaluate these answers against the ground truth using BLEU-4 and ROUGE-L.
    3.  **Resource Monitoring (Qualitative):** Briefly comment on any resource challenges encountered (e.g., RAM limits during indexing/inference, disk space usage for the vector DB, inference time) particularly relating to the Colab free tier environment. Quantitative tracking (like peak memory) is encouraged if feasible but qualitative description is acceptable given potential variability.
*   **Reporting Table:** Present results clearly:

    | Approach       | BLEU-4 (vs. Ground Truth) | ROUGE-L (vs. Ground Truth) | Notes / Qualitative Resource Impact                                  |
    | :------------- | :------------------------ | :------------------------- | :------------------------------------------------------------------- |
    | Baseline (No RAG) | [Result]                  | [Result]                   | [e.g., Fast inference, low memory]                                   |
    | RAG (Hierarchical) | [Result]                  | [Result]                   | [e.g., Indexing time, VDB disk usage, Inference latency increase] |
    *(Replace [Result] with actual values and provide brief notes)*

## 7. Deliverables

1.  **Code Repository:**
    *   A publicly accessible repository (e.g., GitHub, GitLab).
    *   Well-structured and commented Python code for:
        *   Data selection logic (identifying the book).
        *   Document cleaning and preprocessing.
        *   Hierarchical chunking implementation.
        *   Embedding generation and indexing script (for Milvus/Qdrant).
        *   The complete RAG pipeline (retrieval, context assembly, prompting, generation).
        *   Evaluation script (calculating BLEU/ROUGE).
    *   A `requirements.txt` file.
    *   A detailed `README.md` file including:
        *   Project overview, chosen book, and objectives.
        *   Setup instructions (Python env, Vector DB setup, downloading models/data).
        *   Step-by-step guide to run: preprocessing, indexing, RAG evaluation, baseline evaluation.
        *   Clear description of the hierarchical chunking strategy, embedding model choice, and retrieval logic (k, parent retrieval).
        *   Summary of the results (can reference the report).
        *   Link to the final report.

2.  **Report (Max 2 pages):**
    *   Format: PDF or Markdown.
    *   Content:
        *   **Introduction:** State the goal, chosen book, and scope.
        *   **Approach & Methodology:** Detail the selected book, preprocessing steps, chosen embedding model (with justification), hierarchical chunking parameters (with justification), vector database choice, retrieval strategy (k, parent context usage), and prompt design.
        *   **Implementation Details:** Briefly mention key libraries. Highlight challenges (e.g., Vector DB setup, resource constraints, chunking complexity) and how they were managed.
        *   **Results & Discussion:** Present the evaluation table. Analyze the difference between the baseline and RAG performance. Discuss the effectiveness of the hierarchical RAG approach based on the metrics and potentially qualitative examples (optional: include 1-2 good/bad examples). Comment on resource usage observations.
        *   **Conclusion:** Summarize key findings, the viability of this advanced RAG approach on constrained hardware, and potential areas for improvement (e.g., different chunking, better embedding models if resources allowed, re-ranking).

## 8. Execution Notes & Tips

*   **Resource Limits:** The Colab free tier has strict RAM, disk, and runtime limits. Efficient coding, careful library choices, and potentially saving intermediate states are crucial. Indexing a full book might be slow or memory-intensive; consider processing in batches if needed.
*   **Vector DB Choice:** Milvus Lite and local Qdrant are designed to be easier to set up than their full server counterparts. Consult their documentation for installation and usage within a notebook environment. Ensure data is persisted to disk.
*   **NarrativeQA Data:** Accessing and filtering NarrativeQA might require using the `datasets` library. Pay close attention to its structure to correctly isolate QA pairs for your chosen book.
*   **Chunking Complexity:** Hierarchical chunking adds complexity compared to simple fixed-size chunks. Ensure your logic correctly links children to parents.
*   **Reproducibility:** Set random seeds where applicable (though less critical in RAG retrieval itself compared to training). Focus on documenting parameters (chunk sizes, k, model names) clearly.
