# Project Troubleshooting Log

This document records the technical challenges encountered during the development of the Advanced RAG System and their respective solutions.

## 1. Environment & Dependencies

### Issue: `ModuleNotFoundError` despite installation
**Problem:** Packages installed via `pip install -r requirements.txt` were not accessible in the running script.
**Cause:** The packages were installed in the system's base Python environment, but the script might have been running in a different context or hitting version conflicts.
**Solution:**
*   Created a dedicated Python virtual environment (`python3 -m venv venv`).
*   Installed dependencies specifically inside this environment (`./venv/bin/pip install ...`).
*   Executed scripts using the virtual environment's Python binary (`./venv/bin/python ...`).

### Issue: Missing `ipywidgets` in Notebook
**Problem:** Jupyter Notebook displayed `Error rendering output item...` instead of progress bars.
**Cause:** The `tqdm` library used for progress bars requires `ipywidgets` to render correctly in Jupyter interfaces.
**Solution:** Added `ipywidgets` to `requirements.txt` and verified installation.

## 2. Data Selection & Availability

### Issue: "Dracula" QA Pairs Missing
**Problem:** The initial plan was to use "Dracula" by Bram Stoker. However, the data loader returned 0 QA pairs.
**Cause:** Inspection of the `NarrativeQA` dataset's **Test split** revealed that while it contains many books, it does NOT contain "Dracula" (ID 345).
**Solution:**
*   Wrote a script (`find_available_books.py`) to scan the Test split for available Project Gutenberg books.
*   identified **"The School for Scandal" (ID 1845)** as having the highest number of questions (40 pairs) in the test set.
*   Updated `config.py` and `README.md` to use this book instead.

## 3. Implementation & Logic Bugs

### Issue: Infinite Loop in Chunker
**Problem:** The `test_local_pipeline.py` script hung indefinitely during the chunking phase.
**Cause:** A logic error in `src/rag/chunking.py`. When `overlap` was calculated, potentially leading to the `start` index not moving forward or moving backward, creating an infinite `while` loop.
**Solution:** Added a safety check to ensure the `start` index always increments by at least 1 in every iteration.

### Issue: Out Of Memory (OOM) / Process Killed
**Problem:** The local test script was killed (Exit Code 137) by the OS.
**Cause:** Processing the entire book and generating embeddings for all chunks simultaneously exhausted the available RAM on the local machine/Colab instance.
**Solution:**
*   Modified the `test_local_pipeline.py` to use a smaller subset of the text (first 10,000 characters) for logic verification.
*   Added `psutil` based resource logging to track memory usage at each step.

### Issue: `AttributeError: 'QdrantClient' object has no attribute 'search'`
**Problem:** The vector database search failed with an attribute error.
**Cause:** The `qdrant-client` library version installed (1.16.2) had deprecated or removed the high-level `.search()` method in favor of a newer API.
**Solution:** Updated `src/rag/vector_db.py` to use the correct `client.query_points()` method.

### Issue: Qdrant File Lock (`RuntimeError` / `BlockingIOError`)
**Problem:** Re-running notebook cells fails with `RuntimeError: Storage folder ... is already accessed` or `BlockingIOError: [Errno 35] Resource temporarily unavailable`.
**Cause:** Qdrant in local, on-disk mode uses file locks to prevent concurrent access. If a kernel is interrupted or a cell is re-run without properly closing the previous client instance, the lock file remains held by the "zombie" process/thread.
**Solution:**
*   **Automatic:** Added a safety check in the notebook to call `vdb.close()` before initializing a new `VectorDBHandler`.
*   **Manual:** Restart the Jupyter Kernel (*Kernel -> Restart*) to force-release all file locks held by the process.

## 4. Metadata & Performance Optimization

### Issue: "School for Scandal" vs "Zuleika Dobson" (Metadata Mismatch)
**Problem:** The `config.py` file incorrectly labeled ID 1845 as "The School for Scandal".
**Cause:** A human error in commenting/titling. The actual content of the file (ID 1845) IS "Zuleika Dobson". The RAG system was working seamlessly, but the results were poor due to typical retrieval challenges, not wrong data.
**Solution:**
*   Verified file content using `head`.
*   Corrected `config.py` comments and filename to `zuleika_dobson.txt`.

### Issue: Low BLEU/ROUGE Scores and "Please provide context"
**Problem:** The RAG system returned BLEU scores near 0.0 and frequently complained about missing context, even though the correct book was loaded.
**Cause:**
1.  **Small `TOP_K`:** Retrieving only 5 chunks (implied small context window) was insufficient to capture dispersed information in a novel.
2.  **Small Chunk Size:** 250 characters is likely too small for a narrative text, splitting sentences or context meaningful for retrieval.
**Solution:**
*   Increased `TOP_K` from 5 to **10**.
*   Doubled Chunk Sizes: Parent (1000 -> **2000**), Child (250 -> **500**).
