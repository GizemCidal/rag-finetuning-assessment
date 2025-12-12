from typing import List, Dict
import uuid

class HierarchicalChunker:
    def __init__(self, parent_chunk_size: int = 1000, child_chunk_size: int = 250, overlap: int = 50):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.overlap = overlap

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple sliding window text splitter."""
        # Clean text basic normalization could go here
        words = text.split() # Splitting by words to be safer than chars for "meaning"
        # However, for strict size limits, char splitting is often used. 
        # Let's use character-based splitting with some heuristics for sentence boundaries if we had time,
        # but for this task, a robust overlapping window or N-word window is good.
        # Let's stick to character count for simplicity but try to respect word boundaries if possible.
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # If we are not at the end, try to backup to the last space to avoid cutting words
            if end < text_len:
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Ensure we make progress to prevent infinite loops
            new_start = end - overlap
            if new_start <= start:
                new_start = start + 1
            start = new_start
            
            if start >= text_len:
                break
        
        return chunks

    def chunk_data(self, text: str) -> Dict:
        """
        Produce a structure:
        {
            "parents": {parent_id: parent_text},
            "children": [
                 {"child_id": cid, "text": ctext, "parent_id": pid}
            ]
        }
        """
        parents = {}
        children = []
        
        # 1. Create Parent Chunks
        parent_texts = self._split_text(text, self.parent_chunk_size, self.overlap)
        
        for p_idx, p_text in enumerate(parent_texts):
            p_id = str(uuid.uuid4())
            parents[p_id] = p_text
            
            # 2. Create Child Chunks from this Parent
            # Note: We are splitting the PARENT text, so the children are guaranteed to be within the parent context.
            child_texts = self._split_text(p_text, self.child_chunk_size, self.overlap)
            
            for c_text in child_texts:
                c_id = str(uuid.uuid4())
                children.append({
                    "child_id": c_id,
                    "text": c_text,
                    "parent_id": p_id
                })
                
        return {"parents": parents, "children": children}
