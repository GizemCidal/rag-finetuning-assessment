import uuid
from typing import List, Dict

"""
Text chunking module for RAG.

This module provides hierarchical chunking logic, splitting text into larger parent chunks
to preserve context and smaller child chunks for precise retrieval.
"""

class HierarchicalChunker:
    """
    Splits text into parent and child chunks for hierarchical retrieval.
    
    Attributes:
        parent_chunk_size (int): Size of parent chunks.
        child_chunk_size (int): Size of child chunks.
        overlap (int): Overlap between chunks.
    """
    def __init__(self, parent_chunk_size: int = 1000, child_chunk_size: int = 250, overlap: int = 50):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.overlap = overlap

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Splits text into chunks using a sliding window.

        Args:
            text (str): Input text to split.
            chunk_size (int): Target size of each chunk.
            overlap (int): Number of characters to overlap.

        Returns:
            List[str]: List of text chunks.
        """
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
        Processes text into hierarchical chunks (parents and children).

        Args:
            text (str): The full input text.

        Returns:
            Dict: A dictionary containing:
                - "parents": {parent_id: parent_text}
                - "children": [{"child_id": str, "text": str, "parent_id": str}]
        """
        parents = {}
        children = []
        
        # Create Parent Chunks
        parent_texts = self._split_text(text, self.parent_chunk_size, self.overlap)
        
        for p_idx, p_text in enumerate(parent_texts):
            p_id = str(uuid.uuid4())
            parents[p_id] = p_text
            
            # Create Child Chunks from this Parent
            child_texts = self._split_text(p_text, self.child_chunk_size, self.overlap)
            
            for c_text in child_texts:
                c_id = str(uuid.uuid4())
                children.append({
                    "child_id": c_id,
                    "text": c_text,
                    "parent_id": p_id
                })
                
        return {"parents": parents, "children": children}
