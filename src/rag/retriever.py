from sentence_transformers import SentenceTransformer
from typing import List, Dict
from .config import RAGConfig
from .vector_db import VectorDBHandler

class HierarchicalRetriever:
    def __init__(self, config: RAGConfig, vector_db: VectorDBHandler, parents_map: Dict[str, str]):
        self.config = config
        self.vector_db = vector_db
        self.parents_map = parents_map # In-memory map of {parent_id: parent_text}
        
        print(f"Loading embedding model: {self.config.EMBEDDING_MODEL_NAME}")
        self.encoder = SentenceTransformer(self.config.EMBEDDING_MODEL_NAME)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings."""
        return self.encoder.encode(texts).tolist()

    def retrieve(self, query: str) -> str:
        """
        Retrieves context for a query.
        Strategy:
        1. Embed Query
        2. Search Top-K Child Chunks
        3. Identify unique Parent IDs from children
        4. Return concatenated Parent Texts as context
        """
        query_vector = self.encode([query])[0]
        search_results = self.vector_db.search(query_vector, top_k=self.config.TOP_K)
        
        # Extract unique parent IDs to avoid duplicate context
        unique_parent_ids = []
        seen_parents = set()
        
        for sc in search_results:
            pid = sc.payload['parent_id']
            if pid not in seen_parents:
                seen_parents.add(pid)
                unique_parent_ids.append(pid)
        
        # Retrieve parent texts
        context_parts = []
        for pid in unique_parent_ids:
            if pid in self.parents_map:
                context_parts.append(self.parents_map[pid])
            else:
                # Fallback if parent not found (shouldn't happen)
                pass
                
        # Join parents with some separator
        full_context = "\n\n".join(context_parts)
        return full_context
