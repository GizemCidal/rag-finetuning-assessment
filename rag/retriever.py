from sentence_transformers import SentenceTransformer
from typing import List, Dict
from .config import RAGConfig
from .vector_db import VectorDBHandler
from .reranker import Reranker

class HierarchicalRetriever:
    def __init__(self, config: RAGConfig, vector_db: VectorDBHandler, parents_map: Dict[str, str], embedding_model):
        self.config = config
        self.vector_db = vector_db
        self.parents_map = parents_map # In-memory map of {parent_id: parent_text}
        
        # Use shared embedding model
        self.encoder = embedding_model
        
        # Initialize Reranker
        self.reranker = Reranker(config)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings."""
        return self.encoder.encode(texts).tolist()

    def retrieve_context(self, query: str, top_k: int = 5, use_reranker: bool = True) -> str:
        """
        Retrieves context for a query.
        Strategy:
        1. Embed Query
        2. Search Top-K * 3 Child Chunks (fetch more candidates)
        3. Identify unique Parent IDs from children
        4. Retrieve Parent Texts
        5. (Optional) Rerank Parent Texts
        6. Return Top-K Parents
        """
        query_vector = self.encode([query])[0]
        
        # Fetch more candidates for re-ranking (e.g., 3x)
        search_limit = top_k * 3 if use_reranker else top_k
        search_results = self.vector_db.search(query_vector, top_k=search_limit)
        
        # Extract unique parent IDs to avoid duplicate context
        unique_parent_ids = []
        seen_parents = set()
        
        for sc in search_results:
            pid = sc.payload['parent_id']
            if pid not in seen_parents:
                seen_parents.add(pid)
                unique_parent_ids.append(pid)
        
        # Retrieve parent texts
        candidate_parents = []
        for pid in unique_parent_ids:
            if pid in self.parents_map:
                candidate_parents.append(self.parents_map[pid])
        
        # Rerank if enabled
        if use_reranker and candidate_parents:
            # Rerank the parent chunks directly against the query
            final_docs = self.reranker.rerank(query, candidate_parents, top_k=top_k)
        else:
            # Just take the top_k found
            final_docs = candidate_parents[:top_k]
                
        # Join parents with some separator
        full_context = "\n\n".join(final_docs)
        return full_context
