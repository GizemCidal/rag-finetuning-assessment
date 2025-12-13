from sentence_transformers import CrossEncoder
from typing import List, Tuple
from .config import RAGConfig

class Reranker:
    def __init__(self, config: RAGConfig):
        self.config = config
        model_name = getattr(config, 'RERANKER_MODEL_NAME', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        print(f"Loading Reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[str], top_k: int = 5) -> List[str]:
        """
        Reranks a list of documents based on the query.
        Returns the top_k documents.
        """
        if not docs:
            return []
            
        # Prepare pairs for cross-encoder
        pairs = [[query, doc] for doc in docs]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Combine docs with scores
        doc_scores = list(zip(docs, scores))
        
        # Sort by score descending
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k docs
        return [doc for doc, score in doc_scores[:top_k]]
