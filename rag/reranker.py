from sentence_transformers import CrossEncoder
from typing import List, Tuple
from .config import RAGConfig
"""
Reranking module.

Uses a CrossEncoder model to re-score and re-order a list of retrieved documents
given a query, improving relevance of the final context.
"""

class Reranker:
    """
    Reranks retrieved documents using a CrossEncoder model.
    """

    def __init__(self, config: RAGConfig):
        """
        Initializes the Reranker.

        Args:
            config: RAG configuration object.
        """
        self.config = config
        model_name = getattr(config, 'RERANKER_MODEL_NAME', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        print(f"Loading Reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[str], top_k: int = 5) -> List[str]:
        """
        Reranks a list of documents based on the query.

        Args:
            query (str): The search query.
            docs (List[str]): List of candidate document strings.
            top_k (int): Number of top documents to return.

        Returns:
            List[str]: The top_k documents sorted by relevance.
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
