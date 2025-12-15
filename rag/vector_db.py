from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import os
from .config import RAGConfig
from typing import List, Dict
"""
Vector Database module.

Manages connections to Qdrant, including creating collections, upserting chunk embeddings,
and searching for vectors.
"""

class VectorDBHandler:
    """
    Manages Qdrant vector database interactions.
    """
    def __init__(self, config: RAGConfig):
        self.config = config
        # Initialize Qdrant Client (Disk mode)
        print(f"Initializing Qdrant at {self.config.QDRANT_PATH}")
        self.client = QdrantClient(path=self.config.QDRANT_PATH)
        self.collection_name = self.config.COLLECTION_NAME
        self.vector_size = self.config.EMBEDDING_DIM

    def create_collection(self):
        """Creates the Qdrant collection if it doesn't exist."""
        collections = self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)
        
        if not exists:
            print(f"Creating collection {self.collection_name}...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, 
                    distance=models.Distance.COSINE
                )
            )
        else:
            print(f"Collection {self.collection_name} already exists.")

    def upsert_chunks(self, children: List[Dict], embeddings: List[List[float]]):
        """
        Upserts child chunks with their embeddings and metadata (parent_id).

        Args:
            children (List[Dict]): List of child chunk dictionaries.
            embeddings (List[List[float]]): Corresponding embeddings for the chunks.
        """
        points = []
        for i, child in enumerate(children):
            points.append(models.PointStruct(
                id=child['child_id'],
                vector=embeddings[i],
                payload={
                    "text": child['text'],
                    "parent_id": child['parent_id']
                }
            ))
        
        # Batch upsert is handled by client, but explicit batching recommended for huge datasets
        # For a single book, one-shot or chunks of 100 is fine.
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        print(f"Upserted {len(points)} points.")

    def search(self, query_vector: List[float], top_k: int = 5):
        """
        Searches for closest child chunks.

        Args:
           query_vector (List[float]): The embedding vector of the query.
           top_k (int): Number of closest points to return.

        Returns:
            list: List of ScoredPoint objects from Qdrant.
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        ).points
        return results

    def index_chunks(self, chunks: Dict, embedding_model):
        """
        Generates embeddings for child chunks and upserts them.
        Args:
           chunks: Output from HierarchicalChunker
           embedding_model: Loaded SentenceTransformer model instance
        """
        children = chunks['children']
        texts = [c['text'] for c in children]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = embedding_model.encode(texts, show_progress_bar=True).tolist()
        
        self.upsert_chunks(children, embeddings)

    def close(self):
        """Closes the Qdrant client connection."""
        if hasattr(self, 'client'):
            self.client.close()
