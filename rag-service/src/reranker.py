"""Reranker implementation using sentence-transformers."""

import os
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class DocumentReranker:
    """Document reranker for multilingual document reranking using sentence-transformers."""
    
    def __init__(self, model_name: str = None,
                 device: str = "cpu", cache_dir: str = None):
        """
        Initialize reranker using sentence-transformers.

        Args:
            model_name: Path to local model or HuggingFace model name (uses RERANKER_MODEL_PATH env var if not provided)
            device: Device to run model on ('cpu' or 'cuda')
            cache_dir: Directory to cache the model
        """
        # Use environment variable if model_name not provided
        if model_name is None:
            model_name = os.getenv("RERANKER_MODEL_PATH")

        self.model_name = model_name
        self.device = device
        
        # Set cache directory if provided
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            
        try:
            logger.info(f"Loading reranker model: {model_name}")
            self.reranker = CrossEncoder(
                model_name,
                device=device,
                trust_remote_code=True
            )
            logger.info("✅ Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading reranker model: {e}")
            self.reranker = None
    
    def is_available(self) -> bool:
        """Check if reranker is available."""
        return self.reranker is not None
    
    def rerank(self, query: str, documents: List[Document], 
               top_k: int = 4) -> List[Document]:
        """
        Rerank documents using reranker.
        
        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents with scores in metadata
        """
        if not self.is_available():
            logger.warning("Reranker not available, returning original order")
            return documents[:top_k]
        
        if not documents:
            return documents
        
        try:
            # Prepare query-document pairs for reranking
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get reranking scores using sentence-transformers CrossEncoder
            scores = self.reranker.predict(pairs)
            
            # Handle single document case (convert numpy array to list)
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = [scores]
            
            # Combine documents with scores
            doc_scores = list(zip(documents, scores))
            
            # Sort by score (descending)
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_k documents
            reranked_docs = []
            for i, (doc, score) in enumerate(doc_scores[:top_k]):
                # Create a copy to avoid modifying original
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata.copy()
                )
                
                # Add reranking information to metadata
                new_doc.metadata.update({
                    'rerank_score': float(score),
                    'rerank_position': i + 1,
                    'reranker_model': self.model_name
                })
                
                reranked_docs.append(new_doc)

            logger.debug(f"Reranked {len(documents)} docs → top {len(reranked_docs)}")
            return reranked_docs

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback to original order
            return documents[:top_k]
    
    def rerank_with_scores(self, query: str, documents: List[Document], 
                          top_k: int = 4) -> List[Tuple[Document, float]]:
        """
        Rerank documents and return with explicit scores.
        
        Returns:
            List of (document, score) tuples
        """
        reranked_docs = self.rerank(query, documents, top_k)
        
        # Extract scores from metadata
        doc_score_pairs = []
        for doc in reranked_docs:
            score = doc.metadata.get('rerank_score', 0.0)
            doc_score_pairs.append((doc, score))
        
        return doc_score_pairs


class HybridRetriever:
    """Combines vector similarity search with reranking for improved accuracy."""
    
    def __init__(self, vector_store_manager, reranker: DocumentReranker = None,
                 retrieval_k: int = 10):
        """
        Initialize hybrid retriever.

        Args:
            vector_store_manager: Vector store for initial retrieval
            reranker: Reranker instance
            retrieval_k: Number of documents to retrieve before reranking
        """
        self.vector_store_manager = vector_store_manager
        self.reranker = reranker
        self.retrieval_k = retrieval_k
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Hybrid retrieval: vector search + reranking.
        
        Args:
            query: Search query
            k: Final number of documents to return
            
        Returns:
            List of top-k reranked documents
        """
        # Retrieve more documents for reranking
        retrieval_count = max(self.retrieval_k, k * 2)
        
        # Initial vector similarity search
        logger.debug(f"Vector search: retrieving {retrieval_count} candidates")
        documents = self.vector_store_manager.similarity_search(
            query, k=retrieval_count
        )

        if not documents:
            logger.warning("No documents found in vector search")
            return []

        logger.debug(f"Retrieved {len(documents)} documents from vector store")

        # Apply reranking if available
        if self.reranker and self.reranker.is_available():
            logger.debug(f"Reranking {len(documents)} documents")
            final_docs = self.reranker.rerank(query, documents, top_k=k)
        else:
            logger.debug("No reranker available, using vector similarity order")
            final_docs = documents[:k]

        logger.debug(f"Final result: {len(final_docs)} documents")
        return final_docs
    
    def retrieve_with_scores(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Retrieve documents with explicit scores."""
        documents = self.retrieve(query, k)
        
        # Extract scores from metadata or use default
        doc_score_pairs = []
        for doc in documents:
            score = doc.metadata.get('rerank_score') or doc.metadata.get('similarity_score', 0.0)
            doc_score_pairs.append((doc, score))
        
        return doc_score_pairs


def get_reranker_info() -> Dict[str, Any]:
    """Get information about available reranker models."""
    reranker_path = os.getenv("RERANKER_MODEL_PATH", "Local reranker model")
    return {
        "current_model": reranker_path,
        "description": "Multilingual reranker supporting 100+ languages",
        "features": [
            "Cross-encoder architecture",
            "Multilingual support (English, Chinese, Japanese, etc.)",
            "High accuracy for retrieval reranking",
            "Optimized for question-answering tasks"
        ]
    }