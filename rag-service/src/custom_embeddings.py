"""Custom embedding implementation using transformers library directly."""

import os
from typing import List, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings


class Qwen3Embeddings(Embeddings):
    """Custom embedding class for Qwen3-Embedding-0.6B using transformers directly."""
    
    def __init__(self, model_path: str = "/app/models/embedding/Qwen3-Embedding-0.6B",
                 device: str = "cpu", normalize: bool = True):
        """
        Initialize Qwen3 embeddings.
        
        Args:
            model_path: Path to the local Qwen3-Embedding model
            device: Device to run model on ('cpu' or 'cuda')
            normalize: Whether to normalize embeddings
        """
        self.model_path = model_path
        self.device = device
        self.normalize = normalize
        
        try:
            print(f"Loading Qwen3 embedding model from: {model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                local_files_only=True
            ).to(device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            print(f"✅ Qwen3 embedding model loaded successfully on {device}")
            
        except Exception as e:
            print(f"❌ Error loading Qwen3 embedding model: {e}")
            raise
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        with torch.no_grad():
            # Tokenize texts
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            outputs = self.model(**inputs)
            
            # Use mean pooling over sequence length
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Normalize if requested
            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Convert to numpy
            return embeddings.cpu().numpy()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            embeddings = self._encode_texts(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"❌ Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        try:
            embedding = self._encode_texts([text])
            return embedding[0].tolist()
        except Exception as e:
            print(f"❌ Error embedding query: {e}")
            raise
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        return self.embed_query(text)


class FallbackEmbeddings(Embeddings):
    """Fallback embedding class that returns simple similarity-based embeddings."""
    
    def __init__(self, dimension: int = 384):
        """Initialize fallback embeddings with SentenceTransformer."""
        self.dimension = dimension
        print(f"⚠️ Using fallback SentenceTransformer embeddings with dimension {dimension}")
        
        try:
            from sentence_transformers import SentenceTransformer
            # Use a small, fast model for fallback
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.is_working = True
            print("✅ Loaded SentenceTransformer as fallback")
        except Exception as e:
            print(f"❌ Failed to load SentenceTransformer: {e}")
            print("⚠️ Using random embeddings as last resort")
            self.model = None
            self.is_working = False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for documents."""
        if self.is_working and self.model:
            try:
                embeddings = self.model.encode(texts, convert_to_tensor=False)
                return embeddings.tolist()
            except Exception as e:
                print(f"Error in SentenceTransformer embedding: {e}")
        
        # Fallback to random embeddings with some text-based variation
        import hashlib
        import random
        embeddings = []
        for text in texts:
            # Use text hash to seed random generator for consistency
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            random.seed(seed)
            embedding = [random.gauss(0, 0.1) for _ in range(self.dimension)]
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Return embedding for query."""
        if self.is_working and self.model:
            try:
                embedding = self.model.encode([text], convert_to_tensor=False)
                return embedding[0].tolist()
            except Exception as e:
                print(f"Error in SentenceTransformer query embedding: {e}")
        
        # Fallback to random embedding with text-based variation
        import hashlib
        import random
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        return [random.gauss(0, 0.1) for _ in range(self.dimension)]
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        return self.embed_query(text)


def create_embeddings(model_path: str = "/app/models/embedding/Qwen3-Embedding-0.6B",
                     device: str = "cpu") -> Embeddings:
    """
    Create embedding instance with fallback.
    
    Args:
        model_path: Path to the local model
        device: Device to run on
        
    Returns:
        Embeddings instance (Qwen3Embeddings or FallbackEmbeddings)
    """
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"⚠️ Model path does not exist: {model_path}")
            print("Using fallback embeddings")
            return FallbackEmbeddings()
        
        # Try to load Qwen3 embeddings
        return Qwen3Embeddings(model_path=model_path, device=device)
        
    except Exception as e:
        print(f"⚠️ Failed to load Qwen3 embeddings: {e}")
        print("Using fallback embeddings")
        return FallbackEmbeddings()