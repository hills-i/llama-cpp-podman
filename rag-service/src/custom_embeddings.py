"""Custom embedding implementation using transformers library directly."""

import os
import hashlib
import random
import logging
from typing import List
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class LocalEmbeddings(Embeddings):
    """Custom embedding class using transformers directly."""
    
    def __init__(self, model_path: str = None,
                 device: str = "cpu", normalize: bool = True):
        """
        Initialize embeddings.

        Args:
            model_path: Path to the local embedding model (uses EMBEDDING_MODEL_PATH env var if not provided)
            device: Device to run model on ('cpu' or 'cuda')
            normalize: Whether to normalize embeddings
        """
        # Use environment variable if model_path not provided
        if model_path is None:
            model_path = os.getenv("EMBEDDING_MODEL_PATH")

        self.model_path = model_path
        self.device = device
        self.normalize = normalize
        
        try:
            logger.info(f"Loading embedding model from: {model_path}")

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

            logger.info(f"✅ Embedding model loaded successfully on {device}")

        except Exception as e:
            logger.error(f"❌ Error loading embedding model: {e}", exc_info=True)
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

            # Convert to numpy and ensure tensors are detached from computation graph
            result = embeddings.cpu().numpy()

            # Explicitly delete intermediate tensors to free memory
            del inputs, outputs, embeddings

            return result
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            embeddings = self._encode_texts(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"❌ Error embedding documents: {e}", exc_info=True)
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        try:
            embedding = self._encode_texts([text])
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"❌ Error embedding query: {e}", exc_info=True)
            raise
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        return self.embed_query(text)


class FallbackEmbeddings(Embeddings):
    """Fallback embedding class that uses random but consistent embeddings."""

    def __init__(self, dimension: int = 1024):
        """Initialize fallback embeddings with random vectors."""
        self.dimension = dimension
        logger.warning(f"⚠️ Using fallback random embeddings with dimension {dimension}")
        logger.warning("⚠️ This fallback should only be used when embedding model is unavailable")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for documents using consistent random vectors."""
        # Use random embeddings with text-based variation for consistency
        embeddings = []
        for text in texts:
            # Use SHA256 hash to seed random generator for consistency
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
            random.seed(seed)
            embedding = [random.gauss(0, 0.1) for _ in range(self.dimension)]
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Return embedding for query using consistent random vector."""
        # Use random embedding with text-based variation for consistency
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
        random.seed(seed)
        return [random.gauss(0, 0.1) for _ in range(self.dimension)]
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        return self.embed_query(text)


def create_embeddings(model_path: str = None,
                     device: str = "cpu") -> Embeddings:
    """
    Create embedding instance with fallback.
    
    Args:
        model_path: Path to the local model
        device: Device to run on
        
    Returns:
        Embeddings instance (Embeddings or FallbackEmbeddings)
    """
    try:
        # Check if model path is provided and exists
        if not model_path or not os.path.exists(model_path):
            print(f"⚠️ Model path not provided or does not exist: {model_path}")
            print("Using fallback embeddings")
            return FallbackEmbeddings()
        
        # Try to load embeddings
        return LocalEmbeddings(model_path=model_path, device=device)

    except Exception as e:
        print(f"⚠️ Failed to load embeddings: {e}")
        print("Using fallback embeddings")
        return FallbackEmbeddings()