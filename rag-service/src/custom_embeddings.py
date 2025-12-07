"""Custom embedding implementation using transformers library directly."""

import asyncio
import os
import hashlib
import random
import logging
import signal
import time
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Default configuration constants (can be overridden per-instance)
DEFAULT_MAX_BATCH_SIZE = 1000  # Maximum number of texts per batch request
DEFAULT_MAX_TEXT_LENGTH = 100000  # Maximum characters per text (100KB)
DEFAULT_MAX_SEQUENCE_LENGTH = 512  # Maximum tokens for model input
DEFAULT_EMBEDDING_TIMEOUT = 30  # Seconds before embedding times out
DEFAULT_CACHE_MAX_SIZE = 10000  # Maximum cached embeddings
EPSILON = 1e-9  # Small value to prevent division by zero


class EmbeddingTimeoutError(Exception):
    """Raised when an embedding operation exceeds the configured timeout."""
    pass


@contextmanager
def timeout_context(seconds: int):
    """
    Context manager for operation timeout using SIGALRM (Unix only).
    
    Args:
        seconds: Timeout in seconds (0 disables timeout)
        
    Raises:
        EmbeddingTimeoutError: If operation exceeds timeout
    """
    if seconds <= 0 or not hasattr(signal, 'SIGALRM'):
        # No timeout or Windows (no SIGALRM)
        yield
        return
    
    def timeout_handler(signum, frame):
        raise EmbeddingTimeoutError(f"Operation exceeded {seconds} second timeout")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class EmbeddingCache:
    """
    Thread-safe LRU cache for embeddings.
    
    Uses OrderedDict for LRU eviction and threading.Lock for thread safety.
    """
    
    def __init__(self, max_size: int = DEFAULT_CACHE_MAX_SIZE):
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _get_key(self, text: str) -> str:
        """Generate cache key using fast blake2b hash."""
        return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache. Returns None if not found."""
        key = self._get_key(text)
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key].copy()  # Return copy to prevent mutation
            self._misses += 1
            return None
    
    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache with LRU eviction."""
        # Don't cache placeholder embeddings
        if text == "[EMPTY]":
            return
        
        key = self._get_key(text)
        with self._lock:
            if key in self._cache:
                # Update existing and move to end
                self._cache[key] = embedding.copy()
                self._cache.move_to_end(key)
            else:
                # Evict oldest if at capacity
                while len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = embedding.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0
            }
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


class LocalEmbeddings(Embeddings):
    """Custom embedding class using transformers directly."""
    
    def __init__(self, model_path: str = None,
                 device: str = "cpu", normalize: bool = True,
                 allowed_model_base: Optional[str] = None,
                 max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
                 max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
                 max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
                 embedding_timeout_seconds: int = DEFAULT_EMBEDDING_TIMEOUT,
                 cache_max_size: int = DEFAULT_CACHE_MAX_SIZE,
                 enable_cache: bool = True):
        """
        Initialize embeddings.

        Args:
            model_path: Path to the local embedding model (uses EMBEDDING_MODEL_PATH env var if not provided)
            device: Device to run model on ('cpu' or 'cuda')
            normalize: Whether to normalize embeddings
            allowed_model_base: Optional base directory for path validation (security)
            max_batch_size: Maximum texts per embedding request (default: 1000)
            max_text_length: Maximum characters per text (default: 100000)
            max_sequence_length: Maximum tokens for model input (default: 512)
            embedding_timeout_seconds: Timeout for embedding operations, 0 to disable (default: 30)
            cache_max_size: Maximum embeddings to cache (default: 10000)
            enable_cache: Whether to enable embedding caching (default: True)
        """
        # Use environment variable if model_path not provided
        if model_path is None:
            model_path = os.getenv("EMBEDDING_MODEL_PATH")
        
        # SECURITY: Validate model path to prevent path traversal attacks
        model_path = self._validate_model_path(model_path, allowed_model_base)

        self.model_path = model_path
        self.device = device
        self.normalize = normalize
        self.embedding_dim: Optional[int] = None  # Set after warmup
        
        # Configurable limits
        self.max_batch_size = max_batch_size
        self.max_text_length = max_text_length
        self.max_sequence_length = max_sequence_length
        self.embedding_timeout = embedding_timeout_seconds
        
        # Embedding cache
        self._cache: Optional[EmbeddingCache] = EmbeddingCache(cache_max_size) if enable_cache else None
        
        try:
            logger.info(f"Loading embedding model from: {model_path}")

            # Load tokenizer and model
            # SECURITY: trust_remote_code=False prevents arbitrary code execution
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=False,
                local_files_only=True
            )

            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=False,
                torch_dtype=torch.float32,
                local_files_only=True
            ).to(device)

            # Set model to evaluation mode
            self.model.eval()
            
            # Warmup: verify model works and get embedding dimension
            self._warmup()

            logger.info(f"✅ Embedding model loaded successfully on {device}")

        except Exception as e:
            logger.error(f"❌ Error loading embedding model: {e}", exc_info=True)
            raise
    
    def _validate_model_path(self, model_path: str, allowed_base: Optional[str]) -> str:
        """
        Validate model path to prevent path traversal attacks.
        
        Args:
            model_path: Path to validate
            allowed_base: Optional whitelist base directory
            
        Returns:
            Validated absolute path
            
        Raises:
            ValueError: If path is invalid or outside allowed directory
        """
        if not model_path:
            raise ValueError(
                "model_path must be provided or EMBEDDING_MODEL_PATH environment variable must be set"
            )
        
        # Resolve to absolute path (follows symlinks, resolves ..)
        try:
            resolved_path = os.path.realpath(model_path)
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid model path '{model_path}': {e}") from e
        
        # Check existence
        if not os.path.exists(resolved_path):
            raise ValueError(f"Model path does not exist: {resolved_path}")
        
        # If whitelist base is specified, enforce it
        if allowed_base:
            resolved_base = os.path.realpath(allowed_base)
            if not resolved_path.startswith(resolved_base + os.sep) and resolved_path != resolved_base:
                raise ValueError(
                    f"Model path '{model_path}' resolves to '{resolved_path}' "
                    f"which is outside allowed directory '{resolved_base}'"
                )
        
        return resolved_path
    
    def _warmup(self) -> None:
        """
        Warm up the model with a test input to:
        1. Verify the model works correctly
        2. Determine the embedding dimension
        3. Prime any lazy initialization
        """
        logger.info("Warming up embedding model...")
        try:
            test_text = "Model warmup test sentence."
            test_embedding = self._encode_batch([test_text])
            
            if test_embedding.shape[0] != 1:
                raise ValueError(
                    f"Warmup failed: expected 1 embedding, got {test_embedding.shape[0]}"
                )
            
            self.embedding_dim = test_embedding.shape[1]
            logger.info(f"✅ Model warmup successful. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"❌ Model warmup failed: {e}", exc_info=True)
            raise RuntimeError(f"Model failed warmup test: {e}") from e
    
    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check of the embedding model.
        
        Returns:
            Dict with health status, metrics, and diagnostics:
            - healthy: bool - overall health status
            - checks: dict - individual check results
            - timestamp: float - check timestamp
            - error: str (optional) - error message if unhealthy
        """
        health_status: Dict[str, Any] = {
            "healthy": True,
            "checks": {},
            "timestamp": time.time()
        }
        
        try:
            # Check 1: Model loaded
            health_status["checks"]["model_loaded"] = self.model is not None
            
            # Check 2: Embedding dimension set
            health_status["checks"]["embedding_dim_set"] = self.embedding_dim is not None
            health_status["checks"]["embedding_dim"] = self.embedding_dim
            
            # Check 3: GPU availability and memory (if using GPU)
            if self.device != "cpu":
                gpu_available = torch.cuda.is_available()
                health_status["checks"]["gpu_available"] = gpu_available
                if gpu_available:
                    health_status["checks"]["gpu_memory_allocated_gb"] = round(
                        torch.cuda.memory_allocated() / (1024 ** 3), 4
                    )
                    health_status["checks"]["gpu_memory_reserved_gb"] = round(
                        torch.cuda.memory_reserved() / (1024 ** 3), 4
                    )
            
            # Check 4: Test embedding with latency measurement
            start_time = time.time()
            result = self.embed_query("health check test query")
            embedding_time = time.time() - start_time
            
            embedding_valid = len(result) == self.embedding_dim
            health_status["checks"]["embedding_works"] = embedding_valid
            health_status["checks"]["embedding_latency_ms"] = round(embedding_time * 1000, 2)
            
            # Check 5: Cache stats (if enabled)
            if self._cache is not None:
                health_status["checks"]["cache_stats"] = self._cache.get_stats()
            
            # Check 6: Configuration
            health_status["checks"]["config"] = {
                "max_batch_size": self.max_batch_size,
                "max_text_length": self.max_text_length,
                "max_sequence_length": self.max_sequence_length,
                "embedding_timeout": self.embedding_timeout,
                "device": self.device
            }
            
            # Overall health: all boolean checks must pass
            health_status["healthy"] = all(
                v for k, v in health_status["checks"].items()
                if isinstance(v, bool)
            )
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["healthy"] = False
            health_status["error"] = str(e)
            return health_status
    
    def _encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into embeddings with batching to prevent OOM."""
        if not texts:
            # Return empty 2D array with correct embedding dimension for shape consistency
            # This allows safe concatenation with non-empty results
            dim = self.embedding_dim or self.model.config.hidden_size
            return np.empty((0, dim), dtype=np.float32)
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        return np.concatenate(all_embeddings, axis=0)
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a single batch of texts with caching and timeout protection.
        
        Uses try/finally to ensure GPU memory is cleaned up even on exceptions.
        """
        # Check cache for all texts first
        if self._cache is not None:
            cached_results: Dict[int, np.ndarray] = {}
            uncached_texts: List[str] = []
            uncached_indices: List[int] = []
            
            for i, text in enumerate(texts):
                cached = self._cache.get(text)
                if cached is not None:
                    cached_results[i] = cached
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # All cached - return immediately
            if not uncached_texts:
                result = np.stack([cached_results[i] for i in range(len(texts))])
                return result
            
            # Encode only uncached texts
            uncached_embeddings = self._encode_batch_impl(uncached_texts)
            
            # Store in cache
            for text, embedding in zip(uncached_texts, uncached_embeddings):
                self._cache.put(text, embedding)
            
            # Reconstruct result in original order
            dim = uncached_embeddings.shape[1]
            result = np.empty((len(texts), dim), dtype=np.float32)
            uncached_idx = 0
            for i in range(len(texts)):
                if i in cached_results:
                    result[i] = cached_results[i]
                else:
                    result[i] = uncached_embeddings[uncached_idx]
                    uncached_idx += 1
            
            return result
        else:
            # No cache - direct encoding
            return self._encode_batch_impl(texts)
    
    def _encode_batch_impl(self, texts: List[str]) -> np.ndarray:
        """
        Core batch encoding implementation with timeout protection.
        
        Uses try/finally to ensure GPU memory is cleaned up even on exceptions.
        """
        try:
            with timeout_context(self.embedding_timeout):
                with torch.no_grad():
                    try:
                        # Tokenize texts
                        inputs = self.tokenizer(
                            texts,
                            padding=True,
                            truncation=True,
                            max_length=self.max_sequence_length,
                            return_tensors="pt"
                        ).to(self.device)

                        # Get embeddings
                        outputs = self.model(**inputs)

                        # Attention Mask-aware Mean Pooling
                        # This ensures padding tokens are excluded from the average
                        last_hidden_state = outputs.last_hidden_state  # (Batch, Seq, Hidden)
                        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(
                            last_hidden_state.size()
                        ).float()  # (Batch, Seq, Hidden)
                        
                        # Zero out padding positions
                        masked_embeddings = last_hidden_state * attention_mask
                        
                        # Calculate mean by dividing by actual token count (not total sequence length)
                        sum_embeddings = torch.sum(masked_embeddings, 1)  # (Batch, Hidden)
                        sum_mask = torch.clamp(attention_mask.sum(1), min=EPSILON)  # (Batch, Hidden)
                        embeddings = sum_embeddings / sum_mask

                        # Normalize if requested
                        if self.normalize:
                            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                        # Convert to numpy before cleanup
                        result = embeddings.cpu().numpy()
                        return result
                        
                    finally:
                        # Always clean up GPU memory, even on exception
                        if self.device != "cpu" and torch.cuda.is_available():
                            torch.cuda.empty_cache()
        except EmbeddingTimeoutError as e:
            logger.error(f"Embedding timeout after {self.embedding_timeout}s for {len(texts)} texts")
            raise RuntimeError(f"Embedding operation timed out: {e}") from e
    
    def _validate_texts(self, texts: List[str]) -> List[str]:
        """
        Validate and sanitize input texts for embedding.
        
        Args:
            texts: List of texts to validate
            
        Returns:
            List of validated texts (may have placeholders for empty strings)
            
        Raises:
            ValueError: If validation fails
        """
        if texts is None:
            raise ValueError("texts cannot be None")
        
        if not isinstance(texts, list):
            raise ValueError(f"texts must be a list, got {type(texts).__name__}")
        
        batch_size = len(texts)
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds maximum allowed {self.max_batch_size}"
            )
        
        # Early exit for empty batch
        if batch_size == 0:
            return []
        
        validated = []
        for i, text in enumerate(texts):
            if text is None:
                raise ValueError(f"Text at index {i} is None")
            
            if not isinstance(text, str):
                raise ValueError(
                    f"Text at index {i} must be a string, got {type(text).__name__}"
                )
            
            text_len = len(text)
            if text_len > self.max_text_length:
                raise ValueError(
                    f"Text at index {i} exceeds maximum length "
                    f"({text_len} > {self.max_text_length} characters)"
                )
            
            # Handle empty strings gracefully - use isspace() for efficiency (avoids allocation)
            if text_len == 0 or text.isspace():
                logger.debug(f"Empty text at index {i}, using placeholder")
                validated.append("[EMPTY]")
            else:
                validated.append(text)
        
        return validated
    
    def _validate_query(self, text: str) -> str:
        """
        Validate a single query text.
        
        Args:
            text: Query text to validate
            
        Returns:
            Validated text
            
        Raises:
            ValueError: If validation fails
        """
        if text is None:
            raise ValueError("Query text cannot be None")
        
        if not isinstance(text, str):
            raise ValueError(f"Query text must be a string, got {type(text).__name__}")
        
        text_len = len(text)
        if text_len > self.max_text_length:
            raise ValueError(
                f"Query text exceeds maximum length ({text_len} > {self.max_text_length} characters)"
            )
        
        # Use isspace() for efficiency (avoids allocation from strip())
        if text_len == 0 or text.isspace():
            logger.debug("Empty query text, using placeholder")
            return "[EMPTY]"
        
        return text
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of document strings to embed
            
        Returns:
            List of embedding vectors (each a list of floats)
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If embedding fails
        """
        try:
            # Validate inputs
            validated_texts = self._validate_texts(texts)
            
            # Handle empty list case
            if not validated_texts:
                return []
            
            embeddings = self._encode_texts(validated_texts)
            return embeddings.tolist()
            
        except ValueError:
            # Re-raise validation errors without wrapping
            raise
        except Exception as e:
            error_context = {
                "num_texts": len(texts) if texts else 0,
                "model_path": self.model_path,
                "device": self.device,
            }
            logger.error(
                f"❌ Error embedding documents: {e}. Context: {error_context}",
                exc_info=True
            )
            raise RuntimeError(f"Failed to embed {len(texts) if texts else 0} documents") from e

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query string to embed
            
        Returns:
            Embedding vector as a list of floats
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If embedding fails
        """
        try:
            # Validate input
            validated_text = self._validate_query(text)
            
            embedding = self._encode_texts([validated_text])
            return embedding[0].tolist()
            
        except ValueError:
            # Re-raise validation errors without wrapping
            raise
        except Exception as e:
            logger.error(
                f"❌ Error embedding query: {e}. Text length: {len(text) if text else 0}",
                exc_info=True
            )
            raise RuntimeError("Failed to embed query") from e
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents - runs in thread pool to avoid blocking."""
        return await asyncio.to_thread(self.embed_documents, texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query - runs in thread pool to avoid blocking."""
        return await asyncio.to_thread(self.embed_query, text)


class FallbackEmbeddings(Embeddings):
    """Fallback embedding class that uses random but consistent embeddings."""

    def __init__(self, dimension: int = 1024):
        """Initialize fallback embeddings with random vectors."""
        self.dimension = dimension
        logger.warning(f"⚠️ Using fallback random embeddings with dimension {dimension}")
        logger.warning("⚠️ This fallback should only be used when embedding model is unavailable")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for documents using consistent random vectors."""
        embeddings = []
        for text in texts:
            embeddings.append(self._generate_random_embedding(text))
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Return embedding for query using consistent random vector."""
        return self._generate_random_embedding(text)
    
    def _generate_random_embedding(self, text: str) -> List[float]:
        """Generate a deterministic random embedding for a given text.
        
        Uses a local Random instance to avoid thread-safety issues with global state.
        """
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
        # Thread-safe: use local Random instance instead of global random.seed()
        rng = random.Random(seed)
        return [rng.gauss(0, 0.1) for _ in range(self.dimension)]
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        return await asyncio.to_thread(self.embed_documents, texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        return await asyncio.to_thread(self.embed_query, text)


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
            logger.warning(f"⚠️ Model path not provided or does not exist: {model_path}")
            logger.warning("Using fallback embeddings")
            return FallbackEmbeddings()
        
        # Try to load embeddings
        return LocalEmbeddings(model_path=model_path, device=device)

    except Exception as e:
        logger.error(f"⚠️ Failed to load embeddings: {e}", exc_info=True)
        logger.warning("Using fallback embeddings")
        return FallbackEmbeddings()