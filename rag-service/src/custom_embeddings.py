"""Embedding client that calls llama.cpp /v1/embeddings without tiktoken fetches."""

import logging
import os
from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings

from .http_session import create_http_session

logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_API_BASE = "http://embedding-service:8080/v1"
DEFAULT_EMBEDDING_MODEL = "embedding-model"  # Update to your GGUF model name


class LlamaCppEmbeddingClient(Embeddings):
  """Simple HTTP client for llama.cpp embeddings (avoids tiktoken downloads)."""

  def __init__(
    self,
    *,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    timeout_seconds: int = 30,
  ) -> None:
    self.api_base = api_base or os.getenv("EMBEDDING_API_BASE", DEFAULT_EMBEDDING_API_BASE)
    self.model = model or os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
    self.timeout_seconds = timeout_seconds
    self._session = self._create_session()

    logger.info(
      "Using remote embedding service",
      extra={"base_url": self.api_base, "model": self.model},
    )

  @staticmethod
  def _create_session() -> requests.Session:
    return create_http_session()

  def _embed(self, texts: List[str]) -> List[List[float]]:
    if not texts:
      return []

    payload = {
      "model": self.model,
      "input": texts,
    }

    endpoint = f"{self.api_base}/embeddings"
    response = self._session.post(endpoint, json=payload, timeout=self.timeout_seconds)
    response.raise_for_status()
    data = response.json()

    # llama.cpp /v1/embeddings response: {data: [{embedding: [...], index: 0}, ...]}
    embeds = []
    for item in data.get("data", []):
      embeds.append(item.get("embedding", []))
    return embeds

  def embed_documents(self, texts: List[str]) -> List[List[float]]:
    return self._embed(texts)

  def embed_query(self, text: str) -> List[float]:
    result = self._embed([text])
    return result[0] if result else []


def create_embeddings(
  *,
  model_name: Optional[str] = None,
  api_base: Optional[str] = None,
) -> Embeddings:
  """Factory for embedding client without external tokenizer downloads."""

  return LlamaCppEmbeddingClient(api_base=api_base, model=model_name)
