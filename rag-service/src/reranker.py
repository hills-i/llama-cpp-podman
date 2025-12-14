"""HTTP-based reranker client for llama.cpp reranking server."""

import logging
import os
from typing import Any, Dict, List, Optional

import requests
from langchain_core.documents import Document
from langchain_core.documents import BaseDocumentCompressor
from pydantic import ConfigDict, PrivateAttr

logger = logging.getLogger(__name__)

DEFAULT_RERANK_API_BASE = "http://rerank-service:8080/v1"
DEFAULT_RERANK_MODEL = "reranker-model"  # Update to your GGUF model name


class LlamaCppReranker(BaseDocumentCompressor):
    """Compress documents by delegating scoring to a llama.cpp rerank server."""

    api_base: str
    model_name: str
    top_n: int
    timeout_seconds: int

    _session: requests.Session = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        *,
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        top_n: int = 5,
        timeout_seconds: int = 30,
    ) -> None:
        resolved_api_base = api_base or os.getenv("RERANK_API_BASE", DEFAULT_RERANK_API_BASE)
        resolved_model_name = model_name or os.getenv("RERANK_MODEL_NAME", DEFAULT_RERANK_MODEL)
        resolved_top_n = max(1, int(top_n))
        resolved_timeout = int(timeout_seconds)

        # BaseDocumentCompressor is a Pydantic model in recent LangChain versions.
        # Pass required fields up-front to avoid validation on an empty dict.
        super().__init__(
            api_base=resolved_api_base,
            model_name=resolved_model_name,
            top_n=resolved_top_n,
            timeout_seconds=resolved_timeout,
        )

        self._session = self._create_session()

        logger.info(
            "Using remote reranker",
            extra={"base_url": self.api_base, "model": self.model_name, "top_n": self.top_n},
        )

    def _create_session(self) -> requests.Session:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _post_rerank(self, query: str, documents: List[Document]) -> List[Dict[str, Any]]:
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": min(self.top_n, len(documents)),
        }

        endpoint = f"{self.api_base}/rerank"
        response = self._session.post(endpoint, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        # Accept either {"results": [...]} or {"data": [...]} formats
        return data.get("results") or data.get("data") or []

    def _parse_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parsed: List[Dict[str, Any]] = []
        for item in results:
            # Common keys from llama.cpp / OpenAI-compatible rerank responses
            score = item.get("score") or item.get("relevance_score") or item.get("similarity")
            index = item.get("index")
            if index is None:
                # Fallback when server returns the document string
                continue
            try:
                parsed.append({"index": int(index), "score": float(score) if score is not None else 0.0})
            except (TypeError, ValueError):
                parsed.append({"index": int(index), "score": 0.0})
        return parsed

    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[Any] = None,  # noqa: ARG002
    ) -> List[Document]:
        if not documents:
            return []

        try:
            results = self._post_rerank(query, documents)
            scored = self._parse_results(results)

            if not scored:
                logger.warning("Rerank service returned no scores; falling back to vector order")
                return documents[: self.top_n]

            # Sort by score descending and slice top_n
            scored.sort(key=lambda x: x["score"], reverse=True)
            top_docs: List[Document] = []
            for rank, item in enumerate(scored[: self.top_n]):
                idx = item["index"]
                if 0 <= idx < len(documents):
                    doc = documents[idx]
                    new_doc = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
                    new_doc.metadata.update(
                        {
                            "rerank_score": float(item.get("score", 0.0)),
                            "rerank_position": rank + 1,
                            "reranker_model": self.model_name,
                        }
                    )
                    top_docs.append(new_doc)

            if not top_docs:
                return documents[: self.top_n]

            return top_docs

        except Exception as exc:  # broad to ensure fallback
            logger.error(f"Rerank request failed: {exc}")
            return documents[: self.top_n]

    def is_available(self) -> bool:
        """Lightweight availability probe."""
        try:
            response = self._session.get(f"{self.api_base}/health", timeout=3)
            return response.status_code == 200
        except Exception:
            return False
