"""RAG chain implementation using LangChain and llama.cpp."""

import os
import uuid
from collections import deque
from typing import List, Dict, Any, Optional
import requests
import time
import logging
from threading import Lock
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from .vector_store import VectorStoreManager
from .reranker import LlamaCppReranker

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_requests: int = 10, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self._requests: deque = deque()
        self._lock = Lock()

    def allow_request(self) -> bool:
        """Check if request is allowed under rate limit."""
        with self._lock:
            now = time.time()
            cutoff = now - self.time_window
            while self._requests and self._requests[0] < cutoff:
                self._requests.popleft()

            if len(self._requests) < self.max_requests:
                self._requests.append(now)
                return True
            return False

    def get_retry_after(self) -> int:
        """Get seconds until rate limit resets."""
        with self._lock:
            if not self._requests:
                return 0
            return max(0, int(self.time_window - (time.time() - self._requests[0])))


def create_llm(base_url: str) -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for llama.cpp server."""
    return ChatOpenAI(
        base_url=f"{base_url}/v1",
        api_key="not-needed",
        model=os.getenv("LLM_MODEL", "default"),
        temperature=0.3,
        top_p=0.8,
        max_tokens=1024,
        stop=[],
        extra_body={
            "repeat_penalty": 1.1,
            "top_k": 40,
        },
        timeout=180,
    )


class RAGChain:
    """RAG chain that combines document retrieval with reranking and text generation."""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llama_base_url: Optional[str] = None,
        use_reranker: bool = True,
        reranker_model: str = None,
        rate_limit_requests: int = 10,
        rate_limit_window: int = 60,
    ):
        self.vector_store_manager = vector_store_manager
        self.llama_base_url = llama_base_url or os.getenv("LLAMA_CPP_BASE_URL", "http://llama-cpp-server:11434")
        self.llm = create_llm(self.llama_base_url)
        self.use_reranker = use_reranker

        # Retrieval sizing
        self.initial_retrieval_k = int(os.getenv("RETRIEVAL_CANDIDATES", "50"))
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "5"))

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_requests=rate_limit_requests,
            time_window=rate_limit_window,
        )

        # Base retriever always available
        self.base_retriever = self.vector_store_manager.get_retriever(
            search_kwargs={"k": self.initial_retrieval_k}
        )

        # Initialize reranker and contextual compression
        self.reranker = None
        self.compression_retriever: Optional[ContextualCompressionRetriever] = None

        if use_reranker:
            try:
                logger.info("Initializing HTTP reranker...")
                self.reranker = LlamaCppReranker(
                    api_base=os.getenv("RERANK_API_BASE"),
                    model_name=reranker_model or os.getenv("RERANK_MODEL_NAME"),
                    top_n=self.rerank_top_k,
                )

                if self.reranker.is_available():
                    self.compression_retriever = ContextualCompressionRetriever(
                        base_compressor=self.reranker,
                        base_retriever=self.base_retriever,
                    )
                    logger.info("✅ Contextual compression retriever enabled (vector → rerank)")
                else:
                    logger.warning("Rerank service unavailable; falling back to vector search")
                    self.reranker = None
            except Exception as e:
                logger.error(f"Error initializing reranker: {e}")
                self.reranker = None
        else:
            logger.info("Using simple vector retrieval (no reranking)")

        # Define the prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""以下のコンテキスト情報のみを使って質問に回答してください。コンテキストに答えがない場合は「情報が見つかりませんでした」と回答してください。
回答は直接的に、簡潔に記述してください。思考過程や計画は出力しないでください。

コンテキスト:
{context}

質問: {question}

回答:"""
        )

    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """Query the RAG system with hybrid retrieval."""
        
        # Check rate limit
        if not self.rate_limiter.allow_request():
            retry_after = self.rate_limiter.get_retry_after()
            return {
                "answer": f"Rate limit exceeded. Please try again in {retry_after} seconds.",
                "question": question if isinstance(question, str) else "",
                "sources": [],
                "error": "rate_limited",
                "retry_after": retry_after
            }

        # Validate and sanitize input
        if not question or not isinstance(question, str):
            return {
                "answer": "Invalid question format. Please provide a text question.",
                "question": question if isinstance(question, str) else "",
                "sources": [],
                "error": "validation_failed"
            }

        # Enforce length limits
        MAX_QUESTION_LENGTH = 2000
        question = question.strip()

        if len(question) > MAX_QUESTION_LENGTH:
            return {
                "answer": f"Question too long. Maximum {MAX_QUESTION_LENGTH} characters allowed.",
                "question": question,
                "sources": [],
                "error": "question_too_long"
            }

        if len(question) < 3:
            return {
                "answer": "Question too short. Please provide more details (minimum 3 characters).",
                "question": question,
                "sources": [],
                "error": "question_too_short"
            }

        # Sanitize control characters
        question = ''.join(char for char in question if char.isprintable() or char.isspace())

        try:
            # RETRIEVAL STEP
            retrieval_method = "vector_only"

            if self.use_reranker and self.compression_retriever:
                logger.debug("Using contextual compression retriever (vector → rerank)")
                retrieval_method = "vector_rerank"
                # LangChain retrievers are Runnables; prefer invoke() for compatibility.
                retrieved_docs = self.compression_retriever.invoke(question)
            else:
                logger.debug("Using standard vector retrieval")
                retrieved_docs = self.vector_store_manager.similarity_search(
                    question, k=self.rerank_top_k
                )
            
            if not retrieved_docs:
                return {
                    "answer": "No relevant documents found in the knowledge base.",
                    "sources": [],
                    "question": question
                }
            
            # CONTEXT PREPARATION
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # GENERATION STEP
            prompt = self.prompt_template.format(context=context, question=question)
            llm_response = self.llm.invoke(prompt)
            answer = llm_response.content.strip()
            
            # RESPONSE FORMATTING
            response = {
                "answer": answer,
                "question": question,
                "retrieval_method": retrieval_method
            }
            
            # Add source information
            if include_sources:
                sources = []
                for i, doc in enumerate(retrieved_docs):
                    source_info = {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("source", "Unknown"),
                        "retrieval_rank": i + 1
                    }
                    
                    # Add reranking score if available
                    if "rerank_score" in doc.metadata:
                        source_info["rerank_score"] = doc.metadata["rerank_score"]
                    
                    sources.append(source_info)
                
                response["sources"] = sources
            
            return response
            
        except Exception as e:
            error_id = str(uuid.uuid4())[:8]
            logger.error("Error during RAG query", extra={"error_id": error_id}, exc_info=True)
            return {
                "answer": "Error processing query. Please try again later.",
                "sources": [],
                "error": "internal_error",
                "error_id": error_id,
                "question": question
            }
    
    def simple_retrieval(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Simple document retrieval without generation, with optional reranking."""
        try:
            if self.use_reranker and self.compression_retriever:
                logger.debug("Using hybrid retrieval for search")
                # LangChain retrievers are Runnables; prefer invoke() for compatibility.
                documents = self.compression_retriever.invoke(query)

                results = []
                for i, doc in enumerate(documents):
                    result = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("source", "Unknown"),
                        "retrieval_rank": i + 1,
                        "retrieval_method": "vector_rerank",
                    }

                    if "rerank_score" in doc.metadata:
                        result["rerank_score"] = doc.metadata["rerank_score"]

                    results.append(result)

                return results

            logger.debug("Using vector similarity search")
            documents = self.vector_store_manager.similarity_search_with_score(query, k=k)
            logger.debug(f"Retrieved {len(documents)} documents from vector store")

            results = []
            for i, (doc, score) in enumerate(documents):
                if float(score) == 0.0:
                    similarity_percentage = max(50, 90 - (i * 10))
                else:
                    similarity_percentage = max(0, min(100, (1 - float(score) / 2) * 100))

                results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": similarity_percentage,
                        "source": doc.metadata.get("source", "Unknown"),
                        "retrieval_rank": i + 1,
                        "retrieval_method": "vector_only",
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error during simple retrieval: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of the RAG system components including reranker."""
        status = {
            "vector_store": self.vector_store_manager.get_collection_info(),
            "llm_available": False,
            "chain_ready": True, # Changed to True as we use manual logic
            "reranker_enabled": self.use_reranker,
            "reranker_available": False,
            "embedding_model": os.getenv("EMBEDDING_MODEL_NAME", "Remote embedding (llama.cpp)")
        }
        
        # Test LLM availability
        try:
            test_response = requests.get(f"{self.llama_base_url}/health", timeout=5)
            status["llm_available"] = test_response.status_code == 200
        except requests.exceptions.RequestException:
            status["llm_available"] = False
        
        if self.use_reranker and self.reranker:
            status["reranker_available"] = self.reranker.is_available()
            status["reranker_model"] = self.reranker.model_name if status["reranker_available"] else "Not reachable"
        else:
            status["reranker_model"] = "Disabled"
        
        return status
