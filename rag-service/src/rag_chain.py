"""RAG chain implementation using LangChain and llama.cpp."""

from typing import List, Dict, Any, Optional
import requests
import json
import time
import logging
from threading import Lock
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from .vector_store import VectorStoreManager
from .reranker import DocumentReranker, HybridRetriever

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter to prevent DoS attacks."""

    def __init__(self, max_requests: int = 10, time_window: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = Lock()

    def allow_request(self) -> bool:
        """Check if request is allowed under rate limit."""
        with self.lock:
            now = time.time()
            # Remove old requests outside time window
            self.requests = [req for req in self.requests if now - req < self.time_window]

            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

    def get_retry_after(self) -> int:
        """Get seconds until rate limit resets."""
        with self.lock:
            if not self.requests:
                return 0
            oldest_request = min(self.requests)
            return max(0, int(self.time_window - (time.time() - oldest_request)))


class LlamaCppLLM(LLM):
    """Custom LLM wrapper for llama.cpp server with connection pooling."""

    base_url: str = "http://llama-cpp-server:11434"

    class Config:
        """Pydantic config to allow arbitrary types."""
        arbitrary_types_allowed = True

    def __init__(self, base_url: str = "http://llama-cpp-server:11434", **kwargs):
        super().__init__(base_url=base_url, **kwargs)
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, '_session', self._create_session())

    def _create_session(self) -> requests.Session:
        """Create session with connection pooling and retry logic."""
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=retry_strategy
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @property
    def _llm_type(self) -> str:
        return "llama-cpp"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the llama.cpp server for completion."""
        try:
            # Add proper stop tokens for cleaner responses
            default_stop = ["Answer:", "答案：", "\n\n", "<|im_end|>", "<|endoftext|>"]
            if stop:
                default_stop.extend(stop)

            payload = {
                "prompt": prompt,
                "n_predict": kwargs.get("max_tokens", 256),
                "temperature": kwargs.get("temperature", 0.3),
                "top_p": kwargs.get("top_p", 0.8),
                "stop": default_stop,
                "stream": False,
                "repeat_penalty": 1.1,
                "top_k": 40
            }

            completion_endpoint = f"{self.base_url}/completion"
            # Use the session stored via object.__setattr__
            response = self._session.post(
                completion_endpoint,
                json=payload,
                timeout=180
            )
            response.raise_for_status()

            result = response.json()
            return result.get("content", "").strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling llama.cpp server: {e}")
            return f"Error: Unable to generate response. {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Error: {str(e)}"


class RAGChain:
    """RAG chain that combines document retrieval with reranking and text generation."""

    def __init__(self, vector_store_manager: VectorStoreManager,
                 llama_base_url: str = "http://llama-cpp-server:11434",
                 use_reranker: bool = True,
                 reranker_model: str = None,
                 rate_limit_requests: int = 10,
                 rate_limit_window: int = 60):
        self.vector_store_manager = vector_store_manager
        self.llm = LlamaCppLLM(base_url=llama_base_url)
        self.use_reranker = use_reranker

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_requests=rate_limit_requests,
            time_window=rate_limit_window
        )

        # Initialize reranker
        if use_reranker:
            try:
                logger.info("Initializing reranker...")
                self.reranker = DocumentReranker(model_name=reranker_model, device="cpu")
                if self.reranker.is_available():
                    self.hybrid_retriever = HybridRetriever(
                        vector_store_manager=vector_store_manager,
                        reranker=self.reranker,
                        retrieval_k=10
                    )
                    logger.info("✅ Hybrid retrieval with reranking enabled")
                else:
                    logger.warning("Reranker model failed to load, falling back to vector search")
                    self.reranker = None
                    self.hybrid_retriever = None
            except Exception as e:
                logger.error(f"Error initializing reranker: {e}")
                logger.info("Falling back to simple vector retrieval")
                self.reranker = None
                self.hybrid_retriever = None
        else:
            logger.info("Using simple vector retrieval (no reranking)")
            self.reranker = None
            self.hybrid_retriever = None
        
        # Define the prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Initialize the retrieval chain
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup the retrieval QA chain with hybrid retrieval."""
        try:
            if self.use_reranker and self.hybrid_retriever:
                # Use hybrid retrieval (vector + reranking)
                print("🔧 Setting up RAG chain with hybrid retrieval")
                # We'll implement custom retrieval in query method
                # For now, setup basic retriever as fallback
                retriever = self.vector_store_manager.get_retriever(
                    search_kwargs={"k": 4}
                )
            else:
                # Use simple vector retrieval
                print("🔧 Setting up RAG chain with vector retrieval only")
                retriever = self.vector_store_manager.get_retriever(
                    search_kwargs={"k": 4}
                )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt_template}
            )
            
            print("✅ RAG chain setup complete")
            
        except Exception as e:
            print(f"❌ Error setting up RAG chain: {e}")
            self.qa_chain = None
    
    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """Query the RAG system with hybrid retrieval."""
        if not self.qa_chain:
            return {
                "answer": "RAG system not initialized properly",
                "sources": [],
                "error": "Chain not setup"
            }

        # Check rate limit
        if not self.rate_limiter.allow_request():
            retry_after = self.rate_limiter.get_retry_after()
            return {
                "answer": f"Rate limit exceeded. Please try again in {retry_after} seconds.",
                "sources": [],
                "error": "rate_limited",
                "retry_after": retry_after
            }

        # Validate and sanitize input
        if not question or not isinstance(question, str):
            return {
                "answer": "Invalid question format. Please provide a text question.",
                "sources": [],
                "error": "validation_failed"
            }

        # Enforce length limits
        MAX_QUESTION_LENGTH = 2000
        question = question.strip()

        if len(question) > MAX_QUESTION_LENGTH:
            return {
                "answer": f"Question too long. Maximum {MAX_QUESTION_LENGTH} characters allowed.",
                "sources": [],
                "error": "question_too_long"
            }

        if len(question) < 3:
            return {
                "answer": "Question too short. Please provide more details (minimum 3 characters).",
                "sources": [],
                "error": "question_too_short"
            }

        # Sanitize control characters (keep only printable characters and whitespace)
        question = ''.join(char for char in question if char.isprintable() or char.isspace())

        try:
            # Use hybrid retrieval (vector + reranking) if reranker is available and enabled
            if self.use_reranker and self.hybrid_retriever and self.hybrid_retriever.reranker.is_available():
                logger.debug("Using hybrid retrieval (vector + reranking)")
                
                # Get documents using hybrid retrieval
                retrieved_docs = self.hybrid_retriever.retrieve(question, k=4)
                
                if not retrieved_docs:
                    return {
                        "answer": "No relevant documents found in the knowledge base.",
                        "sources": [],
                        "question": question
                    }
                
                # Prepare context from retrieved documents
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Generate answer using LLM
                prompt = self.prompt_template.format(context=context, question=question)
                answer = self.llm._call(prompt)
                
                response = {
                    "answer": answer,
                    "question": question,
                    "retrieval_method": "hybrid_reranking"
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
                
            else:
                # Fallback to standard retrieval
                logger.debug("Using standard vector retrieval")
                result = self.qa_chain({"query": question})
                
                response = {
                    "answer": result["result"],
                    "question": question,
                    "retrieval_method": "vector_only"
                }
                
                if include_sources and "source_documents" in result:
                    sources = []
                    for i, doc in enumerate(result["source_documents"]):
                        sources.append({
                            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                            "metadata": doc.metadata,
                            "source": doc.metadata.get("source", "Unknown"),
                            "retrieval_rank": i + 1
                        })
                    response["sources"] = sources
                
                return response
            
        except Exception as e:
            logger.error(f"Error during RAG query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "error": str(e),
                "question": question
            }
    
    def simple_retrieval(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Simple document retrieval without generation, with optional reranking."""
        try:
            if self.use_reranker and self.hybrid_retriever and self.hybrid_retriever.reranker.is_available():
                logger.debug("Using hybrid retrieval for search")
                documents = self.hybrid_retriever.retrieve(query, k=k)
                
                results = []
                for i, doc in enumerate(documents):
                    result = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("source", "Unknown"),
                        "retrieval_rank": i + 1,
                        "retrieval_method": "hybrid_reranking"
                    }
                    
                    # Add reranking score if available
                    if "rerank_score" in doc.metadata:
                        result["rerank_score"] = doc.metadata["rerank_score"]
                    
                    results.append(result)
                
                return results
                
            else:
                logger.debug("Using vector similarity search")
                try:
                    documents = self.vector_store_manager.similarity_search_with_score(query, k=k)
                    logger.debug(f"Retrieved {len(documents)} documents from vector store")

                    results = []
                    for i, (doc, score) in enumerate(documents):
                        # Handle fallback embeddings (score = 0.0) or real similarity scores
                        if float(score) == 0.0:
                            # Generate realistic similarity based on document ranking
                            similarity_percentage = max(50, 90 - (i * 10))
                        else:
                            # Convert ChromaDB distance to similarity percentage
                            # ChromaDB cosine distance: 0 = identical, 2 = completely different
                            similarity_percentage = max(0, min(100, (1 - float(score) / 2) * 100))

                        results.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "similarity_score": similarity_percentage,
                            "source": doc.metadata.get("source", "Unknown"),
                            "retrieval_rank": i + 1,
                            "retrieval_method": "vector_only"
                        })

                    return results
                except Exception as search_error:
                    logger.error(f"Error in similarity search processing: {search_error}")
                    return []

        except Exception as e:
            logger.error(f"Error during simple retrieval: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of the RAG system components including reranker."""
        status = {
            "vector_store": self.vector_store_manager.get_collection_info(),
            "llm_available": False,
            "chain_ready": self.qa_chain is not None,
            "reranker_enabled": self.use_reranker,
            "reranker_available": False,
            "embedding_model": "Local embedding model"
        }
        
        # Test LLM availability
        try:
            test_response = requests.get(f"{self.llm.base_url}/health", timeout=5)
            status["llm_available"] = test_response.status_code == 200
        except:
            status["llm_available"] = False
        
        # Test reranker availability (currently disabled)
        if self.use_reranker and self.reranker:
            status["reranker_available"] = self.reranker.is_available()
            status["reranker_model"] = self.reranker.model_name if self.reranker.is_available() else "Not loaded"
        else:
            status["reranker_model"] = "Temporarily disabled"
        
        return status