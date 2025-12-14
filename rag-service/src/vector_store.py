"""Vector store implementation using ChromaDB for RAG."""

import os
import logging
import hashlib
import time
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from .custom_embeddings import create_embeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages the vector database for document storage and retrieval."""
    
    def __init__(self, persist_directory: str = "/app/data/chroma_db", 
                 collection_name: str = "rag_documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embeddings client (remote llama.cpp embedding service)
        logger.info("Initializing embeddings via remote service...")
        self.embeddings = create_embeddings()
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the Chroma vector store."""
        try:
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info(f"Initialized vector store with collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        if not documents:
            logger.info("No documents to add")
            return []

        try:
            # Generate stable, unique IDs using SHA256
            ids = []
            for doc in documents:
                # Create hash from content
                content_hash = hashlib.sha256(
                    doc.page_content.encode('utf-8')
                ).hexdigest()[:16]

                # Add timestamp for uniqueness
                timestamp = str(int(time.time() * 1000))
                doc_id = f"doc_{content_hash}_{timestamp}"
                ids.append(doc_id)

            # Add documents to vector store
            self.vector_store.add_documents(documents, ids=ids)
            logger.info(f"Added {len(documents)} documents to vector store")

            return ids

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 4, 
                         filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search for similar documents."""
        try:
            if filter_metadata:
                results = self.vector_store.similarity_search(
                    query, k=k, filter=filter_metadata
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)

            logger.debug(f"Found {len(results)} similar documents")
            return results

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Search for similar documents with similarity scores."""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.debug(f"Found {len(results)} documents with scores")
            return results

        except Exception as e:
            logger.error(f"Error during similarity search with scores: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "persist_directory": self.persist_directory,
                "error": str(e)
            }

    def delete_collection(self):
        """Delete the current collection (useful for reset)."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            self._initialize_vector_store()

        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def delete_documents_by_content(self, content_query: str) -> int:
        """Delete documents that contain the specified content."""
        try:
            # Get the ChromaDB collection directly
            collection = self.client.get_collection(self.collection_name)

            # Safety guard: avoid loading/scanning huge collections into memory.
            max_scan = int(os.getenv("MAX_DELETE_SCAN", "5000"))
            try:
                collection_size = int(collection.count())
            except Exception:
                collection_size = 0

            if max_scan > 0 and collection_size > max_scan:
                logger.warning(
                    "Refusing delete-by-content on large collection",
                    extra={"collection": self.collection_name, "count": collection_size, "max_scan": max_scan},
                )
                raise RuntimeError(
                    f"delete-by-content disabled for large collections (size={collection_size}). "
                    "Use /documents to clear all documents or implement a metadata-based delete."
                )
            
            # Get all documents to search through them
            results = collection.get(include=['documents', 'ids'])

            if not results['documents']:
                logger.info("No documents found in collection")
                return 0

            # Find documents that contain the query string
            ids_to_delete = []
            deleted_count = 0

            for i, doc_content in enumerate(results['documents']):
                if content_query.lower() in doc_content.lower():
                    doc_id = results['ids'][i]
                    ids_to_delete.append(doc_id)
                    deleted_count += 1
                    logger.debug(f"Found matching document: {doc_id}")

            # Delete the matching documents
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"Successfully deleted {deleted_count} document(s)")
            else:
                logger.info(f"No documents found containing: '{content_query}'")

            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting documents by content: {e}")
            raise
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """Get a retriever object for use with LangChain."""
        if search_kwargs is None:
            search_kwargs = {"k": 4}
            
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)