"""Vector store implementation using ChromaDB for RAG."""

import os
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from .custom_embeddings import create_embeddings
from langchain.schema import Document


class VectorStoreManager:
    """Manages the vector database for document storage and retrieval."""
    
    def __init__(self, persist_directory: str = "/app/data/chroma_db", 
                 collection_name: str = "rag_documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embeddings model - Qwen3-Embedding-0.6B (local)
        print("🔧 Initializing Qwen3 embeddings...")
        self.embeddings = create_embeddings(
            model_path="/app/models/embedding/Qwen3-Embedding-0.6B",
            device="cpu"
        )
        
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
            print(f"Initialized vector store with collection: {self.collection_name}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        if not documents:
            print("No documents to add")
            return []
            
        try:
            # Generate unique IDs for documents
            ids = [f"doc_{i}_{hash(doc.page_content)}" for i, doc in enumerate(documents)]
            
            # Add documents to vector store
            self.vector_store.add_documents(documents, ids=ids)
            print(f"Added {len(documents)} documents to vector store")
            
            return ids
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
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
                
            print(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Search for similar documents with similarity scores."""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            print(f"Found {len(results)} documents with scores for query: {query[:50]}...")
            
            # Debug: Print actual scores returned
            for i, (doc, score) in enumerate(results):
                print(f"Debug vector_store: Document {i}: score={score}, type={type(score)}")
            
            return results
            
        except Exception as e:
            print(f"Error during similarity search with scores: {e}")
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
            print(f"Error getting collection info: {e}")
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
            print(f"Deleted collection: {self.collection_name}")
            self._initialize_vector_store()
            
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def delete_documents_by_content(self, content_query: str) -> int:
        """Delete documents that contain the specified content."""
        try:
            # Get the ChromaDB collection directly
            collection = self.client.get_collection(self.collection_name)
            
            # Get all documents to search through them
            results = collection.get(include=['documents', 'metadatas', 'ids'])
            
            if not results['documents']:
                print("No documents found in collection")
                return 0
            
            # Find documents that contain the query string
            ids_to_delete = []
            deleted_count = 0
            
            for i, doc_content in enumerate(results['documents']):
                if content_query.lower() in doc_content.lower():
                    doc_id = results['ids'][i]
                    ids_to_delete.append(doc_id)
                    deleted_count += 1
                    print(f"Found matching document: {doc_id}")
                    print(f"Content preview: {doc_content[:100]}...")
            
            # Delete the matching documents
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                print(f"Successfully deleted {deleted_count} document(s)")
            else:
                print(f"No documents found containing: '{content_query}'")
            
            return deleted_count
            
        except Exception as e:
            print(f"Error deleting documents by content: {e}")
            raise
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """Get a retriever object for use with LangChain."""
        if search_kwargs is None:
            search_kwargs = {"k": 4}
            
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)