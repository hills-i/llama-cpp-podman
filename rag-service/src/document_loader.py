"""Document loading and processing utilities for RAG."""

import os
import json
from typing import List, Optional
from pathlib import Path

from langchain.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentProcessor:
    """Handles document loading, splitting, and preprocessing for RAG."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_json_document(self, file_path: str) -> List[Document]:
        """Load and process JSON documents."""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_name = Path(file_path).name
            
            # Handle different JSON structures
            if isinstance(data, dict):
                # If it's a dictionary, try different strategies
                if 'content' in data:
                    # Structure: {"title": "...", "content": "...", "metadata": {...}}
                    content = data['content']
                    metadata = {
                        "source": file_path,
                        "file_type": "json",
                        "file_name": file_name,
                    }
                    
                    # Add any additional metadata from the JSON
                    if 'metadata' in data:
                        metadata.update(data['metadata'])
                    
                    # Add other top-level fields as metadata
                    for key, value in data.items():
                        if key not in ['content', 'metadata'] and isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                    
                    documents.append(Document(page_content=content, metadata=metadata))
                
                elif 'chunks' in data and isinstance(data['chunks'], list):
                    # Structure: {"chunks": ["chunk1", "chunk2", ...]}
                    for i, chunk in enumerate(data['chunks']):
                        if isinstance(chunk, str):
                            metadata = {
                                "source": file_path,
                                "file_type": "json",
                                "file_name": file_name,
                                "chunk_index": i,
                            }
                            documents.append(Document(page_content=chunk, metadata=metadata))
                
                else:
                    # Fallback: convert entire JSON to text
                    content = json.dumps(data, ensure_ascii=False, indent=2)
                    metadata = {
                        "source": file_path,
                        "file_type": "json",
                        "file_name": file_name,
                        "json_structure": "object"
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
            
            elif isinstance(data, list):
                # If it's a list, process each item
                for i, item in enumerate(data):
                    if isinstance(item, dict) and 'content' in item:
                        content = item['content']
                        metadata = {
                            "source": file_path,
                            "file_type": "json",
                            "file_name": file_name,
                            "item_index": i,
                        }
                        
                        # Add other fields as metadata
                        for key, value in item.items():
                            if key != 'content' and isinstance(value, (str, int, float, bool)):
                                metadata[key] = value
                        
                        documents.append(Document(page_content=content, metadata=metadata))
                    
                    elif isinstance(item, str):
                        # List of strings
                        metadata = {
                            "source": file_path,
                            "file_type": "json",
                            "file_name": file_name,
                            "item_index": i,
                        }
                        documents.append(Document(page_content=item, metadata=metadata))
                    
                    else:
                        # Fallback: convert item to JSON string
                        content = json.dumps(item, ensure_ascii=False, indent=2)
                        metadata = {
                            "source": file_path,
                            "file_type": "json",
                            "file_name": file_name,
                            "item_index": i,
                        }
                        documents.append(Document(page_content=content, metadata=metadata))
            
            else:
                # Fallback for other data types
                content = json.dumps(data, ensure_ascii=False, indent=2)
                metadata = {
                    "source": file_path,
                    "file_type": "json",
                    "file_name": file_name,
                }
                documents.append(Document(page_content=content, metadata=metadata))
        
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            print(f"Error loading JSON {file_path}: {e}")
        
        return documents
        
    def load_documents(self, directory: str) -> List[Document]:
        """Load documents from a directory."""
        documents = []
        
        # Define loaders for different file types
        loaders = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".json": "json_custom",  # Special marker for JSON handling
        }
        
        directory_path = Path(directory)
        if not directory_path.exists():
            raise ValueError(f"Directory {directory} does not exist")
            
        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                file_extension = file_path.suffix.lower()
                
                if file_extension in loaders:
                    try:
                        if file_extension == ".json":
                            # Handle JSON files with custom loader
                            file_documents = self.load_json_document(str(file_path))
                        else:
                            # Handle other file types with standard loaders
                            loader_class = loaders[file_extension]
                            loader = loader_class(str(file_path))
                            file_documents = loader.load()
                            
                            # Add metadata for non-JSON files
                            for doc in file_documents:
                                doc.metadata.update({
                                    "source": str(file_path),
                                    "file_type": file_extension[1:],  # Remove the dot
                                    "file_name": file_path.name,
                                })
                        
                        documents.extend(file_documents)
                        print(f"Loaded {len(file_documents)} documents from {file_path}")
                        
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)
    
    def process_directory(self, directory: str) -> List[Document]:
        """Load and split documents from a directory."""
        documents = self.load_documents(directory)
        if not documents:
            print(f"No documents loaded from {directory}")
            return []
            
        print(f"Loaded {len(documents)} documents, splitting into chunks...")
        chunks = self.split_documents(documents)
        print(f"Created {len(chunks)} document chunks")
        
        return chunks


def load_sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    sample_docs = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware and can reason about their responses.",
            metadata={"source": "langchain_intro.txt", "type": "introduction"}
        ),
        Document(
            page_content="Retrieval-Augmented Generation (RAG) combines the power of retrieval systems with generative models. It retrieves relevant information from a knowledge base and uses it to generate more informed responses.",
            metadata={"source": "rag_explanation.txt", "type": "explanation"}
        ),
        Document(
            page_content="Vector databases store high-dimensional vectors and enable efficient similarity search. They are essential for RAG systems to quickly find relevant documents based on semantic similarity.",
            metadata={"source": "vector_db_info.txt", "type": "technical"}
        ),
        Document(
            page_content="ChromaDB is an open-source embedding database that makes it easy to build LLM applications. It provides a simple API for storing and querying embeddings with metadata filtering.",
            metadata={"source": "chromadb_info.txt", "type": "database"}
        ),
    ]
    
    return sample_docs