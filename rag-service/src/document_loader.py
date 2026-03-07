"""Document loading and processing utilities for RAG."""

import json
import logging
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


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
    
    def _json_metadata(self, file_path: str, file_name: str, **extra) -> dict:
        """Build base metadata dict for JSON documents."""
        meta = {"source": file_path, "file_type": "json", "file_name": file_name}
        meta.update(extra)
        return meta

    def load_json_document(self, file_path: str) -> List[Document]:
        """Load and process JSON documents."""
        documents = []
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

        try:
            file_size = os.path.getsize(file_path)
            if file_size > MAX_FILE_SIZE:
                logger.warning("JSON file too large; skipping", extra={"path": file_path, "size": file_size})
                return documents

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            file_name = Path(file_path).name

            if isinstance(data, dict):
                if 'content' in data:
                    metadata = self._json_metadata(file_path, file_name)
                    if 'metadata' in data:
                        metadata.update(data['metadata'])
                    for key, value in data.items():
                        if key not in ('content', 'metadata') and isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                    documents.append(Document(page_content=data['content'], metadata=metadata))

                elif 'chunks' in data and isinstance(data['chunks'], list):
                    for i, chunk in enumerate(data['chunks']):
                        if isinstance(chunk, str):
                            documents.append(Document(
                                page_content=chunk,
                                metadata=self._json_metadata(file_path, file_name, chunk_index=i),
                            ))

                else:
                    documents.append(Document(
                        page_content=json.dumps(data, ensure_ascii=False, indent=2),
                        metadata=self._json_metadata(file_path, file_name, json_structure="object"),
                    ))

            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict) and 'content' in item:
                        metadata = self._json_metadata(file_path, file_name, item_index=i)
                        for key, value in item.items():
                            if key != 'content' and isinstance(value, (str, int, float, bool)):
                                metadata[key] = value
                        documents.append(Document(page_content=item['content'], metadata=metadata))

                    elif isinstance(item, str):
                        documents.append(Document(
                            page_content=item,
                            metadata=self._json_metadata(file_path, file_name, item_index=i),
                        ))

                    else:
                        documents.append(Document(
                            page_content=json.dumps(item, ensure_ascii=False, indent=2),
                            metadata=self._json_metadata(file_path, file_name, item_index=i),
                        ))

            else:
                documents.append(Document(
                    page_content=json.dumps(data, ensure_ascii=False, indent=2),
                    metadata=self._json_metadata(file_path, file_name),
                ))

        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON; skipping", extra={"path": file_path, "error": str(e)})
        except Exception as e:
            logger.exception("Error loading JSON", extra={"path": file_path})

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

        base_dir_resolved = directory_path.resolve()
            
        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                # Avoid following symlinks (can escape the allowed base directory)
                if file_path.is_symlink():
                    logger.warning("Skipping symlink", extra={"path": str(file_path)})
                    continue

                try:
                    resolved_file = file_path.resolve()
                except OSError as exc:
                    logger.warning("Skipping unreadable path", extra={"path": str(file_path), "error": str(exc)})
                    continue

                try:
                    # Ensure the resolved file stays within the base directory
                    if os.path.commonpath([str(resolved_file), str(base_dir_resolved)]) != str(base_dir_resolved):
                        logger.warning(
                            "Skipping path outside base directory",
                            extra={"path": str(file_path), "resolved": str(resolved_file), "base": str(base_dir_resolved)},
                        )
                        continue
                except ValueError:
                    continue

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
                        logger.info(
                            "Loaded documents",
                            extra={"path": str(file_path), "count": len(file_documents)},
                        )
                        
                    except Exception as e:
                        logger.exception("Error loading document", extra={"path": str(file_path)})
                        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)
    
    def process_directory(self, directory: str) -> List[Document]:
        """Load and split documents from a directory."""
        documents = self.load_documents(directory)
        if not documents:
            logger.info("No documents loaded", extra={"directory": directory})
            return []
            
        logger.info(
            "Splitting documents into chunks",
            extra={"directory": directory, "documents": len(documents)},
        )
        chunks = self.split_documents(documents)
        logger.info("Created document chunks", extra={"chunks": len(chunks)})
        
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