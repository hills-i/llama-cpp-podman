"""FastAPI application for RAG service."""

import os
import logging
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .document_loader import DocumentProcessor, load_sample_documents
from .vector_store import VectorStoreManager
from .rag_chain import RAGChain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
ALLOWED_DOCUMENT_PATHS = os.getenv(
    "ALLOWED_DOCUMENT_PATHS",
    "/app/documents,/tmp/rag_uploads"
).split(",")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
MAX_FILES_PER_UPLOAD = int(os.getenv("MAX_FILES_PER_UPLOAD", "10"))
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.json'}

MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_UPLOAD_SIZE = MAX_UPLOAD_SIZE_MB * 1024 * 1024


def validate_path(path: str) -> str:
    """Validate path to prevent traversal attacks."""
    real_path = os.path.realpath(os.path.abspath(path))

    for allowed_base in ALLOWED_DOCUMENT_PATHS:
        allowed_real = os.path.realpath(os.path.abspath(allowed_base))
        if real_path.startswith(allowed_real):
            logger.info(f"Path validated: {real_path}")
            return real_path

    logger.warning(f"Path traversal blocked: {path}")
    raise HTTPException(
        status_code=403,
        detail="Access denied: path outside allowed directories"
    )


def sanitize_error(exc: Exception, error_id: str = None) -> dict:
    """Sanitize error for safe client response."""
    if error_id is None:
        error_id = str(uuid.uuid4())[:8]

    logger.error(f"Error {error_id}: {exc}", exc_info=True)

    return {
        "error": "Operation failed",
        "error_id": error_id,
        "message": "An error occurred. Contact support with error ID."
    }


# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    question: str
    include_sources: bool = True
    k: int = 4


class QueryResponse(BaseModel):
    answer: str
    question: str
    sources: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class DocumentInfo(BaseModel):
    content: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None


class StatusResponse(BaseModel):
    status: str
    vector_store: Dict[str, Any]
    llm_available: bool
    chain_ready: bool


# Initialize FastAPI app
app = FastAPI(
    title="RAG Service",
    description="Retrieval-Augmented Generation service using LangChain and llama.cpp",
    version="1.0.0"
)

# CORS: Allow all origins (already protected by Apache Basic Auth)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global error handler - sanitize errors for security
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler with error sanitization."""
    error_response = sanitize_error(exc)
    return JSONResponse(
        status_code=500,
        content=error_response
    )

# Global components
vector_store_manager = None
rag_chain = None
document_processor = DocumentProcessor()


@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup."""
    global vector_store_manager, rag_chain

    logger.info("Initializing RAG service...")

    # Initialize vector store
    vector_store_manager = VectorStoreManager()

    # Initialize RAG chain with reranker enabled
    rag_chain = RAGChain(vector_store_manager, use_reranker=True)

    # Load sample documents if no documents exist
    collection_info = vector_store_manager.get_collection_info()
    if collection_info["document_count"] == 0:
        logger.info("No documents found, loading sample documents...")
        sample_docs = load_sample_documents()
        vector_store_manager.add_documents(sample_docs)
        logger.info("Sample documents loaded")

    logger.info("RAG service initialization complete")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "RAG Service is running", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status."""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    status_info = rag_chain.get_system_status()
    
    return StatusResponse(
        status="ready" if status_info["chain_ready"] and status_info["llm_available"] else "partial",
        vector_store=status_info["vector_store"],
        llm_available=status_info["llm_available"],
        chain_ready=status_info["chain_ready"]
    )


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_chain.query(
            question=request.question,
            include_sources=request.include_sources
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/search", response_model=List[DocumentInfo])
async def search_documents(request: QueryRequest):
    """Search for similar documents without generation."""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        results = rag_chain.simple_retrieval(request.question, k=request.k)
        
        return [DocumentInfo(**result) for result in results]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents to the knowledge base."""
    if not vector_store_manager:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    # Validate file count
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FILES_PER_UPLOAD} files allowed per upload"
        )

    # Create temporary directory for uploaded files
    upload_dir = Path("/tmp/rag_uploads")
    upload_dir.mkdir(exist_ok=True)

    uploaded_files = []
    total_size = 0

    try:
        # Validate and save uploaded files
        for file in files:
            # Check file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type {file_ext} not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
                )

            # Read and validate file size
            content = await file.read()
            file_size = len(content)

            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {file.filename} exceeds {MAX_FILE_SIZE_MB}MB limit"
                )

            total_size += file_size
            if total_size > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Total upload size exceeds {MAX_UPLOAD_SIZE_MB}MB limit"
                )

            # Save file
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            uploaded_files.append(str(file_path))
            logger.info(f"Uploaded: {file.filename} ({file_size} bytes)")
        
        # Process documents
        documents = document_processor.process_directory(str(upload_dir))
        
        if documents:
            # Add to vector store
            ids = vector_store_manager.add_documents(documents)
            
            return {
                "message": f"Successfully uploaded and processed {len(files)} files",
                "documents_added": len(documents),
                "file_names": [file.filename for file in files]
            }
        else:
            return {
                "message": "No processable content found in uploaded files",
                "documents_added": 0,
                "file_names": [file.filename for file in files]
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        for file_path in uploaded_files:
            try:
                os.remove(file_path)
            except:
                pass


@app.post("/ingest")
async def ingest_directory(directory_path: str = Form(...)):
    """Ingest documents from a directory path."""
    if not vector_store_manager:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    # Validate path to prevent traversal attacks
    validated_path = validate_path(directory_path)

    if not os.path.exists(validated_path):
        raise HTTPException(status_code=400, detail="Directory does not exist")

    try:
        documents = document_processor.process_directory(validated_path)
        
        if documents:
            ids = vector_store_manager.add_documents(documents)
            
            return {
                "message": f"Successfully ingested documents from {directory_path}",
                "documents_added": len(documents),
                "directory": directory_path
            }
        else:
            return {
                "message": f"No processable documents found in {directory_path}",
                "documents_added": 0,
                "directory": directory_path
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the vector store."""
    if not vector_store_manager:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        vector_store_manager.delete_collection()
        return {"message": "All documents cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")


class DeleteByContentRequest(BaseModel):
    content_query: str


@app.post("/documents/delete-by-content")
async def delete_document_by_content(request: DeleteByContentRequest):
    """Delete documents by partial content match."""
    if not vector_store_manager:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        deleted_count = vector_store_manager.delete_documents_by_content(request.content_query)
        
        if deleted_count > 0:
            return {
                "message": f"Successfully deleted {deleted_count} document(s)",
                "deleted_count": deleted_count,
                "content_query": request.content_query
            }
        else:
            return {
                "message": "No documents found matching the content query",
                "deleted_count": 0,
                "content_query": request.content_query
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )