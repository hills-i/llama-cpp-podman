"""FastAPI application for RAG service."""

import os
import logging
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
        if not allowed_base:
            continue
        allowed_real = os.path.realpath(os.path.abspath(allowed_base))

        # Prevent prefix-bypass issues like "/app/documents_evil" matching "/app/documents".
        # commonpath is robust across path separators.
        try:
            if os.path.commonpath([real_path, allowed_real]) == allowed_real:
                logger.info(f"Path validated: {real_path}")
                return real_path
        except ValueError:
            # Different drives / invalid paths on some platforms
            continue

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
    k: int = Field(default=4, ge=1, le=50)


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

# CORS:
# - Default to same-origin (no CORS) since Apache proxies UI and API on one origin.
# - If you need browser cross-origin access, set CORS_ALLOW_ORIGINS to a comma-separated allowlist.
cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "")
cors_allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

# Never allow credentialed requests with wildcard origins.
cors_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
if not cors_allow_origins:
    cors_allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global error handler - sanitize errors for security
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler with error sanitization."""
    # Preserve HTTPException semantics (status codes + client-facing detail)
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    # Preserve FastAPI validation errors (422)
    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

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
    rag_chain = RAGChain(
        vector_store_manager,
        use_reranker=True,
        llama_base_url=os.getenv("LLAMA_CPP_BASE_URL"),
    )

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

    result = rag_chain.query(
        question=request.question,
        include_sources=request.include_sources,
    )
    return QueryResponse(**result)


@app.post("/search", response_model=List[DocumentInfo])
async def search_documents(request: QueryRequest):
    """Search for similar documents without generation."""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    results = rag_chain.simple_retrieval(request.question, k=request.k)
    return [DocumentInfo(**result) for result in results]


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents to the knowledge base."""
    if not vector_store_manager:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    chunk_size_bytes = int(os.getenv("UPLOAD_CHUNK_SIZE_BYTES", str(1024 * 1024)))

    # Validate file count
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FILES_PER_UPLOAD} files allowed per upload"
        )

    # Create temporary directory for uploaded files
    upload_dir = Path("/tmp/rag_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

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

            # Save file (sanitize filename to avoid path traversal)
            original_name = file.filename or ""
            safe_name = Path(original_name).name
            if not safe_name or safe_name in {".", ".."}:
                raise HTTPException(status_code=400, detail="Invalid filename")

            file_path = upload_dir / f"{uuid.uuid4().hex}_{safe_name}"
            resolved_target = file_path.resolve()
            if os.path.commonpath([str(resolved_target), str(upload_dir.resolve())]) != str(upload_dir.resolve()):
                raise HTTPException(status_code=400, detail="Invalid upload path")

            # Stream file to disk while enforcing per-file and total size limits.
            bytes_written = 0
            with open(file_path, "wb") as buffer:
                while True:
                    chunk = await file.read(chunk_size_bytes)
                    if not chunk:
                        break

                    bytes_written += len(chunk)
                    if bytes_written > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File {safe_name} exceeds {MAX_FILE_SIZE_MB}MB limit",
                        )

                    total_size += len(chunk)
                    if total_size > MAX_UPLOAD_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Total upload size exceeds {MAX_UPLOAD_SIZE_MB}MB limit",
                        )

                    buffer.write(chunk)

            uploaded_files.append(str(file_path))
            logger.info(f"Uploaded: {safe_name} ({bytes_written} bytes)")
        
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

    finally:
        # Clean up temporary files
        for file_path in uploaded_files:
            try:
                os.remove(file_path)
            except OSError as exc:
                logger.warning("Failed to remove temp upload", extra={"path": file_path, "error": str(exc)})


@app.post("/ingest")
async def ingest_directory(directory_path: str = Form(...)):
    """Ingest documents from a directory path."""
    if not vector_store_manager:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    # Validate path to prevent traversal attacks
    validated_path = validate_path(directory_path)

    if not os.path.exists(validated_path):
        raise HTTPException(status_code=400, detail="Directory does not exist")

    documents = document_processor.process_directory(validated_path)

    if documents:
        vector_store_manager.add_documents(documents)
        return {
            "message": f"Successfully ingested documents from {directory_path}",
            "documents_added": len(documents),
            "directory": directory_path,
        }

    return {
        "message": f"No processable documents found in {directory_path}",
        "documents_added": 0,
        "directory": directory_path,
    }


@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the vector store."""
    if not vector_store_manager:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    vector_store_manager.delete_collection()
    return {"message": "All documents cleared successfully"}


class DeleteByContentRequest(BaseModel):
    content_query: str


@app.post("/documents/delete-by-content")
async def delete_document_by_content(request: DeleteByContentRequest):
    """Delete documents by partial content match."""
    if not vector_store_manager:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    deleted_count = vector_store_manager.delete_documents_by_content(request.content_query)

    if deleted_count > 0:
        return {
            "message": f"Successfully deleted {deleted_count} document(s)",
            "deleted_count": deleted_count,
            "content_query": request.content_query,
        }

    return {
        "message": "No documents found matching the content query",
        "deleted_count": 0,
        "content_query": request.content_query,
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )