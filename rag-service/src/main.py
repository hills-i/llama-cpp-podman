"""FastAPI application for RAG service."""

import os
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handling middleware
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler to ensure JSON responses."""
    import traceback
    
    # Log the error
    print(f"Global exception: {str(exc)}")
    print(f"Traceback: {traceback.format_exc()}")
    
    # Return JSON error response
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "type": "server_error"
        }
    )

# Global components
vector_store_manager = None
rag_chain = None
document_processor = DocumentProcessor()


@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup."""
    global vector_store_manager, rag_chain
    
    print("Initializing RAG service...")
    
    # Initialize vector store
    vector_store_manager = VectorStoreManager()
    
    # Initialize RAG chain with reranker enabled
    rag_chain = RAGChain(vector_store_manager, use_reranker=True)
    
    # Load sample documents if no documents exist
    collection_info = vector_store_manager.get_collection_info()
    if collection_info["document_count"] == 0:
        print("No documents found, loading sample documents...")
        sample_docs = load_sample_documents()
        vector_store_manager.add_documents(sample_docs)
        print("Sample documents loaded")
    
    print("RAG service initialization complete")


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
    
    # Create temporary directory for uploaded files
    upload_dir = Path("/tmp/rag_uploads")
    upload_dir.mkdir(exist_ok=True)
    
    uploaded_files = []
    
    try:
        # Save uploaded files
        for file in files:
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            uploaded_files.append(str(file_path))
        
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
    
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=400, detail=f"Directory {directory_path} does not exist")
    
    try:
        documents = document_processor.process_directory(directory_path)
        
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