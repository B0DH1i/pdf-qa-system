"""
API module.
This module provides REST API endpoints and FastAPI integration.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import uuid
import tempfile
import shutil
import time
import json
import numpy as np
from datetime import datetime
import re

from loguru import logger

from ...core.exceptions import APIError, ProcessingError, StorageError
from ...processing.text import TextProcessor
from ...processing.document import DocumentProcessor
from ...storage.vector_store import VectorStore
from ...storage.knowledge import KnowledgeGraph
from ...core.memory import MemoryManager

class QuestionRequest(BaseModel):
    """Question-answer request data model."""
    question: str = Field(..., description="The question to ask about the document")
    doc_id: Optional[str] = Field(None, description="Document ID to target a specific document")
    threshold: float = Field(0.45, description="Similarity threshold for vector search")
    max_results: int = Field(5, description="Maximum number of results to return")

class DocumentResponse(BaseModel):
    """Document response data model."""
    doc_id: str
    filename: str
    upload_time: str
    page_count: int
    status: str
    file_size: int

class AnswerResponse(BaseModel):
    """Answer response data model."""
    answer: str
    context: List[str] = []
    sources: List[Dict[str, Any]] = []
    processing_time: float
    doc_id: Optional[str] = None

class API:
    """API class."""
    
    def __init__(self, port=8000, host="0.0.0.0"):
        """
        REST API class. FastAPI is used to create REST API.
        
        Args:
            port: API's port to run
            host: API's host to run 
        """
        self.port = port
        self.host = host
        self.app = FastAPI(title="PDF Question-Answer API", version="1.0.0")
        
        # Middleware and CORS settings
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Create upload folder
        self.upload_folder = Path("uploads")
        self.upload_folder.mkdir(exist_ok=True)
        
        # Static file directory
        self.static_folder = Path("static")
        self.static_folder.mkdir(exist_ok=True)
        
        # Static file service declaration
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # Document database
        self.documents = {}
        
        # Services
        self.text_processor = TextProcessor(config={"model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"})
        self.vector_store = VectorStore(collection_name="pdf_qa")
        self.document_processor = DocumentProcessor(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", vector_store_dir="vector_db")
        self.knowledge_graph = KnowledgeGraph()
        self.memory_manager = MemoryManager()
        
        # Define API routes
        self.define_routes()
        
        # Clean up
        try:
            self.vector_store.cleanup()
        except Exception as e:
            logger.warning(f"Vector store cleanup failed: {e}")
    
    def define_routes(self):
        """Define API routes"""
        
        @self.app.get("/")
        async def root():
            """Home page"""
            return FileResponse("static/index.html")
        
        @self.app.post("/upload", response_model=DocumentResponse)
        async def upload_document(file: UploadFile = File(...)):
            """
            Upload PDF file
            """
            start_time = time.time()
            
            # File check
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are accepted")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                try:
                    # Read file content
                    file_content = await file.read()
                    
                    # Save to temporary file
                    temp_pdf.write(file_content)
                    temp_path = temp_pdf.name
                    
                    # Generate unique document ID
                    doc_id = str(uuid.uuid4())
                    
                    # Prepare metadata
                    file_size = len(file_content)
                    current_time = datetime.now().isoformat()
                    file_path = os.path.join(self.upload_folder, f"{doc_id}.pdf")
                    
                    # Metadata for tracking
                    metadata = {
                        "filename": file.filename,
                        "upload_time": current_time,
                        "file_size": file_size,
                        "file_path": file_path,
                        "status": "processing",
                        "page_count": 0,
                    }
                    
                    # Save document info
                    self.documents[doc_id] = metadata
                    
                    # Copy from temp to storage location
                    shutil.copy(temp_path, file_path)
                    
                    # Process the document in background
                    async def process_document():
                        try:
                            # Extract text from PDF
                            document_texts, page_count = self.document_processor.process_pdf(file_path)
                            
                            # Extract embeddings
                            embeddings = []
                            
                            # Process in batches to avoid memory issues
                            batch_size = 32
                            for i in range(0, len(document_texts), batch_size):
                                batch = document_texts[i:i+batch_size]
                                batch_embeddings = self.text_processor.get_embeddings(batch)
                                embeddings.extend(batch_embeddings)
                            
                            # Prepare metadata for vector store
                            metadata_list = []
                            for i, text in enumerate(document_texts):
                                metadata_list.append({
                                    "text": text,
                                    "page_num": i // 10 + 1,  # Approximately 10 chunks per page
                                    "doc_id": doc_id,
                                    "filename": file.filename,
                                    "upload_time": current_time
                                })
                            
                            # Create IDs for chunks
                            chunk_ids = [f"{doc_id}_{i}" for i in range(len(document_texts))]
                            
                            # Add to vector store
                            logger.info(f"Adding {len(embeddings)} embeddings to vector store with collection 'documents'")
                            logger.info(f"Metadata length: {len(metadata_list)}, Embeddings shape: {np.array(embeddings).shape}, IDs length: {len(chunk_ids)}")
                            
                            self.vector_store.change_collection("documents")
                            self.vector_store.add_documents(
                                texts=document_texts,
                                embeddings=embeddings,
                                metadata=metadata_list,
                                ids=chunk_ids
                            )
                            logger.info(f"Successfully added {len(document_texts)} documents to vector store")
                            
                            # Update document status
                            self.documents[doc_id]["status"] = "ready"
                            self.documents[doc_id]["page_count"] = page_count
                            
                            # Create knowledge graph
                            # Temporarily disabled - performance issues
                            # self.knowledge_graph.add_document(doc_id, document_texts)
                            
                        except Exception as e:
                            # Update document status
                            self.documents[doc_id]["status"] = "error"
                            logger.error(f"Error processing document: {str(e)}")
                    
                    # Start background task
                    import asyncio
                    asyncio.create_task(process_document())
                    
                    # Return response
                    response_data = {
                        "doc_id": doc_id,
                        "filename": file.filename,
                        "upload_time": current_time,
                        "status": "processing",
                        "file_size": file_size,
                        "page_count": 0
                    }
                    
                    logger.info(f"Document {doc_id} processed in {time.time() - start_time:.2f} seconds")
                    return response_data
                    
                except Exception as e:
                    logger.error(f"Error during file upload: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")
                finally:
                    # Clean up temp file - use a more robust approach for Windows
                    if os.path.exists(temp_pdf.name):
                        try:
                            # On Windows, files might be locked by another process
                            # Try multiple times with small delays
                            max_attempts = 3
                            for attempt in range(max_attempts):
                                try:
                                    os.unlink(temp_pdf.name)
                                    break  # Successfully deleted
                                except Exception as e:
                                    if attempt < max_attempts - 1:
                                        # Wait a bit and retry
                                        time.sleep(0.5)
                                    else:
                                        # Last attempt failed, just log the error and continue
                                        logger.warning(f"Could not delete temporary file {temp_pdf.name}: {str(e)}")
                                        # Temporary files will eventually be cleaned up by the OS
                        except Exception as e:
                            # Just log the error and continue
                            logger.warning(f"Could not delete temporary file {temp_pdf.name}: {str(e)}")
                            # Temporary files will eventually be cleaned up by the OS
        
        @self.app.post("/ask", response_model=AnswerResponse)
        async def ask_question(request: QuestionRequest):
            """
            Ask a question about the document
            """
            start_time = time.time()
            
            try:
                # Log the request parameters
                logger.info(f"Processing question: {request.question}")
                logger.info(f"Using threshold: {request.threshold}, max_results: {request.max_results}")
                
                # Get question embedding
                question_embedding = self.text_processor.get_embeddings([request.question])[0]
                logger.info(f"Generated question embedding with shape: {question_embedding.shape}")
                
                # Setup filter
                filter_dict = {}
                if request.doc_id:
                    filter_dict = {"doc_id": request.doc_id}
                    logger.info(f"Filtering by doc_id: {request.doc_id}")
                
                # Log collection stats
                try:
                    collection_stats = self.vector_store.get_collection_stats(collection_name="documents")
                    logger.info(f"Collection stats: {collection_stats}")
                except Exception as e:
                    logger.warning(f"Failed to get collection stats: {e}")
                
                # Search for similar texts
                self.vector_store.change_collection("documents")
                results = self.vector_store.search_similar(
                    query_embedding=question_embedding, 
                    n_results=request.max_results,
                    filter_criteria=filter_dict
                )
                
                # Check if we got any results
                if not results or len(results['ids']) == 0:
                    logger.warning("No results found in vector search")
                    return AnswerResponse(
                        answer="No information found to answer this question.",
                        context=[],
                        sources=[],
                        processing_time=time.time() - start_time,
                        doc_id=request.doc_id
                    )
                
                # Log search results    
                logger.info(f"Search results - IDs count: {len(results['ids'])}, Distances: {results['distances']}")
                
                # Show sample metadata if we have results
                if len(results['ids']) > 0 and 'metadatas' in results and results['metadatas']:
                    logger.info(f"Sample metadata: {results['metadatas']}")
                
                # Extract texts and metadata from results
                contexts = []
                sources = []
                valid_results = False
                
                # ChromaDB döndürüleri iç içe liste yapısında olabilir
                # results['ids']: [["id1", "id2", ...]] şeklinde ya da ["id1", "id2", ...] şeklinde
                # İç içe liste kontrolü yapalım
                ids_list = results['ids'][0] if isinstance(results['ids'][0], list) else results['ids']
                distances_list = results['distances'][0] if isinstance(results['distances'][0], list) else results['distances']
                metadatas_list = results['metadatas'][0] if isinstance(results['metadatas'][0], list) else results['metadatas']
                
                # Debug: print all distances and thresholds for analysis
                logger.info(f"Evaluating distances against threshold {request.threshold}")
                
                for i, distance in enumerate(distances_list):
                    logger.info(f"Result {i+1}: distance={distance}, threshold={request.threshold}")
                    
                    # For cosine similarity, LOWER distance is BETTER (more similar)
                    if distance < request.threshold:
                        valid_results = True
                        metadata = metadatas_list[i]
                        contexts.append(metadata['text'])
                        sources.append({
                            'doc_id': metadata['doc_id'],
                            'filename': metadata['filename'],
                            'page_num': metadata['page_num']
                        })
                        logger.info(f"Added result {i+1} as it meets threshold requirement")
                
                # If no documents pass the threshold check after all
                if not valid_results:
                    logger.warning(f"No results met the threshold requirement of {request.threshold}")
                    return AnswerResponse(
                        answer="No information found to answer this question.",
                        context=[],
                        sources=[],
                        processing_time=time.time() - start_time,
                        doc_id=request.doc_id
                    )
                
                # Create context
                context_text = "\n".join(contexts)
                
                # Create a more detailed response
                if len(sources) > 0:
                    # Find the content with highest similarity
                    best_match_idx = distances_list.index(min(distances_list))
                    best_match_text = metadatas_list[best_match_idx]['text']
                    
                    # Match the relevant text with the question for a more meaningful answer
                    # Find the most similar sentences
                    sentences = best_match_text.split('.')
                    best_sentences = []
                    
                    logger.info(f"Question: {request.question}")
                    logger.info(f"Best match text: {best_match_text[:200]}...")
                    
                    # Türkçe ve diğer diller için ön işleme
                    def normalize_text(text):
                        """Metni normalleştir ve basit bir şekilde tokenize et"""
                        if not text:
                            return []
                        text = text.lower().strip()
                        # Noktalama işaretlerini ve özel karakterleri temizle
                        text = re.sub(r'[^\w\s]', ' ', text)
                        # Fazla boşlukları temizle
                        text = re.sub(r'\s+', ' ', text)
                        return text.split()
                    
                    # Soru anahtar kelimelerini normal hale getir
                    question_keywords = normalize_text(request.question)
                    logger.debug(f"Normalized question keywords: {question_keywords}")
                    
                    for sentence in sentences:
                        if len(sentence.strip()) > 15:  # Çok kısa cümleleri atla
                            # Cümleyi normalize et
                            sentence_tokens = normalize_text(sentence)
                            logger.debug(f"Checking sentence: {sentence[:50]}...")
                            
                            match_found = False
                            for keyword in question_keywords:
                                if len(keyword) > 2:  # 2 harften uzun kelimeleri kontrol et
                                    # Kelime eşleşmesi
                                    if keyword in sentence_tokens:
                                        match_found = True
                                        logger.debug(f"Matched keyword: {keyword}")
                                        break
                                    # Alternatif: Kelime eşleşmesi bulunamazsa, kök şeklini de kontrol et
                                    # (Basitleştirilmiş bir yaklaşım)
                                    elif len(keyword) > 5 and any(token.startswith(keyword[:5]) for token in sentence_tokens):
                                        match_found = True
                                        logger.debug(f"Matched keyword root: {keyword[:5]}")
                                        break
                            
                            if match_found:
                                best_sentences.append(sentence.strip())
                    
                    if best_sentences:
                        answer = ". ".join(best_sentences[:3]) + "."
                    else:
                        # En iyi eşleşmenin bir kısmını göster
                        answer = f"Belgelerinizde bulunan en alakalı bilgi: {best_match_text[:500]}..."
                else:
                    answer = "Bu soru hakkında bilgi bulunamadı."
                
                # Memory update - disabled
                # self.memory_manager.add_interaction(
                #     question=request.question,
                #     answer=answer,
                #     context=context_text,
                #     sources=sources
                # )
                
                return AnswerResponse(
                    answer=answer,
                    context=contexts,
                    sources=sources,
                    processing_time=time.time() - start_time,
                    doc_id=request.doc_id if request.doc_id else None
                )
                
            except Exception as e:
                logger.error(f"Question processing error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Question processing error: {str(e)}")
        
        @self.app.get("/documents", response_model=List[DocumentResponse])
        async def get_documents():
            """
            Get all documents
            """
            documents_list = []
            for doc_id, doc_info in self.documents.items():
                documents_list.append(
                    DocumentResponse(
                        doc_id=doc_id,
                        filename=doc_info["filename"],
                        upload_time=doc_info["upload_time"],
                        page_count=doc_info.get("page_count", 0),
                        status=doc_info["status"],
                        file_size=doc_info["file_size"]
                    )
                )
            return documents_list
        
        @self.app.get("/document/{doc_id}")
        async def get_document(doc_id: str):
            """
            Get document by ID
            """
            if doc_id not in self.documents:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
            return self.documents[doc_id]
    
    def start(self, host=None, port=None):
        """
        Start the API
        """
        host = host or self.host
        port = port or self.port
        
        try:
            logger.info(f"Starting API on {host}:{port}")
            uvicorn.run(self.app, host=host, port=port)
        except Exception as e:
            logger.error(f"Error starting API: {str(e)}")
    
    def __del__(self):
        """
        Cleanup when the instance is deleted
        """
        try:
            self.vector_store.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            pass 