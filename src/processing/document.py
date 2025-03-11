"""
PDF processing and question answering module with multilingual support.
"""

import fitz
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import gc
import time
import torch
from datetime import datetime
import os
from dataclasses import dataclass
import logging

from .text import TextProcessor
from ..core.memory import MemoryManager
from ..storage.vector_store import VectorStore
from ..core.feedback import FeedbackSystem, Feedback, PerformanceMetrics
from ..core.exceptions import ProcessingError

class DocumentProcessor:
    """Process PDF documents and answer questions in multiple languages."""
    
    # Memory optimization settings
    MEMORY_SETTINGS = {
        'chunk_batch_size': 32,  # Optimal batch size for processing
        'memory_threshold_mb': 500,  # Lower threshold for memory optimization
        'max_document_size_mb': 50,  # Lower max document size
        'chunk_size': 500,  # Smaller chunk size
        'chunk_overlap': 50  # Smaller overlap
    }
    
    # Supported models and language scopes
    MODELS = {
        'multilingual': {
            'name': 'multilingual',
            'description': 'Multilingual general purpose model',
            'languages': ['tr', 'en', 'de', 'fr', 'es', 'it', 'nl', 'pl', 'pt', 'ru', 'ja', 'ko', 'zh-cn', 'zh-tw', 'ar', 'hi']
        },
        'minilm': {
            'name': 'minilm',
            'description': 'MiniLM-based multilingual model',
            'languages': ['tr', 'en', 'de', 'fr', 'es', 'it', 'nl', 'pl', 'pt', 'ru', 'ja', 'ko', 'zh-cn', 'zh-tw', 'ar', 'hi']
        }
    }
    
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', vector_store_dir: Optional[str] = None):
        """
        Initialize PDF document processing class.
        
        Args:
            model_name: Model name to use. 'multilingual' or 'english'.
            vector_store_dir: Vector storage directory (optional).
        """
        try:
            # Device detection
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Save model type (for compatibility with old model_type parameter)
            self.model_type = "multilingual"  # Multilingual by default
            
            # Text processing model
            self.text_processor = TextProcessor(config={"model_name": model_name})
            
            # Memory manager
            self.memory_manager = MemoryManager()
            
            # Vector store
            self.vector_store = None
            if vector_store_dir:
                self.vector_store = VectorStore(
                    collection_name="documents",
                    dimension=self.text_processor.get_embedding_dimension(),
                    persist_directory=vector_store_dir
                )
                
            # Document folder
            self.docs_folder = Path("docs")
            self.docs_folder.mkdir(exist_ok=True)
            
            # Performans ve geri bildirim
            self.feedback_system = FeedbackSystem()
            self.total_queries = 0
            self.successful_queries = 0
            self.collection_name = "documents"

            logging.info(f"DocumentProcessor initialized with model: {model_name}")
            
        except Exception as e:
            raise ProcessingError(f"Failed to initialize DocumentProcessor: {str(e)}")
        
        # Performance tracking
        self.total_response_time = 0
        self.total_relevance_score = 0
        self.response_times = []
        self.relevance_scores = []
        self.language_pairs = []
        
        self.document_text = ""
        self.document_chunks = []
        self.document_language = None
        
    def extract_text(self, file_path: str) -> Tuple[List[int], List[str]]:
        """
        Extract text from PDF
        
        Args:
            file_path: PDF file path
            
        Returns:
            Tuple[List[int], List[str]]: Page numbers and text list
        """
        try:
            pages = []
            page_texts = []
            
            # Extract text with PyMuPDF
            with fitz.open(file_path) as pdf:
                for i, page in enumerate(pdf):
                    page_text = page.get_text()
                    if page_text.strip():  # Skip empty pages
                        pages.append(i+1)  # Page number starts from 1
                        page_texts.append(page_text)
            
            # Detect document language
            if page_texts:
                sample_text = " ".join(page_texts[:3])  # Get sample from first 3 pages
                self.document_language = self.text_processor.detect_language(sample_text)
                logging.info(f"Document language detected: {self.document_language}")
            
            # Return only page numbers and texts
            return pages, page_texts
            
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {e}")
            raise ProcessingError(f"Failed to extract text from PDF: {str(e)}")
    
    def _split_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        try:
            words = text.split()
            chunks = []
            i = 0
            
            while i < len(words):
                chunk = ' '.join(words[i:i + chunk_size])
                chunks.append(chunk)
                i += chunk_size - overlap
                
            return chunks
            
        except Exception as e:
            raise ProcessingError(f"Error splitting text into chunks: {str(e)}")
            
    def load_pdf(self, file_path: str) -> None:
        """Load PDF with memory optimization."""
        try:
            self.memory_manager.start_tracking("pdf_load")
            
            # Check file size
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            if file_size_mb > self.MEMORY_SETTINGS['max_document_size_mb']:
                raise ValueError(
                    f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size "
                    f"({self.MEMORY_SETTINGS['max_document_size_mb']}MB)"
                )
            
            # Process PDF in chunks
            doc = fitz.open(file_path)
            text_chunks = []
            
            for page in doc:
                chunk = page.get_text()
                if chunk.strip():  # Only add non-empty chunks
                    text_chunks.append(chunk)
                    
                # Optimize memory after each page
                self.memory_manager.optimize_memory(self.MEMORY_SETTINGS['memory_threshold_mb'])
            
            # Combine chunks and detect language
            self.document_text = "\n".join(text_chunks)
            self.document_language = self.text_processor.detect_language(self.document_text)
            
            # Split into smaller chunks
            self.document_chunks = self._split_into_chunks(
                self.document_text,
                chunk_size=self.MEMORY_SETTINGS['chunk_size'],
                overlap=self.MEMORY_SETTINGS['chunk_overlap']
            )
            
            # Create collection or use existing
            self.collection_name = Path(file_path).stem
            
            # Process chunks in smaller batches
            for i in range(0, len(self.document_chunks), self.MEMORY_SETTINGS['chunk_batch_size']):
                batch = self.document_chunks[i:i + self.MEMORY_SETTINGS['chunk_batch_size']]
                
                # Get embeddings for batch
                embeddings = self.text_processor.get_embeddings(batch)
                
                # Create metadata
                metadata = [{
                    "source": file_path,
                    "chunk_id": i + j,
                    "language": self.document_language,
                    "model": self.model_type,
                } for j in range(len(batch))]
                
                # Add to vector store
                self.vector_store.add_documents(batch, embeddings, metadata)
                
                # Optimize memory after each batch
                self.memory_manager.optimize_memory(self.MEMORY_SETTINGS['memory_threshold_mb'])
            
            self.memory_manager.stop_tracking("pdf_load")
            
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")
            raise ProcessingError(f"Failed to load PDF: {str(e)}")
            
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
    def cleanup(self):
        """Clean up resources."""
        if self.text_processor:
            self.text_processor.cleanup()
        if self.vector_store:
            self.vector_store.cleanup()
        if self.memory_manager:
            self.memory_manager.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def answer_question(self, question: str, n_results: int = 3) -> Dict:
        """Answer questions with memory optimization and performance tracking."""
        try:
            self.memory_manager.start_tracking("question_answering")
            start_time = time.time()
            
            if not self.collection_name:
                raise ValueError("Please load a PDF first")
                
            # Detect question language
            question_language = self.text_processor.detect_language(question)
            if question_language:
                print(f"Question language: {self.text_processor.get_language_name(question_language)}")
                
                if question_language not in self.model_info['languages']:
                    print(f"Warning: {self.text_processor.get_language_name(question_language)} "
                          f"is not fully supported by the selected model")
                
                if question_language != self.document_language:
                    print(f"Note: Question language ({self.text_processor.get_language_name(question_language)}) "
                          f"differs from document language ({self.text_processor.get_language_name(self.document_language)})")
            
            # Search with memory optimization
            results = self.vector_store.search(question, n_results=n_results)
            
            # Update performance metrics
            self.total_queries += 1
            if results and len(results['documents']) > 0:
                self.successful_queries += 1
            
            end_time = time.time()
            response_time = end_time - start_time
            self.response_times.append(response_time)
            
            # Add language pair
            if question_language and self.document_language:
                self.language_pairs.append((question_language, self.document_language))
            
            return results
            
        finally:
            # Cleanup and metrics
            self.memory_manager.optimize(self.MEMORY_SETTINGS['memory_threshold_mb'])
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            self.memory_manager.end_tracking("question_answering")
            
    def add_feedback(self, question: str, answer: Any, relevance_score: float):
        """Add user feedback for a question-answer pair."""
        # Convert answer to string if it's a list
        if isinstance(answer, list):
            answer = str(answer[0]) if answer else ""
            
        feedback = Feedback(
            question=question,
            answer=answer,
            relevance_score=relevance_score,
            question_language=self.text_processor.detect_language(question),
            document_language=self.document_language,
            model_type=self.model_type,
            timestamp=datetime.now(),
            metadata={
                'memory_usage': self.memory_manager.get_current_usage(),
                'device': self.device,
                'collection': self.collection_name
            }
        )
        
        self.feedback_system.add_feedback(feedback)
        self.relevance_scores.append(relevance_score)
        
    def save_performance_metrics(self):
        """Save current performance metrics."""
        if not self.response_times:
            return
            
        metrics = PerformanceMetrics(
            model_type=self.model_type,
            avg_response_time=np.mean(self.response_times),
            avg_relevance_score=np.mean(self.relevance_scores) if self.relevance_scores else 0,
            language_pairs=list(set(self.language_pairs)),
            total_queries=self.total_queries,
            successful_queries=self.successful_queries,
            memory_usage=self.memory_manager.get_current_usage(),
            timestamp=datetime.now()
        )
        
        self.feedback_system.save_metrics(metrics)

    def process_pdf(self, file_path: str) -> Tuple[List[str], int]:
        """
        Process a PDF file, extract text and split into chunks.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple[List[str], int]: List of text chunks and page count
        """
        try:
            # Extract text from PDF
            page_nums, page_texts = self.extract_text(file_path)
            
            # Get page count
            page_count = len(page_texts)
            
            # Process texts into chunks
            all_chunks = []
            for text in page_texts:
                if text.strip():  # Skip empty texts
                    chunks = self._split_into_chunks(
                        text, 
                        chunk_size=self.MEMORY_SETTINGS['chunk_size'], 
                        overlap=self.MEMORY_SETTINGS['chunk_overlap']
                    )
                    all_chunks.extend(chunks)
            
            return all_chunks, page_count
            
        except Exception as e:
            logging.error(f"Error processing PDF: {e}")
            raise ProcessingError(f"Failed to process PDF: {str(e)}") 