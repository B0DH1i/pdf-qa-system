"""
Text processing module.
This module contains text analysis, language detection, text normalization, and entity recognition functions.
"""

from typing import Any, Dict, List, Optional, Tuple, Set
import logging
from dataclasses import dataclass
import re
from pathlib import Path
import unicodedata
from collections import Counter, OrderedDict, defaultdict
import threading
import gc
import json
from datetime import datetime
import numpy as np
import random
import hashlib

from loguru import logger
from transformers import AutoTokenizer, AutoModel
import torch
from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from torch.nn import functional as F
import time

from src.core.exceptions import ProcessingError
from src.core.memory import MemoryManager
from sentence_transformers import SentenceTransformer

# Set seed for consistent results
DetectorFactory.seed = 0
random.seed(42)

@dataclass
class TextChunk:
    """Data class for text chunk."""
    text: str
    start: int
    end: int
    metadata: Dict[str, Any]

@dataclass
class TextStats:
    """Data class for text statistics."""
    char_count: int
    word_count: int
    sentence_count: int
    avg_word_length: float
    avg_sentence_length: float
    unique_words: int
    language: str
    language_confidence: float

@dataclass
class ProcessedText:
    """Data class for processed text."""
    raw_text: str
    cleaned_text: str
    sentences: List[str]
    embeddings: np.ndarray
    features: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class TokenStats:
    """Data class for token statistics."""
    total_tokens: int
    unique_tokens: int
    token_frequency: Dict[str, int]

class LRUCache:
    """Simple Least Recently Used (LRU) cache implementation."""
    
    def __init__(self, capacity: int = 5):
        """Initialize cache with specified capacity."""
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Any:
        """Get item from cache, moving it to the most recently used position."""
        if key not in self.cache:
            return None
        value = self.cache.pop(key)
        self.cache[key] = value
        return value
    
    def set(self, key: str, value: Any):
        """Add item to cache, removing least recently used item if capacity is reached."""
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value
    
    def clear(self):
        """Clear all items from cache."""
        self.cache.clear()

class TextProcessor:
    """
    Class that performs various NLP operations on text data.
    Performs operations such as embedding creation, text classification,
    and entity recognition.
    """
    
    # Regex patterns for date formats
    DATE_PATTERNS = {
        'formal': r'\d{1,2}[./]\d{1,2}[./]\d{2,4}',
        'written': r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
        'relative': r'(?i)(yesterday|today|tomorrow|last week|next week)',
    }
    
    # Regex patterns for number formats
    NUMBER_PATTERNS = {
        'integer': r'\d+',
        'decimal': r'\d+[.,]\d+',
        'written': r'(?i)(one|two|three|four|five|six|seven|eight|nine|ten)',
        'percentage': r'%\d+|\d+\s*%',
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize TextProcessor.
        
        Args:
            config: Configuration dictionary
        """
        # Set up configurations
        self._config = config or {}
        self.cache = {}
        self.entity_cache = {}
        
        # Regex patterns for date formats
        self.date_patterns = [
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}\.\d{2}\.\d{4}',
            r'\d{4}-\d{2}-\d{2}'
        ]
        
        # Regex patterns for number formats
        self.integer_patterns = [r'\b\d+\b']
        self.decimal_patterns = [r'\b\d+\.\d+\b', r'\b\d+,\d+\b']
        self.percentage_patterns = [r'\b\d+%\b', r'\b\d+\.\d+%\b', r'\b\d+,\d+%\b']
        self.written_number_patterns = [r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b']
        self.written_date_patterns = [r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b']
        self.relative_date_patterns = [r'\b(today|tomorrow|yesterday)\b']
        
        # Load model
        self._load_model()
        
        # Regex patterns for preprocessing
        self.url_pattern = r'https?://\S+|www\.\S+'
        self.email_pattern = r'\S+@\S+\.\S+'
        self.number_pattern = r'\d+'
        
        # Custom patterns for entity extraction
        self.custom_patterns = {}
        
        # NLP model for entity extraction
        self.nlp_model = None
        
        try:
            import spacy
            self.nlp_model = spacy.load("en_core_web_sm")
        except Exception as e:
            logging.warning(f"Failed to load SpaCy model: {e}")
            self.nlp_model = None
        
    def get_embedding_dimension(self):
        """Return the embedding dimension."""
        return self.embedding_dimension
    
    def _load_model(self):
        """
        Loads a SentenceTransformer model.
        
        Returns:
            The loaded SentenceTransformer model.
        """
        try:
            # Get model name from config or use default
            model_name = self._config.get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded with embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ProcessingError(f"Error loading model: {str(e)}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Creates embedding vectors for the given list of texts.
        
        Args:
            texts: List of texts to create embeddings for.
            
        Returns:
            Numpy array of embedding vectors.
        """
        try:
            if not texts:
                logger.warning("Empty text list provided for embedding")
                return np.array([])
                
            # Check for non-empty texts
            non_empty_texts = []
            for text in texts:
                if text and text.strip():
                    non_empty_texts.append(text)
                else:
                    logger.warning("Empty text encountered, skipping")
            
            if not non_empty_texts:
                logger.warning("No valid texts to embed")
                return np.array([])
                
            # Direct encoding - not using Dask/Distributed
            embeddings = self.model.encode(non_empty_texts, show_progress_bar=False)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise ProcessingError(f"Error generating embeddings: {str(e)}")
    
    def process_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Process a batch of texts."""
        if not texts:
            return []
        
        try:
            # Using a simple loop instead of dask/distributed
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_results = []
                
                # Process each text
                for text in batch:
                    processed = self.process_text(text)
                    batch_results.append(processed)
                
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise ProcessingError(f"Failed to process text batch: {str(e)}")
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
        """
        Divides text into overlapping chunks.
        
        Args:
            text: Text to be divided.
            chunk_size: Maximum character count for each chunk.
            overlap: Amount of overlap between chunks.
            
        Returns:
            List of divided text chunks.
        """
        try:
            if not text:
                return []
                
            chunks = []
            start = 0
            
            while start < len(text):
                # Calculate the end of the chunk
                end = min(start + chunk_size, len(text))
                
                # If we're cutting in the middle of a sentence, extend to the end of the sentence
                if end < len(text):
                    # Find the end of the sentence
                    possible_ends = [text.find('. ', end-overlap, end+overlap)]
                    possible_ends += [text.find('! ', end-overlap, end+overlap)]
                    possible_ends += [text.find('? ', end-overlap, end+overlap)]
                    
                    # Find the closest sentence end
                    valid_ends = [e for e in possible_ends if e != -1]
                    if valid_ends:
                        end = max(valid_ends) + 2  # Include the period and space
                
                # Add the chunk
                chunks.append(text[start:end])
                
                # Calculate the start of the next chunk
                start = end - overlap
                
            return chunks
        except Exception as e:
            logging.error(f"Error in chunking text: {str(e)}")
            return []
    
    def clean_text(self, text: str) -> str:
        """
        Performs basic text cleaning operations.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Simple cleaning operations
        if not text:
            return ""
        text = ' '.join(text.split())  # Remove extra spaces
    
        return text
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text.
        
        Args:
            text: Text to detect language for
            
        Returns:
            str: Detected language code (en, tr, fr, etc.)
        """
        try:
            # For this example, only return en (English) or tr (Turkish)
            if 'tr' in detect_langs(text):
                return 'tr'
                
            # If there are Turkish characters, it's probably Turkish
            special_chars = 'ıİğĞüÜşŞöÖçÇ'
            
            # If there are Turkish characters, it's probably Turkish
            if any(c in special_chars for c in text):
                return "tr"
            return "en"  # Default to English
        except Exception as e:
            logging.warning(f"Language detection failed: {e}")
            return "en"  # Default to English if there's a problem
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            self.model = None
        torch.cuda.empty_cache()
        gc.collect()
    
    def normalize_text(self, text: str) -> str:
        """
        Normalizes text for language detection.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        try:
            # Remove URLs
            text = re.sub(r'http\S+|www.\S+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove numbers
            text = re.sub(r'\d+', '', text)
            
            # Normalize Unicode characters
            text = unicodedata.normalize('NFKC', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text
        except Exception as e:
            logger.error(f"Text normalization error: {str(e)}")
            raise ProcessingError(f"Text normalization error: {str(e)}")
    
    def _check_special_chars(self, text: str) -> Dict[str, float]:
        """
        Checks for language-specific special characters.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Language codes and confidence scores
        """
        total_len = len(text)
        if total_len == 0:
            return {}
            
        scores = {}
        for lang, pattern in self.lang_patterns.items():
            matches = len(re.findall(pattern, text))
            if matches > 0:
                scores[lang] = matches / total_len
                
        return scores
    
    def _check_common_words(self, text: str) -> Dict[str, float]:
        """
        Checks for common words in different languages.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Language codes and confidence scores
        """
        text = text.lower()
        words = set(re.findall(r'\b\w+\b', text))
        scores = {}
        
        for lang, common_words in self.common_words.items():
            matches = sum(1 for word in words if word in common_words)
            if matches > 0:
                scores[lang] = matches / len(words)
                
        return scores
    
    def _get_multiple_samples(self, text: str, num_samples: int = 3) -> List[str]:
        """
        Takes multiple samples from text for more accurate detection.
        
        Args:
            text (str): Input text
            num_samples (int): Number of samples to take
            
        Returns:
            List[str]: List of text samples
        """
        words = text.split()
        if len(words) <= num_samples:
            return [text]
            
        sample_size = len(words) // num_samples
        samples = []
        
        for i in range(num_samples):
            start = i * sample_size
            end = start + sample_size
            sample = ' '.join(words[start:end])
            if sample.strip():
                samples.append(sample)
                
        return samples
    
    def get_language_name(self, lang_code: str) -> str:
        """
        Returns the language name corresponding to the language code.
        
        Args:
            lang_code (str): ISO 639-1 language code
            
        Returns:
            str: Language name
        """
        if lang_code in self.language_names:
            return self.language_names[lang_code][0]
        return lang_code
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generates embedding for text.
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: Text embedding
        """
        try:
            self.memory_manager.start_tracking("embedding_generation")
            
            # Check if in cache
            cache_key = hash(text)
            cached_embedding = self.embedding_cache.get(cache_key)
            if cached_embedding is not None:
                return cached_embedding
            
            # Generate embedding
            if not self.vectorizer_fitted:
                self.vectorizer.fit([text])
                self.vectorizer_fitted = True
                embedding = self.vectorizer.transform([text]).toarray()[0]
            else:
                embedding = self.vectorizer.transform([text]).toarray()[0]
            
            # Add result to cache
            self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        finally:
            self.memory_manager.end_tracking("embedding_generation")
    
    def _process_batch_chunk(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Processes text chunks in parallel."""
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self.process_text, texts))
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into sentences.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def preprocess_text(self, text: str) -> str:
        """
        Applies all preprocessing steps.
        
        Args:
            text (str): Text to process
            
        Returns:
            str: Processed text
        """
        text = self.clean_text(text)
        return text
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Processes a single text.
        
        Args:
            text (str): Text to process
            
        Returns:
            Dict[str, Any]: Processed text results
        """
        try:
            self.memory_manager.start_tracking("process_text")
            
            # Check for empty text
            if not text.strip():
                return {
                    'text': '',
                    'processed': False,
                    'error': 'Empty text',
                    'length': 0,
                    'words': 0
                }
            
            # Clean and preprocess the text
            cleaned_text = self.preprocess_text(text)
            sentences = self.split_into_sentences(cleaned_text)
            words = cleaned_text.split()
            
            # Generate embedding
            embedding = self._generate_embedding(cleaned_text)
            
            # Create processed text object
            result = ProcessedText(
                raw_text=text,
                cleaned_text=cleaned_text,
                sentences=sentences,
                embeddings=embedding,
                features={},
                metadata={
                    'timestamp': datetime.now(),
                    'length': len(text),
                    'num_sentences': len(sentences),
                    'num_words': len(words)
                }
            )
            
            return {
                'text': text,
                'processed': True,
                'result': result,
                'length': len(text),
                'words': len(words),
                'sentences': len(sentences)
            }
            
        finally:
            self.memory_manager.end_tracking("process_text")
            gc.collect()
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Detects entities in the text.
        
        Args:
            text (str): Text to process
            
        Returns:
            List[Dict[str, str]]: Detected entities
        """
        if text is None:
            raise TypeError("Text cannot be None")
            
        if not isinstance(text, str):
            raise TypeError("Text must be a string")
            
        if not text.strip():
            return []
            
        try:
            self.memory_manager.start_tracking("entity_extraction")
            
            # Check if in cache
            cache_key = hash(text)
            with self._lock:
                if cache_key in self.entity_cache:
                    self.stats['cache_hits'] += 1
                    return self.entity_cache[cache_key]
                    
            # Process the text
            doc = self.nlp(text)
            
            # Collect entities
            entities = []
            for ent in doc.ents:
                entity = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'description': spacy.explain(ent.label_)
                }
                entities.append(entity)
                
            # Add result to cache
            with self._lock:
                self.entity_cache[cache_key] = entities
                self._clean_entity_cache()
                self.stats['cache_misses'] += 1
                
            return entities
            
        finally:
            self.memory_manager.end_tracking("entity_extraction")
    
    def _clean_entity_cache(self):
        """Cleans the entity cache."""
        if len(self.entity_cache) > self.cache_size:
            # Remove oldest entries
            while len(self.entity_cache) > self.cache_size * 0.8:  # Leave 20% free space
                self.entity_cache.popitem()
    
    def extract_dates(self, text: str) -> List[str]:
        """
        Extract date entities from text.
        
        Args:
            text: Text to extract dates from
            
        Returns:
            List of extracted date strings
        """
        dates = []
        
        # Formal dates
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                date_str = match.group(0)
                if date_str not in dates:
                    dates.append(date_str)
        
        # Written dates
        for pattern in self.written_date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(0)
                if date_str not in dates:
                    dates.append(date_str)
        
        # Relative dates
        for pattern in self.relative_date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(0)
                if date_str not in dates:
                    dates.append(date_str)
        
        return dates
    
    def extract_numbers(self, text: str) -> List[str]:
        """
        Extract number entities from text.
        
        Args:
            text: Text to extract numbers from
            
        Returns:
            List of extracted number strings
        """
        numbers = []
        
        # Integers
        for pattern in self.integer_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                num_str = match.group(0)
                if num_str not in numbers:
                    numbers.append(num_str)
        
        # Decimal numbers
        for pattern in self.decimal_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                num_str = match.group(0)
                if num_str not in numbers:
                    numbers.append(num_str)
        
        # Written numbers
        for pattern in self.written_number_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                num_str = match.group(0)
                if num_str not in numbers:
                    numbers.append(num_str)
        
        # Percentages
        for pattern in self.percentage_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                num_str = match.group(0)
                if num_str not in numbers:
                    numbers.append(num_str)
        
        return numbers
    
    def custom_entity_recognition(self, 
                                text: str, 
                                patterns: List[Dict],
                                context_window: int = 5) -> List[Dict[str, str]]:
        """
        Detects domain-specific entities.
        
        Args:
            text (str): Text to process
            patterns (List[Dict]): Custom entity recognition patterns
            context_window (int): Context window size
            
        Returns:
            List[Dict[str, str]]: Detected custom entities
        """
        if not text or not patterns:
            return []
            
        try:
            self.memory_manager.start_tracking("custom_entity_recognition")
            
            # Add custom patterns
            ruler = self.nlp.get_pipe("entity_ruler") if "entity_ruler" in self.nlp.pipe_names else self.nlp.add_pipe("entity_ruler")
            ruler.add_patterns(patterns)
            
            # Process the text
            doc = self.nlp(text)
            
            # Collect entities
            entities = []
            for ent in doc.ents:
                # Get context window
                start = max(0, ent.start - context_window)
                end = min(len(doc), ent.end + context_window)
                context = doc[start:end].text
                
                entity = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'context': context,
                    'description': spacy.explain(ent.label_)
                }
                entities.append(entity)
                
            return entities
            
        finally:
            self.memory_manager.end_tracking("custom_entity_recognition")
    
    def clear_cache(self):
        """Clears all caches."""
        self.embedding_cache.clear()
        self.entity_cache.clear()
        gc.collect()
    
    def _lock(self):
        """Return a lock object for thread-safe operations."""
        return threading.Lock()
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with URLs removed
        """
        # Remove URLs
        return re.sub(self.url_pattern, ' ', text)
        
    def remove_emails(self, text: str) -> str:
        """
        Remove email addresses from text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with email addresses removed
        """
        # Remove email addresses
        return re.sub(self.email_pattern, ' ', text)
        
    def remove_numbers(self, text: str) -> str:
        """
        Remove numbers from text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with numbers removed
        """
        # Remove numbers
        return re.sub(self.number_pattern, ' ', text)
        
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters in text.
        
        Args:
            text: Text to normalize
            
        Returns:
            Text with normalized Unicode characters
        """
        # Normalize Unicode characters
        return unicodedata.normalize('NFKD', text)
        
    def remove_extra_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace from text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with extra whitespace removed
        """
        # Remove extra spaces
        return re.sub(r'\s+', ' ', text).strip()
    
    def preprocess_and_embed(self, text: str) -> Optional[np.ndarray]:
        """
        Preprocess text and create embeddings.
        
        Args:
            text: Text to process
            
        Returns:
            Embedding vector for the text, or None if processing fails
        """
        try:
            # Check if it's in the cache
            cache_key = hashlib.md5(text.encode()).hexdigest()
            with self._get_lock():
                if cache_key in self.cache:
                    return self.cache[cache_key]
                    
            # Create embedding
            clean_text = self.clean_text(text)
            if not clean_text:
                return None
                
            embedding = self.get_embeddings([clean_text])[0]
            
            # Add result to cache
            with self._get_lock():
                self.cache[cache_key] = embedding
                
            return embedding
        except Exception as e:
            logging.error(f"Error preprocessing and embedding text: {e}")
            return None
            
    def process_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Process multiple texts and create embeddings.
        
        Args:
            texts: List of texts to process
            batch_size: Size of processing batches
            
        Returns:
            List of embedding vectors
        """
        # Using a simple loop instead of Dask/distributed
        results = []
        
        try:
            # Process each text
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                embeddings = self.process_batch(batch, batch_size)
                results.extend(embeddings)
        except Exception as e:
            logging.error(f"Error processing texts: {e}")
            
        return results

    def extract_entities_with_cache(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text with caching.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entity types and their values
        """
        # Check if it's in the cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        with self._get_lock():
            if cache_key in self.entity_cache:
                return self.entity_cache[cache_key]
        
        # Process the text
        result = self.extract_entities(text)
        
        # Collect entities
        entities = {
            "dates": self.extract_dates(text),
            "numbers": self.extract_numbers(text),
            **result
        }
        
        # Add result to cache
        with self._get_lock():
            self.entity_cache[cache_key] = entities
            
            # Clean up old entries
            while len(self.entity_cache) > self.cache_size * 0.8:  # Keep 20% free space
                oldest_key = next(iter(self.entity_cache))
                del self.entity_cache[oldest_key]
                
        return entities 