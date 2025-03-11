"""
Vector database management advanced class.
Manages vector storage and search operations using ChromaDB.
Comes with PyTorch and NumPy support as well as advanced error handling and logging features.
"""

import logging
import numpy as np
import time
import os
import json
import uuid
import torch
import torch.nn.functional as F
import shutil
from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Tuple

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    logging.warning("ChromaDB not installed. Vector storage functionality will be limited.")

from ..core.exceptions import StorageError

class VectorStore:
    def __init__(
        self, 
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist the database to
            dimension: Dimension of the embeddings (optional)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_function = None
        self.logger = logging.getLogger(__name__)
        
        try:
            if persist_directory:
                if not os.path.exists(persist_directory):
                    os.makedirs(persist_directory)
                
                self.client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(allow_reset=True)
                )
            else:
                self.client = chromadb.Client(
                    settings=Settings(allow_reset=True)
                )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(collection_name)
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(collection_name)
                
            self.logger.info(f"VectorStore started: {collection_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {str(e)}")
            raise StorageError(f"Failed to initialize vector store: {str(e)}")
    
    def change_collection(self, collection_name: str) -> None:
        """
        Change active collection or create a new one
        
        Args:
            collection_name: New collection name
        """
        try:
            self.collection_name = collection_name
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            self.logger.info(f"Collection changed: {collection_name}")
        except Exception as e:
            self.logger.error(f"Error changing collection: {str(e)}")
            raise StorageError(f"Failed to change collection: {str(e)}")
    
    def _process_embedding(self, embedding: Union[np.ndarray, List[float], torch.Tensor]) -> List[List[float]]:
        """
        Convert embedding to appropriate format
        
        Args:
            embedding: Input embedding
            
        Returns:
            Processed embedding in the correct format
        """
        # Convert torch tensor to numpy if needed
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
            
        # Ensure it's a 2D array
        if isinstance(embedding, np.ndarray):
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            embedding = embedding.tolist()
        elif isinstance(embedding, list):
            if not isinstance(embedding[0], list):
                embedding = [embedding]
                
        return embedding
    
    def _process_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Process embeddings for storage.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            np.ndarray: Processed embeddings
        """
        try:
            # Convert to torch tensor if needed
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.from_numpy(embeddings).float()
                
            # Move to device
            embeddings = embeddings.to(self.device)
            
            # Normalize if needed
            if torch.norm(embeddings, dim=1).mean() != 1.0:
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
            # Move back to CPU for storage
            return embeddings.cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"Error processing embeddings: {str(e)}")
            raise StorageError(f"Failed to process embeddings: {str(e)}")
    
    def add_documents(self,
                     texts: List[str],
                     embeddings: Union[List[np.ndarray], np.ndarray],
                     metadata: Optional[List[Dict[str, Any]]] = None,
                     ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents and embeddings to the database
        
        Args:
            texts: Document texts
            embeddings: Embedding vectors (numpy array)
            metadata: Metadata for each document
            ids: Custom IDs (generated automatically if not provided)
            
        Returns:
            IDs of the added documents
            
        Raises:
            StorageError: If addition fails
        """
        try:
            # Handle inputs
            if not texts or (isinstance(embeddings, np.ndarray) and embeddings.size == 0):
                return []
                
            if len(texts) != len(embeddings):
                raise StorageError("Number of texts and embeddings must match")
                
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{int(time.time())}_{i}" for i in range(len(texts))]
                
            # Generate metadata if not provided
            if metadata is None:
                metadata = [{"timestamp": time.time()} for _ in range(len(texts))]
            
            # Process embeddings
            embeddings_array = np.array(embeddings)
            processed_embeddings = self._process_embeddings(embeddings_array)
            
            # Check collection, create if it doesn't exist
            if self.collection_name not in self.client.list_collections():
                self.logger.info(f"Creating new collection '{self.collection_name}'")
                self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,
                    metadata={"hnsw:space": "cosine"}
                )
                self.collection = self.client.get_collection(self.collection_name)
            
            # Add to database
            self.collection.add(
                documents=texts,
                embeddings=processed_embeddings.tolist(),
                metadatas=metadata,
                ids=ids
            )
            
            self.logger.info(f"Added {len(texts)} documents to collection {self.collection_name}")
            return ids
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise StorageError(f"Failed to add documents: {str(e)}")
    
    def search_similar(self, 
                     query_embedding: Union[np.ndarray, List[float], torch.Tensor],
                     n_results: int = 5,
                     where_filter: Optional[Dict[str, Any]] = None,
                     filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for similar vectors in the vector database
        
        Args:
            query_embedding: Query embedding
            n_results: Number of results to return
            where_filter: Filter for metadata (backward compatibility)
            filter_criteria: Alternative filter for metadata, alias for where_filter
            
        Returns:
            Search results
            
        Raises:
            StorageError: If the search operation fails
        """
        
        # Process the query embedding
        query_embedding = self._process_embedding(query_embedding)
        
        try:
            # Allow filter_criteria as an alias for where_filter for backward compatibility
            where_condition = where_filter
            if filter_criteria is not None:
                where_condition = filter_criteria
            
            # Search in the vector database
            query_params = {
                "query_embeddings": query_embedding,
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"]
            }
            
            # Only add where condition if it's not empty and contains valid operators
            if where_condition and len(where_condition) > 0:
                # ChromaDB expects certain operators in the where clause
                # Only add where if there are valid operators
                query_params["where"] = where_condition
            
            # Execute the query
            results = self.collection.query(**query_params)
            
            return results
        except Exception as e:
            error_msg = f"Error searching for similar documents: {str(e)}"
            self.logger.error(error_msg)
            raise StorageError(error_msg) from e
    
    def update_document(self, 
                       doc_id: str, 
                       text: Optional[str] = None,
                       embedding: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update an existing document
        
        Args:
            doc_id: ID of the document to update
            text: New text (if None, won't be updated)
            embedding: New embedding (if None, won't be updated)
            metadata: New metadata (if None, won't be updated)
            
        Raises:
            StorageError: If the update operation fails
        """
        try:
            update_dict = {}
            
            if text is not None:
                update_dict["documents"] = [text]
                
            if embedding is not None:
                update_dict["embeddings"] = [self._process_embedding(embedding)]
                
            if metadata is not None:
                update_dict["metadatas"] = [metadata]
            
            if update_dict:
                update_dict["ids"] = [doc_id]
                self.collection.update(**update_dict)
                self.logger.info(f"Document updated: {doc_id}")
            
        except Exception as e:
            raise StorageError(f"Document update error: {str(e)}")
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """
        Delete documents
        
        Args:
            doc_ids: IDs of documents to delete
            
        Raises:
            StorageError: If the deletion operation fails
        """
        try:
            self.collection.delete(ids=doc_ids)
            self.logger.info(f"{len(doc_ids)} documents deleted")
        except Exception as e:
            raise StorageError(f"Document deletion error: {str(e)}")
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Get document by ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document information
            
        Raises:
            StorageError: If the document is not found or retrieval operation fails
        """
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if not result["documents"]:
                raise StorageError(f"Document with ID {doc_id} not found")
                
            return {
                "id": doc_id,
                "document": result["documents"][0],
                "metadata": result["metadatas"][0],
                "embedding": result["embeddings"][0] if "embeddings" in result else None
            }
            
        except Exception as e:
            raise StorageError(f"Document retrieval error: {str(e)}")
    
    def count_documents(self) -> int:
        """
        Return the total document count
        
        Returns:
            Number of documents
            
        Raises:
            StorageError: If the counting operation fails
        """
        try:
            return self.collection.count()
        except Exception as e:
            raise StorageError(f"Document counting error: {str(e)}")
    
    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get collection statistics
        
        Args:
            collection_name: Optional collection name (uses current collection if None)
            
        Returns:
            Statistics about the collection
            
        Raises:
            StorageError: If getting statistics fails
        """
        try:
            # If a collection name is provided, try to get that collection
            if collection_name is not None and collection_name != self.collection_name:
                try:
                    collection = self.client.get_collection(collection_name)
                    return {
                        'name': collection.name,
                        'count': collection.count(),
                        'metadata': collection.metadata
                    }
                except Exception as e:
                    raise StorageError(f"Collection '{collection_name}' not found: {str(e)}")
            
            # Otherwise use the current collection
            return {
                'name': self.collection.name,
                'count': self.collection.count(),
                'metadata': self.collection.metadata
            }
        except Exception as e:
            raise StorageError(f"Collection statistics error: {str(e)}")
    
    def clear_collection(self) -> None:
        """
        Clear the entire collection
        
        Raises:
            StorageError: If the cleanup operation fails
        """
        try:
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
                self.logger.info("Collection cleared")
        except Exception as e:
            raise StorageError(f"Collection clearing error: {str(e)}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Collection cleanup
            if hasattr(self, 'collection') and self.collection is not None:
                try:
                    self.logger.info(f"Clearing collection: {self.collection_name}")
                    self.clear_collection()
                    self.logger.info(f"Collection cleared successfully: {self.collection_name}")
                except Exception as e:
                    self.logger.warning(f"Collection clearing error: {str(e)}")
            else:
                self.logger.info("No collection to clear")
            
            # Client reset
            if hasattr(self, 'client') and self.client is not None:
                try:
                    self.logger.info("Resetting client")
                    self.client.reset()
                    self.logger.info("Client reset successful")
                except Exception as e:
                    self.logger.warning(f"Client reset error: {str(e)}")
            else:
                self.logger.info("No client to reset")
                
            # Persistent directory cleanup
            if hasattr(self, 'persist_directory') and self.persist_directory is not None and os.path.exists(self.persist_directory):
                try:
                    self.logger.info(f"Removing directory: {self.persist_directory}")
                    shutil.rmtree(self.persist_directory, ignore_errors=True)
                    self.logger.info(f"Directory removed: {self.persist_directory}")
                except Exception as e:
                    self.logger.warning(f"Directory cleanup error: {str(e)}")
            elif hasattr(self, 'persist_directory') and self.persist_directory is None:
                self.logger.info("No persistent directory specified")
            
            self.logger.info("Vector store cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up vector store: {str(e)}")
    
    def add_vectors(self, 
                   vectors: np.ndarray, 
                   metadata: Optional[List[Dict[str, Any]]] = None,
                   ids: Optional[List[str]] = None) -> List[str]:
        """
        Add vectors to the database
        
        Args:
            vectors: Vectors to add (numpy array)
            metadata: Metadata for each vector (optional)
            ids: Custom IDs (will be generated automatically if not provided)
            
        Returns:
            IDs of the added vectors
            
        Raises:
            StorageError: If the addition operation fails
        """
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [f"vec_{int(time.time())}_{i}" for i in range(len(vectors))]
                
            # Generate metadata if not provided
            if metadata is None:
                metadata = [{"timestamp": time.time()} for _ in range(len(vectors))]
                
            # Process embeddings
            processed_vectors = self._process_embeddings(vectors)
            
            # Add to database
            self.collection.add(
                embeddings=processed_vectors.tolist(),
                documents=[""] * len(vectors),
                metadatas=metadata,
                ids=ids
            )
            
            self.logger.info(f"Added {len(vectors)} vectors to collection {self.collection_name}")
            return ids
            
        except Exception as e:
            self.logger.error(f"Error adding vectors: {str(e)}")
            raise StorageError(f"Failed to add vectors: {str(e)}")
            
    def add_vectors_batch(self,
                         vectors: List[np.ndarray],
                         batch_size: int = 100,
                         metadata: Optional[List[Dict[str, Any]]] = None,
                         ids: Optional[List[str]] = None) -> List[str]:
        """
        Add vectors to the database in batches
        
        Args:
            vectors: Vectors to add (numpy array)
            batch_size: Number of vectors in each batch
            metadata: Metadata for each vector (optional)
            ids: Custom IDs (will be generated automatically if not provided)
            
        Returns:
            IDs of the added vectors
            
        Raises:
            StorageError: If the addition operation fails
        """
        try:
            all_ids = []
            total_vectors = len(vectors)
            
            for i in range(0, total_vectors, batch_size):
                batch_vectors = vectors[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size] if metadata else None
                batch_ids = ids[i:i + batch_size] if ids else None
                
                batch_result = self.add_vectors(
                    vectors=batch_vectors,
                    metadata=batch_metadata,
                    ids=batch_ids
                )
                all_ids.extend(batch_result)
                
                self.logger.debug(f"Batch {i//batch_size + 1} completed")
                
            return all_ids
            
        except Exception as e:
            self.logger.error(f"Error adding vectors: {str(e)}")
            raise StorageError(f"Failed to add vectors: {str(e)}")
            
    def search_vectors(self,
                      query_vectors: List[np.ndarray],
                      n_results: int = 5,
                      filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the vector database
        
        Args:
            query_vectors: Query vectors (numpy array)
            n_results: Number of results to return for each query
            filter_criteria: Filtering criteria
            
        Returns:
            Similar vectors and metadata
            
        Raises:
            StorageError: If the search operation fails
        """
        try:
            # Convert query vectors to appropriate format
            processed_queries = self._process_embeddings(np.array(query_vectors))
            
            # Prepare query parameters
            query_params = {
                "query_embeddings": processed_queries.tolist(),
                "n_results": n_results,
                "include": ["metadatas", "distances", "embeddings"]
            }
            
            # If filter_criteria exists and is not empty, add where parameter
            if filter_criteria and len(filter_criteria) > 0:
                query_params["where"] = filter_criteria
            
            # Search for similar vectors
            results = self.collection.query(**query_params)
            
            # Format results
            formatted_results = []
            for i, query_results in enumerate(results["metadatas"]):
                query_matches = []
                for j, metadata in enumerate(query_results):
                    match = {
                        "metadata": metadata,
                        "distance": results["distances"][i][j],
                        "embedding": results["embeddings"][i][j] if "embeddings" in results else None
                    }
                    query_matches.append(match)
                formatted_results.append(query_matches)
                
            self.logger.debug(f"Search completed: {len(query_vectors)} queries with {n_results} results")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching vectors: {str(e)}")
            raise StorageError(f"Failed to search vectors: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.cleanup() 