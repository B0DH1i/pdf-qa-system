"""
Context management module.
Manages conversation history and context window.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from ..storage.vector_store import VectorStore
from ..utils.token_counter import TokenCounter
from ..storage.knowledge import KnowledgeGraph

class ContextManager:
    def __init__(self,
                 max_tokens: int = 4096,
                 similarity_threshold: float = 0.7,
                 vector_store: Optional[VectorStore] = None,
                 knowledge_graph: Optional[KnowledgeGraph] = None,
                 model_name: str = "gpt-3.5-turbo"):
        """
        Initialize context manager
        
        Args:
            max_tokens: Maximum token count
            similarity_threshold: Similarity threshold
            vector_store: Vector database (if not provided, a new one will be created)
            knowledge_graph: Knowledge graph (if not provided, a new one will be created)
            model_name: Model name used
        """
        self.max_tokens = max_tokens
        self.similarity_threshold = similarity_threshold
        self.conversation_history = []
        
        # Initialize vector store
        self.vector_store = vector_store or VectorStore(
            collection_name="conversation_history",
            persist_directory="./data/context_db"
        )
        
        # Initialize token counter
        self.token_counter = TokenCounter(
            model_name=model_name,
            max_tokens=max_tokens
        )
        
        # Initialize knowledge graph
        self.knowledge_graph = knowledge_graph or KnowledgeGraph(
            storage_path="./data/knowledge_graph.json",
            similarity_threshold=similarity_threshold
        )
        
    def add_interaction(self,
                       user_input: str,
                       assistant_response: str,
                       user_embedding: List[float],
                       assistant_embedding: List[float],
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new interaction
        
        Args:
            user_input: User input
            assistant_response: Assistant response
            user_embedding: User input embedding
            assistant_embedding: Assistant response embedding
            metadata: Additional metadata
            
        Returns:
            Interaction ID
        """
        # Create metadata
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "type": "interaction"
        })
        
        # Add new interaction to vector store
        interaction_id = self.vector_store.add_documents(
            texts=[user_input, assistant_response],
            embeddings=[user_embedding, assistant_embedding],
            metadata=[
                {**metadata, "role": "user"},
                {**metadata, "role": "assistant"}
            ]
        )[0]  # Return the first ID
        
        # Prepare new interaction
        new_interaction = {
            "id": interaction_id,
            "user_input": user_input,
            "assistant_response": assistant_response,
            "timestamp": metadata["timestamp"]
        }
        
        # Calculate new interaction token count
        new_tokens = self.token_counter.count_tokens(new_interaction)
        
        # Optimize context window
        optimized_history = self.token_counter.optimize_context(
            self.conversation_history,
            new_message_tokens=new_tokens
        )
        
        # Remove interactions from knowledge graph
        removed_interactions = set(i["id"] for i in self.conversation_history) - set(i["id"] for i in optimized_history)
        for interaction_id in removed_interactions:
            user_node_id = f"user_{interaction_id}"
            assistant_node_id = f"assistant_{interaction_id}"
            if user_node_id in self.knowledge_graph.graph:
                self.knowledge_graph.graph.remove_node(user_node_id)
            if assistant_node_id in self.knowledge_graph.graph:
                self.knowledge_graph.graph.remove_node(assistant_node_id)
        
        # Add new interaction to optimized history
        self.conversation_history = optimized_history + [new_interaction]
        
        # Add to knowledge graph
        self._add_to_knowledge_graph(
            interaction_id=interaction_id,
            user_input=user_input,
            assistant_response=assistant_response,
            user_embedding=user_embedding,
            assistant_embedding=assistant_embedding,
            metadata=metadata
        )
        
        return interaction_id
    
    def _add_to_knowledge_graph(self,
                              interaction_id: str,
                              user_input: str,
                              assistant_response: str,
                              user_embedding: List[float],
                              assistant_embedding: List[float],
                              metadata: Dict[str, Any]) -> None:
        """
        Add interaction to knowledge graph
        
        Args:
            interaction_id: Interaction ID
            user_input: User input
            assistant_response: Assistant response
            user_embedding: User input embedding
            assistant_embedding: Assistant response embedding
            metadata: Metadata
        """
        # Add user and assistant nodes
        user_node_id = f"user_{interaction_id}"
        assistant_node_id = f"assistant_{interaction_id}"
        
        self.knowledge_graph.add_entity(
            entity_id=user_node_id,
            entity_type="user_message",
            properties={
                "text": user_input,
                "timestamp": metadata["timestamp"]
            },
            embedding=user_embedding
        )
        
        self.knowledge_graph.add_entity(
            entity_id=assistant_node_id,
            entity_type="assistant_message",
            properties={
                "text": assistant_response,
                "timestamp": metadata["timestamp"]
            },
            embedding=assistant_embedding
        )
        
        # Add relations
        self.knowledge_graph.add_relation(
            source_id=user_node_id,
            target_id=assistant_node_id,
            relation_type="followed_by"
        )
        
        # Connect with previous messages
        if len(self.conversation_history) > 1:  # If there is at least one previous interaction
            last_interaction = self.conversation_history[-2]  # Not the last, but the second last
            last_assistant_node = f"assistant_{last_interaction['id']}"
            
            self.knowledge_graph.add_relation(
                source_id=last_assistant_node,
                target_id=user_node_id,
                relation_type="precedes"
            )
    
    def get_relevant_context(self,
                           query_embedding: List[float],
                           n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant context for query
        
        Args:
            query_embedding: Query embedding
            n_results: Number of results to return
            
        Returns:
            List of relevant context
        """
        # Find similar interactions
        results = self.vector_store.search_similar(
            query_embedding=query_embedding,
            n_results=n_results
        )
        
        relevant_context = []
        for i in range(len(results["documents"])):
            if results["distances"][i] <= self.similarity_threshold:
                relevant_context.append({
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "similarity": 1 - (results["distances"][i] / 2)  # Convert cosine distance to similarity
                })
                
        return relevant_context
    
    def get_conversation_history(self,
                               n_recent: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            n_recent: Get recent n interactions (None for all)
            
        Returns:
            Conversation history
        """
        if n_recent is None:
            return self.conversation_history
        return self.conversation_history[-n_recent:]
    
    def get_token_stats(self) -> Dict[str, int]:
        """
        Get token statistics
        
        Returns:
            Token statistics
        """
        return self.token_counter.get_token_stats(self.conversation_history)
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """
        Get knowledge graph statistics
        
        Returns:
            Knowledge graph statistics
        """
        return self.knowledge_graph.get_stats() 