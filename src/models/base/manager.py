"""
Model management and integration module.
Coordinates all model components.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import torch
import numpy as np
from loguru import logger
import re

from .config import ModelConfig
from ..neural.graph_neural_network import GraphNeuralNetwork, GraphBuilder
from ..neural.online_learner import OnlineLearner
from ...processing.document import DocumentProcessor
from ...processing.text import TextProcessor
from ...storage.knowledge import KnowledgeGraph
from ...interface.feedback import FeedbackManager

class ModelManager:
    """Model management and integration."""
    
    def __init__(self, config: ModelConfig):
        """
        Initializes the model manager.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.logger = logger.bind(module="ModelManager")
        
        # Submodules
        self.document_processor = DocumentProcessor(config.to_dict())
        self.text_processor = TextProcessor(config.to_dict())
        self.online_learner = OnlineLearner(config.to_dict())
        self.graph_builder = GraphBuilder(config.to_dict())
        self.graph_network = GraphNeuralNetwork(config.to_dict())
        self.knowledge_graph = KnowledgeGraph()
        self.feedback_manager = FeedbackManager()
        
    async def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Processes document and adds it to the knowledge base.
        
        Args:
            document_path: Document file path
            
        Returns:
            Processing results
            
        Raises:
            Exception: If an error occurs during processing
        """
        try:
            # Process document
            doc_data = await self.document_processor.process_document(document_path)
            
            # Process texts
            processed_texts = await self.text_processor.process_batch(
                doc_data["chunks"]
            )
            
            # Update graph structure
            self._update_knowledge_graph(processed_texts)
            
            # Update online model
            await self._update_online_model(processed_texts)
            
            # Update knowledge graph
            doc_id = Path(document_path).stem
            await self.knowledge_graph.add_document(doc_id, processed_texts)
            
            return {
                "status": "success",
                "document_id": doc_id,
                "stats": self._calculate_stats(processed_texts)
            }
            
        except Exception as e:
            self.logger.error(f"Document processing error: {str(e)}")
            raise
    
    async def answer_question(self, 
                            question: str, 
                            context_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Answers the question.
        
        Args:
            question: Question text
            context_size: Number of similar sentences to use
            
        Returns:
            Answer and related information
            
        Raises:
            Exception: If an error occurs during processing
        """
        try:
            # Process the question
            processed_question = await self.text_processor.process_text(question)
            
            # Find the most relevant sentences
            relevant_contexts = await self._find_relevant_contexts(
                processed_question.embeddings,
                context_size or self.config.context_size
            )
            
            # Analyze on graph
            graph_context = self._analyze_graph_context(
                processed_question.embeddings
            )
            
            # Online model prediction
            prediction, confidence = await self.online_learner.predict(question)
            
            # Generate answer
            answer = self._generate_answer(
                question,
                relevant_contexts,
                graph_context,
                prediction,
                confidence
            )
            
            # Save feedback
            await self.feedback_manager.record_interaction(
                question,
                answer,
                confidence
            )
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Question answering error: {str(e)}")
            raise
    
    def _update_knowledge_graph(self, processed_texts: List[Any]):
        """Updates the graph structure."""
        # Create nodes and edges
        nodes = []
        edges = []
        
        for text in processed_texts:
            # Add sentences as nodes
            nodes.extend(text.sentences)
            
            # Create edges between consecutive sentences
            for i in range(len(text.sentences) - 1):
                edges.append((
                    text.sentences[i],
                    text.sentences[i + 1]
                ))
            
            # Create edges between similar sentences
            similarities = self._calculate_sentence_similarities(
                text.sentences
            )
            for i, j in similarities:
                edges.append((
                    text.sentences[i],
                    text.sentences[j]
                ))
        
        # Create graph
        self.knowledge_graph.update_graph(nodes, edges)
    
    async def _update_online_model(self, processed_texts: List[Any]):
        """Updates the online model."""
        for text in processed_texts:
            # For each sentence
            for sentence in text.sentences:
                # Labeling
                label = self._generate_simple_label(sentence)
                
                # Update model
                await self.online_learner.update(sentence, label)
    
    async def _find_relevant_contexts(self,
                                    query_embedding: torch.Tensor,
                                    k: int) -> List[Dict[str, Any]]:
        """Finds the most relevant sentences."""
        relevant_contexts = []
        
        # Scan all documents
        documents = await self.knowledge_graph.get_documents()
        for doc_id, doc_data in documents.items():
            for text in doc_data.processed_texts:
                # Calculate similarity
                similarities = torch.nn.functional.cosine_similarity(
                    query_embedding,
                    text.embeddings
                )
                
                # Get highest similarities
                top_k = torch.topk(similarities, min(k, len(similarities)))
                
                for idx, score in zip(top_k.indices, top_k.values):
                    relevant_contexts.append({
                        "text": text.sentences[idx],
                        "score": score.item(),
                        "document_id": doc_id,
                        "features": text.features
                    })
        
        # Sort by scores
        relevant_contexts.sort(key=lambda x: x["score"], reverse=True)
        
        return relevant_contexts[:k]
    
    def _analyze_graph_context(self, 
                             query_embedding: torch.Tensor) -> Dict[str, Any]:
        """Performs analysis on the graph."""
        graph = self.knowledge_graph.get_graph()
        if not graph:
            return {}
        
        # Run GNN on the graph
        graph_embeddings = self.graph_network(graph)
        
        # Compare query with graph nodes
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding,
            graph_embeddings
        )
        
        # Find the most relevant subgraph
        relevant_nodes = torch.topk(similarities, k=5)
        
        return {
            "relevant_nodes": relevant_nodes,
            "graph_state": graph_embeddings.mean(dim=0)
        }
    
    def _generate_answer(self,
                        question: str,
                        contexts: List[Dict[str, Any]],
                        graph_context: Dict[str, Any],
                        prediction: Any,
                        confidence: float) -> Dict[str, Any]:
        """Generates the answer."""
        # Combine the most relevant sentences
        context_text = " ".join([
            ctx["text"] for ctx in contexts
        ])
        
        answer = {
            "question": question,
            "answer_text": context_text,
            "confidence": confidence,
            "prediction": prediction,
            "contexts": contexts,
            "graph_analysis": graph_context,
            "timestamp": datetime.now().isoformat()
        }
        
        return answer
    
    def _calculate_stats(self, processed_texts: List[Any]) -> Dict[str, Any]:
        """Calculates statistics from processed texts."""
        total_sentences = 0
        total_tokens = 0
        total_chars = 0
        entities = {}
        
        for text in processed_texts:
            total_sentences += len(text.sentences)
            total_tokens += sum(len(s.split()) for s in text.sentences)
            total_chars += sum(len(s) for s in text.sentences)
            
            # Collect entities
            for entity_type, values in text.entities.items():
                if entity_type not in entities:
                    entities[entity_type] = set()
                entities[entity_type].update(values)
        
        return {
            "sentences": total_sentences,
            "tokens": total_tokens,
            "chars": total_chars,
            "entities": {k: list(v) for k, v in entities.items()}
        }
    
    def _calculate_sentence_similarities(self, 
                                      sentences: List[str],
                                      threshold: Optional[float] = None) -> List[Tuple[int, int]]:
        """Calculates similarity between sentence pairs."""
        if not threshold:
            threshold = self.config.similarity_threshold
            
        sentence_pairs = []
        
        # Calculate similarity for each sentence pair
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                similarity = self.text_processor.calculate_similarity(
                    sentences[i], sentences[j]
                )
                if similarity > threshold:
                    sentence_pairs.append((i, j))
        
        return sentence_pairs
    
    def _generate_simple_label(self, text: str) -> str:
        """Generates a simple label for the text."""
        # TODO: More advanced labeling strategy can be added
        for entity_type, pattern in self.config.entity_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return entity_type
        
        return "general" 