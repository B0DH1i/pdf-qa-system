"""
Online learning module.
"""

from typing import Dict, Any, Tuple, List, Optional
from collections import deque
from datetime import datetime

import numpy as np
from river import (
    compose,
    feature_extraction,
    linear_model,
    preprocessing,
    metrics
)
from loguru import logger

class OnlineLearner:
    """
    Online learning model that updates continuously with new data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize online learner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("online_learning", {})
        self.window_size = self.config.get("window_size", 1000)
        self.learning_rate = self.config.get("learning_rate", 0.01)
        
        # Initialize model pipeline
        self.model = self._initialize_model()
        
        # Performance tracking
        self.metrics = {
            "accuracy": metrics.Accuracy(),
            "f1": metrics.F1(),
            "precision": metrics.Precision(),
            "recall": metrics.Recall()
        }
        
        # Memory window for recent samples
        self.memory = deque(maxlen=self.window_size)
        
    def _initialize_model(self):
        """Initialize the model pipeline."""
        return compose.Pipeline(
            ("vectorizer", feature_extraction.BagOfWords()),
            ("normalizer", preprocessing.StandardScaler()),
            ("classifier", linear_model.PAClassifier(
                C=self.learning_rate,
                mode=2
            ))
        )
    
    def update(self, text: str, label: str) -> Dict[str, float]:
        """
        Update the model with new data.
        
        Args:
            text: Input text
            label: True label
            
        Returns:
            Dictionary of updated metrics
        """
        try:
            # Make prediction before update
            pred = self.model.predict_one(text)
            
            # Update metrics
            if pred is not None:
                for metric in self.metrics.values():
                    metric.update(y_true=label, y_pred=pred)
            
            # Update model
            self.model.learn_one(text, label)
            
            # Store in memory
            self.memory.append({
                "text": text,
                "label": label,
                "prediction": pred,
                "timestamp": datetime.now().isoformat()
            })
            
            return self._get_metrics()
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            return {}
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Make prediction for new text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            # Get prediction
            pred = self.model.predict_one(text)
            
            # Get probability scores
            proba = self.model.predict_proba_one(text)
            confidence = max(proba.values()) if proba else 0.0
            
            return pred, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None, 0.0
    
    def batch_update(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """
        Update model with multiple samples.
        
        Args:
            texts: List of input texts
            labels: List of true labels
            
        Returns:
            Dictionary of updated metrics
        """
        metrics = {}
        for text, label in zip(texts, labels):
            metrics = self.update(text, label)
        return metrics
    
    def _get_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        return {
            name: metric.get() 
            for name, metric in self.metrics.items()
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory window."""
        if not self.memory:
            return {}
            
        labels = [sample["label"] for sample in self.memory]
        predictions = [sample["prediction"] for sample in self.memory]
        
        return {
            "samples_in_memory": len(self.memory),
            "label_distribution": {
                label: labels.count(label) / len(labels)
                for label in set(labels)
            },
            "accuracy": sum(l == p for l, p in zip(labels, predictions)) / len(labels)
        } 