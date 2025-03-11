"""
User feedback and performance metrics interface.
This module manages user feedback and model performance metrics.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import sqlite3
import logging

from ..core.exceptions import ProcessingError

@dataclass
class Feedback:
    """User feedback for question-answer pair."""
    question: str
    answer: str  # If the answer is a list, it is converted to string
    relevance_score: float  # User evaluation between 0-1
    question_language: str
    document_language: str
    model_type: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """Model performance metrics."""
    model_type: str
    avg_response_time: float
    avg_relevance_score: float
    language_pairs: List[tuple]
    total_queries: int
    successful_queries: int
    memory_usage: float
    timestamp: datetime

class FeedbackSystem:
    """Manages user feedback and model performance metrics."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initializes the feedback system."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Database path
        self.db_path = self.config.get(
            "db_path",
            "data/feedback.db"
        )
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initializes the SQLite database."""
        try:
            # Create database directory
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Feedback table
            c.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    relevance_score REAL NOT NULL,
                    question_language TEXT NOT NULL,
                    document_language TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    metadata TEXT NOT NULL
                )
            ''')
            
            # Performance metrics table
            c.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    avg_response_time REAL NOT NULL,
                    avg_relevance_score REAL NOT NULL,
                    language_pairs TEXT NOT NULL,
                    total_queries INTEGER NOT NULL,
                    successful_queries INTEGER NOT NULL,
                    memory_usage REAL NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            raise ProcessingError(f"Database initialization error: {str(e)}")
        
    def add_feedback(self, 
                    question: str,
                    answer: str,
                    relevance_score: float,
                    question_language: str = "en",
                    document_language: str = "en",
                    model_type: str = "default",
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Adds user feedback.
        
        Args:
            question: Question text
            answer: Answer text
            relevance_score: Relevance score (0-1 range)
            question_language: Question language (default: "en")
            document_language: Document language (default: "en")
            model_type: Model type (default: "default")
            metadata: Additional metadata (optional)
        """
        try:
            if metadata is None:
                metadata = {}
                
            feedback = Feedback(
                question=question,
                answer=answer,
                relevance_score=relevance_score,
                question_language=question_language,
                document_language=document_language,
                model_type=model_type,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''
                INSERT INTO feedback (
                    question, answer, relevance_score, 
                    question_language, document_language,
                    model_type, timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.question,
                feedback.answer,
                feedback.relevance_score,
                feedback.question_language,
                feedback.document_language,
                feedback.model_type,
                feedback.timestamp.isoformat(),
                json.dumps(feedback.metadata)
            ))
            
            # Update performance_metrics table
            # First, check if there is an existing record
            c.execute('''
                SELECT id FROM performance_metrics 
                WHERE model_type = ? 
                AND date(timestamp) = date('now')
            ''', (model_type,))
            
            result = c.fetchone()
            
            if result:
                # Update existing record
                c.execute('''
                    UPDATE performance_metrics 
                    SET total_queries = total_queries + 1,
                        successful_queries = CASE 
                            WHEN ? >= 0.5 THEN successful_queries + 1 
                            ELSE successful_queries 
                        END,
                        avg_relevance_score = (avg_relevance_score * total_queries + ?) / (total_queries + 1)
                    WHERE id = ?
                ''', (relevance_score, relevance_score, result[0]))
            else:
                # Create new performance metric record
                # Default start values for avg_response_time and memory_usage
                c.execute('''
                    INSERT INTO performance_metrics (
                        model_type, avg_response_time, avg_relevance_score,
                        language_pairs, total_queries, successful_queries,
                        memory_usage, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_type,
                    0.0,  # avg_response_time default value
                    relevance_score,
                    json.dumps([[question_language, document_language]]),
                    1,  # total_queries
                    1 if relevance_score >= 0.5 else 0,  # successful_queries
                    0.0,  # memory_usage default value
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info("Feedback added")
            
        except Exception as e:
            self.logger.error(f"Feedback addition error: {str(e)}")
            raise ProcessingError(f"Feedback addition error: {str(e)}")
        
    def add_performance_metrics(self, metrics: PerformanceMetrics):
        """Adds performance metrics."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''
                INSERT INTO performance_metrics (
                    model_type, avg_response_time, avg_relevance_score,
                    language_pairs, total_queries, successful_queries,
                    memory_usage, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.model_type,
                metrics.avg_response_time,
                metrics.avg_relevance_score,
                json.dumps(metrics.language_pairs),
                metrics.total_queries,
                metrics.successful_queries,
                metrics.memory_usage,
                metrics.timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info("Performance metrics added")
            
        except Exception as e:
            self.logger.error(f"Performance metrics addition error: {str(e)}")
            raise ProcessingError(f"Performance metrics addition error: {str(e)}")
        
    def get_model_performance(self, model_type: str, days: int = 7) -> Dict[str, Any]:
        """Gets model performance statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Performance metrics for the last n days
            # SQL query date format corrected
            c.execute('''
                SELECT
                    AVG(avg_response_time) as avg_response,
                    AVG(avg_relevance_score) as avg_relevance,
                    SUM(total_queries) as total_queries,
                    SUM(successful_queries) as successful_queries,
                    AVG(memory_usage) as avg_memory
                FROM performance_metrics
                WHERE model_type = ?
                AND timestamp >= datetime('now', '-' || ? || ' days')
            ''', (model_type, days))
            
            result = c.fetchone()
            
            if result:
                # None values set to 0
                total_queries = result[2] or 0
                successful_queries = result[3] or 0
                
                # Prevent division by zero
                success_rate = 0
                if total_queries > 0 and successful_queries is not None:
                    success_rate = (successful_queries / total_queries) * 100
                
                stats = {
                    'avg_response_time': result[0] or 0,
                    'avg_relevance_score': result[1] or 0,
                    'total_queries': total_queries,
                    'successful_queries': successful_queries,
                    'avg_memory_usage': result[4] or 0,
                    'success_rate': success_rate
                }
            else:
                stats = {
                    'avg_response_time': 0,
                    'avg_relevance_score': 0,
                    'total_queries': 0,
                    'successful_queries': 0,
                    'avg_memory_usage': 0,
                    'success_rate': 0
                }
                
            conn.close()
            return stats
            
        except Exception as e:
            self.logger.error(f"Model performance statistics retrieval error: {str(e)}")
            raise ProcessingError(f"Model performance statistics retrieval error: {str(e)}")
        
    def get_language_pair_performance(self, question_lang: str, doc_lang: str) -> Dict[str, float]:
        """Gets performance statistics for a specific language pair."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''
                SELECT AVG(relevance_score)
                FROM feedback
                WHERE question_language = ?
                AND document_language = ?
            ''', (question_lang, doc_lang))
            
            avg_score = c.fetchone()[0] or 0
            
            c.execute('''
                SELECT COUNT(*)
                FROM feedback
                WHERE question_language = ?
                AND document_language = ?
            ''', (question_lang, doc_lang))
            
            total_queries = c.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'avg_relevance_score': avg_score,
                'total_queries': total_queries
            }
            
        except Exception as e:
            self.logger.error(f"Language pair performance statistics retrieval error: {str(e)}")
            raise ProcessingError(f"Language pair performance statistics retrieval error: {str(e)}")
        
    def get_model_recommendations(self) -> Dict[str, List[str]]:
        """Gets model recommendations based on performance."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Average performance for each model
            c.execute('''
                SELECT 
                    model_type,
                    AVG(avg_relevance_score) as avg_relevance,
                    AVG(avg_response_time) as avg_response,
                    COUNT(*) as sample_size
                FROM performance_metrics
                GROUP BY model_type
                HAVING sample_size >= 10
            ''')
            
            results = c.fetchall()
            conn.close()
            
            recommendations = {
                'best_overall': [],
                'fastest': [],
                'most_accurate': []
            }
            
            if results:
                # Best overall performance
                best_overall = max(results, key=lambda x: (x[1] * 0.7 + (1/x[2]) * 0.3))
                recommendations['best_overall'].append(best_overall[0])
                
                # Fastest model
                fastest = min(results, key=lambda x: x[2])
                recommendations['fastest'].append(fastest[0])
                
                # Most accurate model
                most_accurate = max(results, key=lambda x: x[1])
                recommendations['most_accurate'].append(most_accurate[0])
                
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Model recommendations retrieval error: {str(e)}")
            raise ProcessingError(f"Model recommendations retrieval error: {str(e)}")
        
    def export_feedback(self, output_path: str):
        """Exports feedback data as JSON."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Retrieve feedback
            c.execute('SELECT * FROM feedback')
            feedback_data = [dict(row) for row in c.fetchall()]
            
            # Retrieve performance metrics
            c.execute('SELECT * FROM performance_metrics')
            metrics_data = [dict(row) for row in c.fetchall()]
            
            conn.close()
            
            # Combine data
            export_data = {
                'feedback': feedback_data,
                'performance_metrics': metrics_data
            }
            
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Data exported: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Data export error: {str(e)}")
            raise ProcessingError(f"Data export error: {str(e)}")
        
    def import_feedback(self, input_path: str):
        """Imports feedback data from a JSON file."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Import feedback
            for feedback in import_data.get('feedback', []):
                feedback_obj = Feedback(
                    question=feedback['question'],
                    answer=feedback['answer'],
                    relevance_score=feedback['relevance_score'],
                    question_language=feedback['question_language'],
                    document_language=feedback['document_language'],
                    model_type=feedback['model_type'],
                    timestamp=datetime.fromisoformat(feedback['timestamp']),
                    metadata=json.loads(feedback['metadata'])
                )
                self.add_feedback(feedback_obj.question, feedback_obj.answer, feedback_obj.relevance_score, feedback_obj.question_language, feedback_obj.document_language, feedback_obj.model_type, feedback_obj.metadata)
            
            # Import performance metrics
            for metrics in import_data.get('performance_metrics', []):
                metrics_obj = PerformanceMetrics(
                    model_type=metrics['model_type'],
                    avg_response_time=metrics['avg_response_time'],
                    avg_relevance_score=metrics['avg_relevance_score'],
                    language_pairs=json.loads(metrics['language_pairs']),
                    total_queries=metrics['total_queries'],
                    successful_queries=metrics['successful_queries'],
                    memory_usage=metrics['memory_usage'],
                    timestamp=datetime.fromisoformat(metrics['timestamp'])
                )
                self.add_performance_metrics(metrics_obj)
            
            self.logger.info(f"Data imported: {input_path}")
            
        except Exception as e:
            self.logger.error(f"Data import error: {str(e)}")
            raise ProcessingError(f"Data import error: {str(e)}")
        
    def cleanup_old_data(self, days: int = 30):
        """Cleans up old data."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # SQL query date format corrected
            # Delete old feedback
            c.execute('''
                DELETE FROM feedback
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            ''', (days,))
            
            # Delete old performance metrics
            c.execute('''
                DELETE FROM performance_metrics
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            ''', (days,))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"{days} days old data cleaned up")
            
        except Exception as e:
            self.logger.error(f"Data cleanup error: {str(e)}")
            raise ProcessingError(f"Data cleanup error: {str(e)}")
        
    def get_feedback_summary(self, days: int = 7) -> Dict[str, Any]:
        """Gets feedback summary."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Feedback statistics for the last n days
            c.execute('''
                SELECT 
                    COUNT(*) as total_feedback,
                    AVG(relevance_score) as avg_relevance,
                    COUNT(DISTINCT question_language) as unique_question_langs,
                    COUNT(DISTINCT document_language) as unique_doc_langs,
                    COUNT(DISTINCT model_type) as unique_models
                FROM feedback
                WHERE timestamp >= datetime('now', '-? days')
            ''', (days,))
            
            result = c.fetchone()
            
            if result:
                summary = {
                    'total_feedback': result[0],
                    'avg_relevance_score': result[1] or 0,
                    'unique_question_languages': result[2],
                    'unique_document_languages': result[3],
                    'unique_models': result[4]
                }
            else:
                summary = {
                    'total_feedback': 0,
                    'avg_relevance_score': 0,
                    'unique_question_languages': 0,
                    'unique_document_languages': 0,
                    'unique_models': 0
                }
            
            conn.close()
            return summary
            
        except Exception as e:
            self.logger.error(f"Feedback summary retrieval error: {str(e)}")
            raise ProcessingError(f"Feedback summary retrieval error: {str(e)}") 