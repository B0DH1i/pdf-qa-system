import psutil
import os
from typing import Dict, Optional, Callable, Any
import gc
import logging as py_logging
from functools import wraps
import time
import torch
import numpy as np
from contextlib import contextmanager

class MemoryTracker:
    """Memory tracking utility for monitoring and optimizing memory usage"""
    
    def __init__(self):
        """Initialize memory tracker"""
        self.process = psutil.Process(os.getpid())
        self.start_memory: Dict[str, float] = {}
        self.peak_memory: Dict[str, float] = {}
        self.logger = py_logging.getLogger(__name__)
        self.memory_stats: Dict[str, Dict[str, float]] = {}
        self.current_tracking: Optional[str] = None
    
    def get_current_memory(self) -> float:
        """Return current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def start_tracking(self, section_name: str) -> None:
        """Start tracking memory for a section"""
        self.current_tracking = section_name
        self.memory_stats[section_name] = {
            'start': self._get_memory_usage(),
            'peak': 0,
            'end': 0
        }
    
    def end_tracking(self, section_name: str) -> None:
        """End tracking memory for a section"""
        if section_name in self.memory_stats:
            current_usage = self._get_memory_usage()
            self.memory_stats[section_name]['end'] = current_usage
            self.memory_stats[section_name]['peak'] = max(
                self.memory_stats[section_name]['peak'],
                current_usage
            )
        self.current_tracking = None
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.get_current_memory()
    
    def update_peak(self, section_name: str) -> None:
        """Update peak memory for a section"""
        current = self.get_current_memory()
        if current > self.peak_memory.get(section_name, 0):
            self.peak_memory[section_name] = current
    
    def end_tracking(self, section_name: str) -> Dict[str, float]:
        """End tracking and return memory stats"""
        current = self.get_current_memory()
        stats = {
            'start': self.start_memory.get(section_name, 0),
            'end': current,
            'peak': self.peak_memory.get(section_name, current),
            'diff': current - self.start_memory.get(section_name, 0)
        }
        self.logger.info(f"Memory stats for {section_name}: {stats}")
        return stats
    
    @classmethod
    def track_memory(cls, name: str) -> Callable:
        """Decorator for tracking memory usage"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(instance, *args, **kwargs):
                instance.memory_tracker.start_tracking(name)
                try:
                    result = func(instance, *args, **kwargs)
                    return result
                finally:
                    instance.memory_tracker.end_tracking(name)
            return wrapper
        return decorator
    
    def force_cleanup(self) -> None:
        """Force garbage collection and memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def get_memory_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all tracked sections"""
        return {
            section: {
                'start': self.start_memory.get(section, 0),
                'peak': self.peak_memory.get(section, 0)
            }
            for section in self.start_memory.keys()
        }
    
    @contextmanager
    def track(self, name: str) -> Any:
        """Context manager for tracking memory usage"""
        self.start_tracking(name)
        try:
            yield
        finally:
            self.end_tracking(name)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get memory usage statistics"""
        return self.memory_stats.copy()
    
    def get_peak_memory(self, section_name: str) -> float:
        """Get peak memory usage for a section in MB."""
        return self.peak_memory.get(section_name, 0) 