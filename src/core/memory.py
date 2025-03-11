"""
Memory management and parallel processing module.
"""

import psutil
import time
import logging
from typing import Dict, Optional, List, Any, Callable
from threading import Lock
import gc
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import os
import numpy as np
import torch

from loguru import logger

from .exceptions import ResourceError, ProcessingError

@dataclass
class MemoryConfig:
    """Memory management configuration."""
    threshold_mb: int = 1000
    cleanup_threshold_mb: int = 800
    max_workers: Optional[int] = None
    parallel_backend: str = "default"
    tracking_interval: float = 1.0  # seconds

class MemoryManager:
    """Class that monitors and manages memory usage."""
    
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        """
        Initialize memory manager.
        
        Args:
            config: Memory management configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Memory management
        self.threshold = config.threshold_mb * 1024 * 1024  # Convert MB to bytes
        self.cleanup_threshold = config.cleanup_threshold_mb * 1024 * 1024
        self._lock = Lock()
        self._tracking = {}
        self._start_times = {}
        self._last_cleanup = time.time()
        
        # Parallel processing
        self.n_workers = config.max_workers or mp.cpu_count()
        self.backend = "default"
        self._init_parallel_backend()
        
    def _init_parallel_backend(self):
        """Initialize parallel processing backend."""
        self.logger.info(f"ThreadPoolExecutor backend started: {self.n_workers} workers")
        self.pool = ThreadPoolExecutor(max_workers=self.n_workers)
    
    def start_tracking(self, operation_name: str) -> None:
        """
        Start memory tracking for a specific operation
        
        Args:
            operation_name: Name of the operation to track
        """
        try:
            current_memory = self.get_memory_usage()
            current_time = datetime.now()
            
            self._tracking[operation_name] = {
                "start_memory": current_memory,
                "start_time": current_time,
                "end_memory": None,
                "end_time": None,
                "memory_diff_mb": 0,
                "duration_sec": 0
            }
            
            self.logger.info(f"Memory tracking started: {operation_name}")
            
        except Exception as e:
            self.logger.error(f"Memory tracking start error: {str(e)}")
            raise ProcessingError(f"Memory tracking start error: {str(e)}")
    
    def add_checkpoint(self, operation_name: str, checkpoint_name: str):
        """
        Add checkpoint for the tracked operation.
        
        Args:
            operation_name: Name of the tracked operation
            checkpoint_name: Checkpoint name
        """
        with self._lock:
            if operation_name not in self._tracking:
                self.logger.warning(f"No tracking record found for {operation_name}")
                return
                
            process = psutil.Process()
            current_memory = process.memory_info().rss
            current_time = time.time()
            
            self._tracking[operation_name]["checkpoints"].append({
                "name": checkpoint_name,
                "memory": current_memory,
                "time": current_time
            })
    
    def end_tracking(self, operation_name: str) -> Dict[str, Any]:
        """
        End memory tracking for the operation and return statistics.
        
        Args:
            operation_name: Name of the tracked operation
            
        Returns:
            Dict: Memory usage statistics
        """
        with self._lock:
            if operation_name not in self._tracking:
                self.logger.warning(f"No tracking record found for {operation_name}")
                return {}
                
            process = psutil.Process()
            tracking_data = self._tracking[operation_name]
            
            end_memory = process.memory_info().rss
            start_memory = tracking_data["start_memory"]
            duration = time.time() - tracking_data["start_time"]
            
            memory_diff = end_memory - start_memory
            
            # Warn if memory usage exceeds threshold
            if end_memory > self.threshold:
                self.logger.warning(
                    f"{operation_name} exceeded memory threshold: "
                    f"{end_memory / (1024*1024):.2f}MB / {self.threshold / (1024*1024)}MB"
                )
                self._cleanup()
            
            # Prepare statistics
            stats = {
                "operation": operation_name,
                "start_memory_mb": start_memory / (1024 * 1024),
                "end_memory_mb": end_memory / (1024 * 1024),
                "memory_diff_mb": memory_diff / (1024 * 1024),
                "duration_sec": duration,
                "checkpoints": [
                    {
                        "name": cp["name"],
                        "memory_mb": cp["memory"] / (1024 * 1024),
                        "time_sec": cp["time"] - tracking_data["start_time"]
                    }
                    for cp in tracking_data["checkpoints"]
                ]
            }
            
            # Clean tracking data
            del self._tracking[operation_name]
            
            return stats
    
    def _cleanup(self):
        """Perform memory cleanup operations."""
        current_time = time.time()
        
        # Check minimum cleanup interval
        if current_time - self._last_cleanup < self.config.tracking_interval:
            return
            
        self._last_cleanup = current_time
        
        # Force run garbage collector
        gc.collect()
        
        # Check memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        current_memory = memory_info.rss
        
        self.logger.info(
            f"Memory usage after cleanup: {current_memory / (1024*1024):.2f}MB"
        )
        
        # If memory is still high, perform additional cleanup
        if current_memory > self.cleanup_threshold:
            self._force_cleanup()
    
    def _force_cleanup(self):
        """Force memory cleanup operations."""
        # TODO: More aggressive memory cleanup strategies could be added
        pass
    
    def optimize_memory(self, threshold_mb: float = 500.0) -> Dict[str, Any]:
        """
        Optimize memory usage
        
        Args:
            threshold_mb: Memory threshold for cleanup (MB)
            
        Returns:
            Dict: Optimization results
        """
        try:
            start_memory = self.get_memory_usage()
            
            # Check if memory threshold is exceeded
            if start_memory < threshold_mb:
                self.logger.debug(f"Memory optimization not needed: {start_memory:.2f}MB < {threshold_mb}MB")
                return {
                    "optimized": False,
                    "memory_before": start_memory,
                    "memory_after": start_memory,
                    "memory_diff": 0
                }
                
            # Run garbage collection
            gc.collect()
            
            # Clear PyTorch cache
            if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Measure new memory usage
            end_memory = self.get_memory_usage()
            memory_diff = start_memory - end_memory
            
            self.logger.info(f"Memory optimization: {start_memory:.2f}MB -> {end_memory:.2f}MB (gain: {memory_diff:.2f}MB)")
            
            return {
                "optimized": True,
                "memory_before": start_memory,
                "memory_after": end_memory,
                "memory_diff": memory_diff
            }
            
        except Exception as e:
            self.logger.error(f"Memory optimization error: {str(e)}")
            return {
                "optimized": False,
                "error": str(e)
            }
    
    def get_memory_usage(self) -> float:
        """
        Return current memory usage in MB.
        
        Returns:
            float: Memory usage (MB)
        """
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def check_memory_threshold(self) -> bool:
        """
        Check if memory usage is below the threshold value.
        
        Returns:
            bool: True if memory usage is below the threshold
        """
        process = psutil.Process()
        current_usage = process.memory_info().rss
        return current_usage < self.threshold
    
    def parallel_map(self, 
                    func: Callable, 
                    items: List[Any], 
                    **kwargs) -> List[Any]:
        """
        Apply a function in parallel.
        
        Args:
            func: Function to apply
            items: List of items to process
            **kwargs: Additional arguments to pass to the function
            
        Returns:
            List[Any]: List of processed results
        """
        if not items:
            return []
            
        return list(
            self.pool.map(
                lambda x: func(x, **kwargs),
                items
            )
        )
    
    def close(self):
        """Clean up resources."""
        try:
            self.pool.shutdown()
        except Exception as e:
            self.logger.error(f"Resource cleanup error: {str(e)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def stop_tracking(self, operation_name: str) -> Dict[str, Any]:
        """
        Stop memory tracking for a specific operation
        
        Args:
            operation_name: Name of the operation to stop tracking
            
        Returns:
            Dict: Tracking statistics
            
        Raises:
            ProcessingError: If tracking cannot be stopped
        """
        try:
            if operation_name in self._tracking:
                end_memory = self.get_memory_usage()
                end_time = datetime.now()
                self._tracking[operation_name]["end_memory"] = end_memory
                self._tracking[operation_name]["end_time"] = end_time
                
                # Calculate statistics
                start_time = self._tracking[operation_name]["start_time"]
                start_memory = self._tracking[operation_name]["start_memory"]
                
                # Will work because datetime.now() is a datetime object
                duration_sec = (end_time - start_time).total_seconds()
                memory_diff_mb = end_memory - start_memory
                
                self._tracking[operation_name].update({
                    "memory_diff_mb": memory_diff_mb,
                    "duration_sec": duration_sec
                })
                
                self.logger.info(f"Memory tracking stopped: {operation_name}")
                self.logger.debug(f"Tracking statistics: {self._tracking[operation_name]}")
                
                # Return statistics
                return {
                    "operation": operation_name,
                    "memory_diff_mb": memory_diff_mb,
                    "duration_sec": duration_sec,
                    "start_memory_mb": start_memory,
                    "end_memory_mb": end_memory
                }
            else:
                self.logger.warning(f"Tracking not found: {operation_name}")
                return {
                    "operation": operation_name,
                    "memory_diff_mb": 0,
                    "duration_sec": 0,
                    "start_memory_mb": 0,
                    "end_memory_mb": 0
                }
                
        except Exception as e:
            self.logger.error(f"Memory tracking stop error: {str(e)}")
            raise ProcessingError(f"Memory tracking stop error: {str(e)}") 