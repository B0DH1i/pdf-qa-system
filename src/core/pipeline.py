"""
Central processing system.
This module contains the main pipeline that coordinates all processing workflows.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import asyncio
import logging
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
# import dask.distributed as dd  # We're removing Dask usage

from loguru import logger

from .memory import MemoryManager
from .exceptions import ProcessingError

@dataclass
class ProcessingContext:
    """Data class for processing context."""
    input_data: Any
    metadata: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[Exception] = None

class Pipeline:
    """Main processing pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initializes the pipeline."""
        self.config = config.get("pipeline", {})
        self.stages: List[Callable] = []
        self.context: Optional[ProcessingContext] = None
        self.logger = logging.getLogger(__name__)
        
        # Memory management
        self.memory_manager = MemoryManager(
            threshold_mb=self.config.get("memory_threshold", 1000)
        )
        
        # Parallel processing - using only ThreadPoolExecutor
        self.n_workers = self.config.get("n_workers", mp.cpu_count())
        self.backend = "default"  # Dask removed, using only ThreadPoolExecutor
        
        # Initialize backend
        self.logger.info(f"ThreadPoolExecutor backend started: {self.n_workers} workers")
        self.pool = ThreadPoolExecutor(max_workers=self.n_workers)
    
    def add_stage(self, stage: Callable) -> None:
        """Adds a new stage to the pipeline."""
        self.stages.append(stage)
    
    def setup(self, input_data: Any, metadata: Dict[str, Any]) -> None:
        """Prepares the pipeline for a new process."""
        self.context = ProcessingContext(
            input_data=input_data,
            metadata=metadata,
            start_time=datetime.now()
        )
    
    async def process(self) -> Any:
        """Runs the pipeline and returns the result."""
        if not self.context:
            raise ValueError("Pipeline not set up")
            
        try:
            result = self.context.input_data
            for stage in self.stages:
                # Start memory tracking
                self.memory_manager.start_tracking(stage.__name__)
                
                start_time = datetime.now()
                
                # Parallel processing required
                if self.config.get("parallel", False):
                    result = await self._process_parallel(stage, result)
                else:
                    result = await stage(result, self.context.metadata)
                
                duration = (datetime.now() - start_time).total_seconds()
                
                # End memory tracking
                stats = self.memory_manager.end_tracking(stage.__name__)
                
                self.logger.info(
                    f"Stage {stage.__name__} completed: "
                    f"{duration:.2f}s, Memory: {stats.memory_diff_mb:.2f}MB"
                )
            
            self.context.result = result
            self.context.end_time = datetime.now()
            return result
            
        except Exception as e:
            self.context.error = e
            self.context.end_time = datetime.now()
            self.logger.error(f"Pipeline error: {str(e)}")
            raise ProcessingError(f"Pipeline processing error: {str(e)}")
    
    async def _process_parallel(
        self,
        stage: Callable,
        data: Any,
        chunk_size: int = 100
    ) -> Any:
        """Processes data in parallel."""
        try:
            # Split data into chunks
            if isinstance(data, list):
                chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            else:
                chunks = [data]
            
            # Parallel processing with ThreadPoolExecutor
            results = list(
                self.pool.map(
                    lambda x: stage(x, self.context.metadata),
                    chunks
                )
            )
            
            # Combine results
            if isinstance(data, list):
                return [item for sublist in results for item in sublist]
            return results[0]
            
        except Exception as e:
            self.logger.error(f"Parallel processing error: {str(e)}")
            raise ProcessingError(f"Parallel processing error: {str(e)}")
    
    def close(self) -> None:
        """Cleans up resources."""
        try:
            self.pool.shutdown()
        except Exception as e:
            self.logger.error(f"Pipeline shutdown error: {str(e)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 