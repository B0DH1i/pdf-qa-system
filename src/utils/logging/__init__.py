"""
Logging configuration and utilities.
"""

import logging as py_logging
import os
from pathlib import Path

def setup_logging(
    log_file: str = "app.log",
    log_dir: str = "logs",
    level: int = py_logging.INFO,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """Setup logging configuration."""
    log_path = Path(log_dir) / log_file
    os.makedirs(log_dir, exist_ok=True)
    
    py_logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            py_logging.FileHandler(log_path),
            py_logging.StreamHandler()
        ]
    )

# Create default logger
setup_logging(
    log_file='text_pipeline.log',
    log_dir='logs'
) 