"""
Configuration management.
This module provides configuration management used throughout the application.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import os
import yaml
import logging
from pathlib import Path

from loguru import logger

@dataclass
class TextProcessingConfig:
    """Data class for text processing configuration."""
    # Cache settings
    cache_capacity: int = 100
    
    # Processing settings
    max_workers: int = 4
    chunk_size: int = 100
    embedding_size: int = 100
    
    # Memory settings
    gc_passes: int = 5
    cleanup_threshold: float = 15.0  # MB
    memory_increase_threshold: float = 20.0  # MB
    
    # Performance thresholds
    peak_memory_limit: float = 400.0  # MB
    net_memory_increase_limit: float = 100.0  # MB
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'cache_capacity': self.cache_capacity,
            'max_workers': self.max_workers,
            'chunk_size': self.chunk_size,
            'embedding_size': self.embedding_size,
            'gc_passes': self.gc_passes,
            'cleanup_threshold': self.cleanup_threshold,
            'memory_increase_threshold': self.memory_increase_threshold,
            'peak_memory_limit': self.peak_memory_limit,
            'net_memory_increase_limit': self.net_memory_increase_limit
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TextProcessingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

@dataclass
class Config:
    """Data class for application configuration."""
    app_name: str
    version: str
    environment: str
    resources: Dict[str, Any]
    models: Dict[str, Any]
    processing: TextProcessingConfig
    storage: Dict[str, Any]
    api: Dict[str, Any]

class ConfigManager:
    """Configuration manager."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initializes the configuration manager."""
        self.config_path = config_path
        self.config: Optional[Config] = None
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Loads the configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                # Convert processing dict to TextProcessingConfig
                if 'processing' in config_data:
                    config_data['processing'] = TextProcessingConfig.from_dict(config_data['processing'])
                return config_data
        except Exception as e:
            self.logger.error(f"Config loading error: {str(e)}")
            raise
    
    def load(self) -> None:
        """Loads and validates the configuration."""
        config_data = self._load_config()
        self.config = Config(**config_data)
        self.logger.info(f"Config loaded: {self.config.app_name} v{self.config.version}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Returns the configuration value."""
        if not self.config:
            self.load()
        return getattr(self.config, key, default)
    
    def update(self, key: str, value: Any) -> None:
        """Updates the configuration value."""
        if not self.config:
            self.load()
        setattr(self.config, key, value)
    
    def save(self) -> None:
        """Saves the configuration to file."""
        if not self.config:
            return
            
        try:
            config_data = {
                'app_name': self.config.app_name,
                'version': self.config.version,
                'environment': self.config.environment,
                'resources': self.config.resources,
                'models': self.config.models,
                'processing': self.config.processing.to_dict() if isinstance(self.config.processing, TextProcessingConfig) else self.config.processing,
                'storage': self.config.storage,
                'api': self.config.api
            }
            
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False)
                
            self.logger.info("Config saved successfully")
            
        except Exception as e:
            self.logger.error(f"Config saving error: {str(e)}")
            raise 

    @staticmethod
    def load_config(config_path: str = None) -> Dict[str, Any]:
        """Loads the configuration file."""
        # Implementation of load_config method
        pass
    
    def save_config(self, config_path: str = None) -> None:
        """Saves the configuration to file."""
        # Implementation of save_config method
        pass 