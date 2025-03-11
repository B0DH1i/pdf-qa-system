"""
Model configuration module.
Contains model parameters and settings.
"""

from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    learning_rate: float = 1e-5
    max_length: int = 512
    num_workers: int = 4
    context_size: int = 3
    similarity_threshold: float = 0.7
    
    def to_dict(self) -> dict:
        """Returns the configuration as a dictionary."""
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        } 