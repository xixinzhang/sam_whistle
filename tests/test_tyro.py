from dataclasses import dataclass
from typing import Union
import tyro

# Define the base configuration
@dataclass
class BasicConfig:
    project_name: str = "default_project"
    seed: int = 42

# Define configurations for different models
@dataclass
class ModelAConfig:
    learning_rate: float = 0.001
    layers: int = 4

@dataclass
class ModelBConfig:
    dropout: float = 0.5
    units: int = 128

@dataclass
class ModelCConfig:
    attention_heads: int = 8
    embedding_dim: int = 512

# Define a wrapper for model-specific configs
@dataclass
class Config:
    model_type: str  # Model type selector (e.g., "A", "B", "C")
    basic: BasicConfig 
    model: Union[ModelAConfig, ModelBConfig, ModelCConfig] = None

import sys

# Map model_type to specific dataclasses
model_configs = {
    "A": ModelAConfig,
    "B": ModelBConfig,
    "C": ModelCConfig,
}

def select_cfg(type):
    return model_configs[type]

# Parse command-line arguments
if __name__ == "__main__":
    # First parse the basic and model type config
    args = tyro.cli(select_cfg('A'))
    cfg2 = tyro.cli(ModelBConfig)
    print(args)
    print(cfg2)