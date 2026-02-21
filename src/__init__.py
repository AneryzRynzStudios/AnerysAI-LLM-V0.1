"""
AneryzAI LLM - Large Language Model with Multi-Language Support
"""

__version__ = "0.1.0"
__author__ = "AneryzAI Studios"

from src.config import ModelConfig, TrainingConfig, InferenceConfig, LanguageConfig
from src.model import TransformerModel
from src.tokenizer import BaseTokenizer
from src.training import Trainer, LanguageModelDataset, create_optimizer, create_scheduler
from src.inference import TextGenerator, GenerationType, BatchGenerator
from src.utils import (
    CheckpointManager,
    ConfigManager,
    MetricsTracker,
    LanguageDetector,
    print_model_info,
)

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "InferenceConfig",
    "LanguageConfig",
    "TransformerModel",
    "BaseTokenizer",
    "Trainer",
    "LanguageModelDataset",
    "create_optimizer",
    "create_scheduler",
    "TextGenerator",
    "GenerationType",
    "BatchGenerator",
    "CheckpointManager",
    "ConfigManager",
    "MetricsTracker",
    "LanguageDetector",
    "print_model_info",
]
