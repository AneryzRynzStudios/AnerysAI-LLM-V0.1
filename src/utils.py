"""
Utility functions for AneryzAI LLM
"""

import torch
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        step: int,
        config: Dict[str, Any],
        metrics: Dict[str, float] = None,
        is_best: bool = False,
    ):
        """Save checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}.pt"
        
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "config": config,
            "metrics": metrics or {},
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return checkpoint
    
    def get_latest_checkpoint(self):
        """Get latest checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint-*.pt"))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: int(p.stem.split("-")[1]))
            return str(latest)
        return None


class ConfigManager:
    """Manage configurations"""
    
    @staticmethod
    def save_config(config_dict: Dict[str, Any], path: str):
        """Save config to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"Config saved: {path}")
    
    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        """Load config from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config


class MetricsTracker:
    """Track training metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_average(self, key: str) -> float:
        """Get average of metric"""
        if key in self.metrics:
            return np.mean(self.metrics[key])
        return 0.0
    
    def get_latest(self, key: str):
        """Get latest metric value"""
        if key in self.metrics and len(self.metrics[key]) > 0:
            return self.metrics[key][-1]
        return None
    
    def reset(self):
        """Reset metrics"""
        self.metrics = {}


def print_model_info(model):
    """Print model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*50}")
    print("Model Information")
    print(f"{'='*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024**3):.2f} GB (FP32)")
    print(f"{'='*50}\n")


def count_parameters(model):
    """Count total number of parameters"""
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model):
    """Get model size in MB"""
    return count_parameters(model) * 4 / (1024 ** 2)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Get available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class LanguageDetector:
    """
    Simple language detection untuk prompts
    Mendeteksi Inggris atau Indonesia berdasarkan character patterns
    """
    
    INDONESIAN_INDICATORS = {
        'akhiran': ['kan', 'an', 'i', 'nya', 'lah', 'pun'],
        'kata_umum': ['dan', 'atau', 'yang', 'di', 'ke', 'dari', 'untuk', 'pada', 'adalah'],
        'karakter': ['ñ'],  # Rare in English
    }
    
    @classmethod
    def detect_language(cls, text: str) -> str:
        """
        Detect language of text
        Returns: "en" or "id"
        """
        text_lower = text.lower()
        
        # Count Indonesian indicators
        indo_score = 0
        
        # Check for Indonesian words
        for word in cls.INDONESIAN_INDICATORS['kata_umum']:
            if word in text_lower:
                indo_score += 2
        
        # Check for Indonesian suffixes
        for suffix in cls.INDONESIAN_INDICATORS['akhiran']:
            if text_lower.endswith(suffix):
                indo_score += 1
        
        # Check for Indonesian characters
        for char in cls.INDONESIAN_INDICATORS['karakter']:
            indo_score += text_lower.count(char) * 5
        
        # Simple heuristic: if more than average of 1 Indonesian indicator per 10 words
        words = text_lower.split()
        threshold = len(words) / 10
        
        if indo_score > threshold:
            return "id"
        else:
            return "en"


def prepare_text_for_tokenization(text: str, language: str = "en") -> str:
    """
    Prepare text for tokenization
    Normalize dan clean text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    if language == "id":
        # Indonesian specific preparations
        # Normalize common abbreviations
        replacements = {
            'yg': 'yang',
            'u': 'anda',
            'd': 'di',
            'tp': 'tapi',
            'bgm': 'bagaimana',
        }
        for short, long in replacements.items():
            text = text.replace(f' {short} ', f' {long} ')
    
    return text
