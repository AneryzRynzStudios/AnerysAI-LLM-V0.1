"""
Configuration file for AneryzAI LLM Model
Supports English and Indonesian languages
"""

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TokenizerConfig:
    """Configuration for Tokenizer"""
    vocab_size: int = 32000
    max_length: int = 2048
    special_tokens: List[str] = None
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = [
                "<pad>", "<unk>", "<bos>", "<eos>",
                "<en>", "<id>", "<class>"
            ]

@dataclass
class ModelConfig:
    """Configuration for Transformer Model"""
    vocab_size: int = 32000
    max_sequence_length: int = 2048
    embedding_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    ffn_dim: int = 3072
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True

@dataclass
class TrainingConfig:
    """Configuration for Training"""
    batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    warmup_steps: int = 10000
    max_steps: int = 100000
    num_epochs: int = 3
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 500
    log_steps: int = 100
    device: str = "cuda"
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"

@dataclass
class InferenceConfig:
    """Configuration for Inference"""
    max_length: int = 512
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95
    num_beams: int = 1  # 1 for greedy, > 1 for beam search
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0

class LanguageConfig:
    """Configuration for multi-language support"""
    LANGUAGES = {
        "en": {
            "name": "English",
            "code": "<en>",
            "special_chars": ".,!?;:\"-'()",
            "stopwords": [
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                "of", "with", "by", "from", "as", "is", "was", "are", "be", "been"
            ]
        },
        "id": {
            "name": "Indonesian",
            "code": "<id>",
            "special_chars": ".,!?;:\"-'()",
            "stopwords": [
                "dan", "atau", "yang", "di", "ke", "dari", "untuk", "pada", "adalah",
                "dengan", "oleh", "sebagai", "dalam", "ini", "itu", "ini", "tersebut"
            ]
        }
    }
    
    @classmethod
    def get_language_token(cls, lang_code: str) -> str:
        """Get language token for specific language"""
        if lang_code in cls.LANGUAGES:
            return cls.LANGUAGES[lang_code]["code"]
        raise ValueError(f"Unsupported language: {lang_code}")
    
    @classmethod
    def is_valid_language(cls, lang_code: str) -> bool:
        """Check if language is supported"""
        return lang_code in cls.LANGUAGES

# Global configurations
TOKENIZER_CONFIG = TokenizerConfig()
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
INFERENCE_CONFIG = InferenceConfig()
