#!/usr/bin/env python3
"""
AnerysAI LLM - Complete Usage Guide
Quick reference for all commands and features
"""

COMMANDS = {
    "Installation": [
        ("Install dependencies", "pip install -r requirements.txt"),
    ],
    
    "Training": [
        ("Quick train (2 min, CPU)", 
         "python train.py --embedding_dim 256 --num_layers 2 --batch_size 4 --num_epochs 1 --device cpu"),
        
        ("Standard train (10 min, GPU)",
         "python train.py --num_epochs 3 --batch_size 32 --device cuda"),
        
        ("Medium train (5 min, CPU)",
         "python train.py --embedding_dim 512 --num_layers 4 --batch_size 8 --num_epochs 1 --device cpu"),
        
        ("Custom training parameters",
         "python train.py --vocab_size 32000 --embedding_dim 768 --num_heads 12 --num_layers 12 --learning_rate 5e-5"),
    ],
    
    "Inference - Demo Mode": [
        ("Show examples", "python inference.py --device cpu"),
        ("With batch processing", "python inference.py --device cpu --batch"),
    ],
    
    "Inference - Interactive Mode": [
        ("Interactive prompt input", "python inference.py --device cpu --interactive"),
        ("Commands: /en, /id, /config, quit", ""),
    ],
    
    "Examples": [
        ("View all 10 examples", "python examples.py"),
        ("Then select example number or 'all'", ""),
    ],
    
    "Programmatic Usage": [
        ("Create model", """
from src import TransformerModel, ModelConfig
config = ModelConfig()
model = TransformerModel(config).to('cuda')
        """),
        
        ("Load trained model", """
import torch
state_dict = torch.load('checkpoints/best_model.pt')
model = TransformerModel(config)
model.load_state_dict(state_dict)
        """),
        
        ("Use tokenizer", """
from src import BaseTokenizer
tokenizer = BaseTokenizer()
tokenizer.load('checkpoints/tokenizer.json')
tokens = tokenizer.encode('Hello world', language='en')
text = tokenizer.decode(tokens)
        """),
        
        ("Generate text", """
from src.inference import TextGenerator, GenerationType
generator = TextGenerator(model, tokenizer, device='cuda')
text, score = generator.generate(
    prompt='The future of AI',
    language='en',
    max_new_tokens=100,
    generation_type=GenerationType.TOP_P,
    top_p=0.95
)
        """),
    ],
}

MODELS = {
    "Available Configs": {
        "Small": {
            "embedding_dim": 256,
            "num_heads": 4,
            "num_layers": 2,
            "size": "~10M params",
            "speed": "Very fast",
        },
        "Medium": {
            "embedding_dim": 512,
            "num_heads": 8,
            "num_layers": 4,
            "size": "~40M params",
            "speed": "Fast",
        },
        "Large": {
            "embedding_dim": 768,
            "num_heads": 12,
            "num_layers": 12,
            "size": "~110M params",
            "speed": "Moderate",
        },
        "ExtraLarge": {
            "embedding_dim": 1024,
            "num_heads": 16,
            "num_layers": 24,
            "size": "~300M+ params",
            "speed": "Slow",
        },
    }
}

GENERATION_STRATEGIES = {
    "Greedy": {
        "description": "Most deterministic",
        "quality": "Good",
        "speed": "Fastest",
        "use_case": "When you want consistent output",
    },
    "Top-K": {
        "description": "Sample from top K tokens",
        "quality": "Good",
        "speed": "Fast",
        "use_case": "Balanced quality and diversity",
    },
    "Top-P (Nucleus)": {
        "description": "Sample from tokens with cumsum prob >= P",
        "quality": "Very Good",
        "speed": "Fast",
        "use_case": "Most popular, great diversity",
    },
    "Beam Search": {
        "description": "Track multiple best hypotheses",
        "quality": "Excellent",
        "speed": "Slower",
        "use_case": "When quality is critical",
    },
}

FILES_DESCRIPTION = {
    "Documentation": {
        "README.md": "Full documentation with usage guide (600+ lines)",
        "ARCHITECTURE.md": "Architecture diagrams and explanations (400+ lines)",
        "IMPLEMENTATION_SUMMARY.md": "Summary of implementation with logic",
        "PROJECT_STRUCTURE.md": "File structure and overview",
        "QUICK_START.sh": "Quick start tutorial",
    },
    
    "Scripts": {
        "train.py": "Training script (300+ lines)",
        "inference.py": "Inference script with modes (400+ lines)",
        "examples.py": "10 usage examples (500+ lines)",
    },
    
    "Core Library": {
        "src/config.py": "Configuration classes (200+ lines)",
        "src/utils.py": "Utilities, checkpoint manager (300+ lines)",
        "src/tokenizer/base_tokenizer.py": "BPE Tokenizer (400+ lines)",
        "src/model/transformer.py": "Transformer Model (700+ lines)",
        "src/training/trainer.py": "Training pipeline (500+ lines)",
        "src/inference/text_generator.py": "Generation engine (600+ lines)",
    },
}

TROUBLESHOOTING = {
    "Model generates gibberish": [
        "1. Check tokenizer is trained properly",
        "2. Train longer (increase num_epochs)",
        "3. Use lower temperature (< 0.7)",
        "4. Provide higher quality training data",
    ],
    
    "Out of Memory (OOM)": [
        "1. Reduce batch_size (--batch_size 4)",
        "2. Reduce model size (--embedding_dim 256)",
        "3. Use gradient accumulation",
        "4. Enable mixed precision (FP16)",
    ],
    
    "Training is too slow": [
        "1. Use GPU instead of CPU (--device cuda)",
        "2. Reduce model size",
        "3. Increase batch_size if possible",
        "4. Use smaller max_sequence_length",
    ],
    
    "Tokenizer not loading": [
        "1. Check tokenizer.json exists in checkpoints/",
        "2. Train tokenizer first: python train.py",
        "3. Verify file is not corrupted",
    ],
    
    "Language detection wrong": [
        "1. Ensure text has enough language markers",
        "2. Use /en or /id in interactive mode to force",
        "3. Check language detection logic in utils.py",
    ],
}

TIPS_AND_TRICKS = {
    "Training": [
        "Start with small model to test pipeline",
        "Use gradient accumulation for larger effective batch size",
        "Monitor loss with tensorboard or wandb",
        "Save checkpoints frequently",
        "Use validation early stopping",
        "Adjust learning rate based on loss curve",
    ],
    
    "Inference": [
        "Use KV cache for 2-3x speed improvement",
        "Batch process multiple prompts together",
        "Top-P sampling usually gives best results",
        "Lower temperature for more focused output",
        "Use beam search for critical applications",
        "Cache model on GPU for faster inference",
    ],
    
    "Language Support": [
        "Add language by extending LanguageConfig",
        "Add preprocessing rules in BaseTokenizer",
        "Train tokenizer on language-specific texts",
        "Add special tokens for each language",
    ],
    
    "Performance": [
        "Use smaller model for faster inference",
        "Quantize model to INT8/FP16 for speed",
        "Use distillation to create smaller model",
        "Enable TorchScript for production serving",
        "Use ONNX export for cross-platform",
    ],
}

def print_section(title, content, indent=0):
    """Print formatted section"""
    spaces = " " * indent
    print(f"\n{spaces}{'='*60}")
    print(f"{spaces}{title}")
    print(f"{spaces}{'='*60}")
    
    if isinstance(content, dict):
        for key, value in content.items():
            print(f"\n{spaces}{key}:")
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, tuple):
                        cmd, desc = item
                        print(f"{spaces}  {cmd}")
                        if desc:
                            print(f"{spaces}    $ {desc}")
                    else:
                        print(f"{spaces}  • {item}")
            else:
                print(f"{spaces}  {value}")
    elif isinstance(content, list):
        for item in content:
            print(f"{spaces}  • {item}")
    else:
        print(f"{spaces}{content}")

def main():
    """Show complete usage guide"""
    print("\n" + "="*70)
    print(" " * 15 + "AneryzAI LLM - COMPLETE USAGE GUIDE")
    print("="*70)
    
    # Show all sections
    print_section("QUICK COMMANDS", COMMANDS)
    
    print_section("MODEL CONFIGURATIONS", MODELS)
    
    print_section("TEXT GENERATION STRATEGIES", GENERATION_STRATEGIES)
    
    print_section("PROJECT FILES", FILES_DESCRIPTION)
    
    print_section("GENERATION STRATEGIES COMPARISON", GENERATION_STRATEGIES)
    
    print_section("TROUBLESHOOTING", TROUBLESHOOTING)
    
    print_section("TIPS & TRICKS", TIPS_AND_TRICKS)
    
    # Final summary
    print("\n" + "="*70)
    print("QUICK START SUMMARY")
    print("="*70)
    
    print("""
    1. INSTALL
       $ pip install -r requirements.txt
    
    2. TRAIN
       $ python train.py --device cpu --num_epochs 1
    
    3. TEST (DEMO)
       $ python inference.py --device cpu
    
    4. TEST (INTERACTIVE)
       $ python inference.py --device cpu --interactive
    
    5. LEARN
       $ python examples.py
    
    For more info, see:
       - README.md           (Full documentation)
       - ARCHITECTURE.md     (System design)
       - PROJECT_STRUCTURE.md (File overview)
    """)
    
    print("="*70)

if __name__ == "__main__":
    main()
