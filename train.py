"""
Main training script for AneryzAI LLM
Demonstrates how to train the model on English and Indonesian data
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json

from src.config import ModelConfig, TrainingConfig, LanguageConfig
from src.model import TransformerModel
from src.tokenizer import BaseTokenizer
from src.training import (
    Trainer,
    LanguageModelDataset,
    create_optimizer,
    create_scheduler,
)
from src.utils import (
    CheckpointManager,
    ConfigManager,
    print_model_info,
    set_seed,
    get_device,
)


def create_sample_data():
    """
    Create sample training data dalam Inggris dan Indonesia
    Untuk demo purposes
    """
    
    english_texts = [
        "The quick brown fox jumps over the lazy dog. This is a sample English text for training.",
        "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms.",
        "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.",
        "Deep learning has revolutionized many fields including computer vision and natural language processing.",
        "Transformer models have become the state-of-the-art architecture for various NLP tasks.",
        "Transfer learning allows us to leverage pre-trained models for new tasks with limited data.",
        "Attention mechanisms enable models to focus on relevant parts of the input sequence.",
    ]
    
    indonesian_texts = [
        "Model pembelajaran mesin telah mengubah cara kita memproses dan menganalisis data.",
        "Pemrosesan bahasa alami adalah bidang yang menggabungkan linguistik dan ilmu komputer.",
        "Jaringan neural yang dalam telah memberikan hasil yang luar biasa dalam pengenalan citra.",
        "Mekanisme perhatian memungkinkan model untuk fokus pada bagian yang relevan dari input.",
        "Transfer learning memungkinkan kita memanfaatkan model yang telah dilatih sebelumnya.",
        "Transformasi data adalah langkah penting dalam persiapan data untuk pembelajaran mesin.",
        "Algoritma optimasi memainkan peran krusial dalam melatih model pembelajaran mendalam.",
    ]
    
    return english_texts, indonesian_texts


def prepare_training_data(texts, tokenizer, language):
    """
    Prepare training data by tokenizing texts
    """
    tokenized_texts = []
    for text in texts:
        # Add special tokens untuk setiap document
        tokens = tokenizer.encode(text, language=language, add_special_tokens=True)
        if len(tokens) > 0:
            tokenized_texts.append(tokens)
    return tokenized_texts


def main(args):
    """
    Main training loop
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = args.device if args.device else get_device()
    print(f"Using device: {device}")
    
    # Create model config
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        max_sequence_length=args.max_length,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    
    # Create training config  
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_epochs=args.num_epochs,
        device=device,
    )
    
    print(f"\n{'='*60}")
    print("AneryzAI LLM Training Configuration")
    print(f"{'='*60}")
    print(f"Model Config:")
    print(f"  Vocab Size: {model_config.vocab_size}")
    print(f"  Max Sequence Length: {model_config.max_sequence_length}")
    print(f"  Embedding Dim: {model_config.embedding_dim}")
    print(f"  Num Heads: {model_config.num_heads}")
    print(f"  Num Layers: {model_config.num_layers}")
    print(f"\nTraining Config:")
    print(f"  Batch Size: {training_config.batch_size}")
    print(f"  Learning Rate: {training_config.learning_rate}")
    print(f"  Num Epochs: {training_config.num_epochs}")
    print(f"  Max Steps: {training_config.max_steps}")
    print(f"{'='*60}\n")
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = BaseTokenizer(vocab_size=model_config.vocab_size)
    
    # Get sample data
    english_texts, indonesian_texts = create_sample_data()
    
    # Train tokenizer on both languages
    print("\nTraining tokenizer on English texts...")
    tokenizer.train(english_texts, language="en")
    
    print("Training tokenizer on Indonesian texts...")
    tokenizer.train(indonesian_texts, language="id")
    
    # Save tokenizer
    tokenizer_path = Path("checkpoints/tokenizer.json")
    tokenizer_path.parent.mkdir(exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    
    # Prepare training data
    print("\nPreparing training data...")
    en_tokens = prepare_training_data(english_texts, tokenizer, "en")
    id_tokens = prepare_training_data(indonesian_texts, tokenizer, "id")
    
    # Combine data
    all_tokens = en_tokens + id_tokens
    
    if len(all_tokens) == 0:
        print("Error: No training data available!")
        return
    
    # Create dataset
    dataset = LanguageModelDataset(
        token_ids_list=all_tokens,
        max_length=model_config.max_sequence_length,
    )
    
    print(f"Dataset size: {len(dataset)} examples")
    
    # Create dataloader
    train_dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = TransformerModel(model_config).to(device)
    print_model_info(model)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    
    scheduler = create_scheduler(
        optimizer,
        warmup_steps=training_config.warmup_steps,
        max_steps=training_config.max_steps,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=training_config,
    )
    
    # Save configuration
    config_dict = {
        "model_config": {
            "vocab_size": model_config.vocab_size,
            "max_sequence_length": model_config.max_sequence_length,
            "embedding_dim": model_config.embedding_dim,
            "num_heads": model_config.num_heads,
            "num_layers": model_config.num_layers,
            "ffn_dim": model_config.ffn_dim,
            "dropout_rate": model_config.dropout_rate,
        },
        "training_config": {
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "num_epochs": training_config.num_epochs,
            "max_steps": training_config.max_steps,
        },
        "supported_languages": ["en", "id"],
    }
    
    ConfigManager.save_config(
        config_dict,
        Path("checkpoints/config.json")
    )
    
    # Train model
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    trainer.train(
        num_epochs=training_config.num_epochs,
        save_path="checkpoints/best_model.pt"
    )
    
    # Save final model
    final_model_path = "checkpoints/final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    print(f"\n{'='*60}")
    print("Training Completed Successfully!")
    print(f"Checkpoints saved in: checkpoints/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AnerysAI LLM")
    
    # Model arguments
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="Vocabulary size")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--embedding_dim", type=int, default=768,
                        help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of transformer layers")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Maximum training steps")
    
    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, cpu, mps)")
    
    args = parser.parse_args()
    main(args)
