"""
Example usage of AneryzAI LLM
Demonstrating berbagai features dan use cases
"""

import torch
from pathlib import Path
import sys

# Add source to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ModelConfig, LanguageConfig
from src.model import TransformerModel
from src.tokenizer import BaseTokenizer
from src.inference import TextGenerator, GenerationType, BatchGenerator
from src.utils import print_model_info, LanguageDetector, get_device


def example_1_basic_generation():
    """
    Example 1: Basic text generation dengan model yang sudah dilatih
    """
    print("\n" + "="*70)
    print("Example 1: Basic Text Generation")
    print("="*70)
    
    device = get_device()
    print(f"Device: {device}\n")
    
    # Initialize model config
    config = ModelConfig(
        vocab_size=32000,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
    )
    
    # Create model
    model = TransformerModel(config).to(device)
    print("Model architecture:")
    print_model_info(model)
    
    # Initialize tokenizer
    tokenizer = BaseTokenizer(vocab_size=config.vocab_size)
    
    # Train tokenizer on sample data
    print("Training tokenizer...")
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming technology",
        "Kecerdasan buatan adalah masa depan",
        "Pembelajaran mesin memungkinkan inovasi",
    ]
    
    # Train for both languages
    en_texts = [t for t in sample_texts if len(t.split()) < 7]
    id_texts = [t for t in sample_texts if len(t.split()) >= 7]
    tokenizer.train(en_texts + id_texts)
    
    # Create generator
    generator = TextGenerator(model, tokenizer, device=device)
    
    # Generate example
    print("\n" + "-"*70)
    print("Generating text...")
    
    prompts = [
        ("The future of artificial intelligence", "en"),
        ("Teknologi masa depan adalah", "id"),
    ]
    
    for prompt, lang in prompts:
        print(f"\nPrompt: '{prompt}' (Language: {lang})")
        try:
            generated, _ = generator.generate(
                prompt=prompt,
                language=lang,
                max_new_tokens=30,
                generation_type=GenerationType.GREEDY,
            )
            print(f"Generated: {generated}")
        except Exception as e:
            print(f"Generation note: {e}")


def example_2_language_detection():
    """
    Example 2: Automatic language detection
    """
    print("\n" + "="*70)
    print("Example 2: Language Detection")
    print("="*70)
    
    test_samples = [
        "The quick brown fox jumps over the lazy dog",
        "Hello world, how are you doing today?",
        "Kecerdasan buatan adalah teknologi masa depan",
        "Apa kabar? Bagaimana harimu?",
        "Machine learning dan deep learning adalah",
        "Pembelajaran mendalam menggunakan jaringan neural",
    ]
    
    print("\nDetecting language for sample texts:\n")
    
    for text in test_samples:
        detected = LanguageDetector.detect_language(text)
        language_name = "English" if detected == "en" else "Indonesian"
        print(f"Text: '{text[:50]}...'")
        print(f"Detected: {language_name} ({detected})\n")


def example_3_sampling_strategies():
    """
    Example 3: Different sampling strategies comparison
    """
    print("\n" + "="*70)
    print("Example 3: Sampling Strategies")
    print("="*70)
    
    strategies = [
        (GenerationType.GREEDY, {"description": "Most deterministic"}),
        (GenerationType.TOP_K, {"description": "Top-50 tokens"}),
        (GenerationType.TOP_P, {"description": "Nucleus sampling"}),
    ]
    
    print("\nSampling Strategy Comparison:\n")
    
    for strategy, info in strategies:
        print(f"Strategy: {strategy.value}")
        print(f"Description: {info['description']}")
        print(f"Use case: {info.get('use_case', 'Balanced quality and diversity')}\n")


def example_4_batch_generation():
    """
    Example 4: Batch processing multiple prompts
    """
    print("\n" + "="*70)
    print("Example 4: Batch Generation")
    print("="*70)
    
    device = get_device()
    
    # Setup
    config = ModelConfig()
    model = TransformerModel(config).to(device)
    tokenizer = BaseTokenizer(vocab_size=config.vocab_size)
    
    # Simple training data
    sample_data = ["Hello world"] * 5
    tokenizer.train(sample_data)
    
    generator = TextGenerator(model, tokenizer, device=device)
    batch_gen = BatchGenerator(generator)
    
    # Multiple prompts
    prompts = [
        "Example 1",
        "Example 2",
        "Example 3",
    ]
    
    print(f"\nGenerating completions for {len(prompts)} prompts...\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: {prompt}")
    
    print("\nNote: Batch generation speeds up processing by processing")
    print("multiple sequences simultaneously on the GPU.")


def example_5_configuration():
    """
    Example 5: Model configuration options
    """
    print("\n" + "="*70)
    print("Example 5: Configuration Options")
    print("="*70)
    
    print("\nModel Configuration Parameters:\n")
    
    config_params = {
        "vocab_size": {
            "default": 32000,
            "description": "Size of vocabulary",
            "impact": "Larger = more word coverage but more parameters",
        },
        "embedding_dim": {
            "default": 768,
            "description": "Dimension of embeddings",
            "impact": "Larger = more expressive but slower",
        },
        "num_heads": {
            "default": 12,
            "description": "Number of attention heads",
            "impact": "More heads = better multi-aspect attention",
        },
        "num_layers": {
            "default": 12,
            "description": "Number of transformer layers",
            "impact": "Deeper = more complex patterns but slower",
        },
        "max_sequence_length": {
            "default": 2048,
            "description": "Maximum input length",
            "impact": "Longer = can handle longer texts",
        },
    }
    
    for param, details in config_params.items():
        print(f"{param}:")
        print(f"  Default: {details['default']}")
        print(f"  Purpose: {details['description']}")
        print(f"  Impact: {details['impact']}\n")


def example_6_training_concepts():
    """
    Example 6: Training loop concepts
    """
    print("\n" + "="*70)
    print("Example 6: Training Concepts")
    print("="*70)
    
    concepts = {
        "Gradient Accumulation": {
            "purpose": "Simulate larger batch size with limited memory",
            "formula": "gradient_sum = gradient_sum + backward(loss)",
            "benefit": "Effective larger batch size without OOM",
        },
        "Learning Rate Scheduling": {
            "purpose": "Adapt learning rate during training",
            "formula": "lr(t) = lr_base * schedule_function(t)",
            "benefit": "Better convergence and stability",
        },
        "Gradient Clipping": {
            "purpose": "Prevent exploding gradients",
            "formula": "gradient = gradient * (max_norm / norm(gradient))",
            "benefit": "Stable training, prevent numerical issues",
        },
        "Label Smoothing": {
            "purpose": "Reduce overconfidence and improve generalization",
            "formula": "y_smooth = (1-ε)*y + ε/(K-1)*(1-y)",
            "benefit": "Better generalization to unseen data",
        },
    }
    
    print("\nKey Training Concepts:\n")
    
    for concept, details in concepts.items():
        print(f"{concept}:")
        print(f"  Purpose: {details['purpose']}")
        print(f"  Formula: {details['formula']}")
        print(f"  Benefit: {details['benefit']}\n")


def example_7_inference_tips():
    """
    Example 7: Inference optimization tips
    """
    print("\n" + "="*70)
    print("Example 7: Inference Optimization Tips")
    print("="*70)
    
    tips = [
        {
            "title": "KV Cache",
            "description": "Cache key-value pairs from attention to avoid recomputation",
            "impact": "Speed up generation by 2-3x",
            "when": "Use during text generation",
        },
        {
            "title": "Batch Processing",
            "description": "Process multiple sequences simultaneously",
            "impact": "Increase throughput significantly",
            "when": "Have multiple independent sequences",
        },
        {
            "title": "Quantization",
            "description": "Reduce precision from FP32 to INT8/FP16",
            "impact": "Reduce memory by 2-4x, slight quality loss",
            "when": "Memory-constrained scenarios",
        },
        {
            "title": "Early Stopping",
            "description": "Stop generation when EOS token is reached",
            "impact": "Reduce unnecessary computation",
            "when": "Model predicts well-formed outputs",
        },
        {
            "title": "Top-K/Top-P Sampling",
            "description": "Filter low-probability tokens before sampling",
            "impact": "Faster than full softmax, better quality",
            "when": "Quality matters more than diversity",
        },
    ]
    
    print("\nOptimization Tips:\n")
    
    for i, tip in enumerate(tips, 1):
        print(f"{i}. {tip['title']}")
        print(f"   Description: {tip['description']}")
        print(f"   Impact: {tip['impact']}")
        print(f"   When to use: {tip['when']}\n")


def example_8_language_specific():
    """
    Example 8: Language-specific features
    """
    print("\n" + "="*70)
    print("Example 8: Language-Specific Features")
    print("="*70)
    
    languages = {
        "English": {
            "token": "<en>",
            "special_chars": ".,!?;:\"-'()",
            "challenges": ["Word order variations", "Ambiguous pronouns"],
            "preprocessing": ["URL normalization", "Mention detection"],
        },
        "Indonesian": {
            "token": "<id>",
            "special_chars": ".,!?;:\"-'()",
            "challenges": ["Affixation", "Polysemy"],
            "preprocessing": ["Abbreviation expansion", "Diacritic normalization"],
        },
    }
    
    print("\nLanguage-Specific Configuration:\n")
    
    for lang, config in languages.items():
        print(f"{lang}:")
        print(f"  Token: {config['token']}")
        print(f"  Special chars: {config['special_chars']}")
        print(f"  Challenges:")
        for challenge in config['challenges']:
            print(f"    - {challenge}")
        print(f"  Preprocessing steps:")
        for step in config['preprocessing']:
            print(f"    - {step}")
        print()


def example_9_model_inspection():
    """
    Example 9: Inspecting model structure
    """
    print("\n" + "="*70)
    print("Example 9: Model Inspection")
    print("="*70)
    
    device = get_device()
    
    # Create small model for demo
    config = ModelConfig(
        embedding_dim=256,
        num_heads=4,
        num_layers=2,
    )
    
    model = TransformerModel(config)
    
    print(f"\nModel Structure:\n{model}\n")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # Layer breakdown
    print(f"\nLayer Breakdown:")
    for name, module in model.named_children():
        if hasattr(module, 'parameters'):
            layer_params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {layer_params:,} parameters")


def example_10_error_handling():
    """
    Example 10: Error handling and debugging
    """
    print("\n" + "="*70)
    print("Example 10: Error Handling & Debugging")
    print("="*70)
    
    common_errors = [
        {
            "error": "RuntimeError: CUDA out of memory",
            "causes": ["Batch size too large", "Model too large"],
            "solutions": [
                "Reduce batch_size",
                "Reduce embedding_dim or num_layers",
                "Use gradient accumulation",
                "Enable mixed precision (FP16)",
            ],
        },
        {
            "error": "ValueError: sequence too long",
            "causes": ["Input exceeds max_sequence_length"],
            "solutions": [
                "Increase max_sequence_length in config",
                "Split long sequences into chunks",
                "Use truncation",
            ],
        },
        {
            "error": "Model generates gibberish",
            "causes": ["Not trained enough", "Bad sampling params"],
            "solutions": [
                "Train longer (more epochs)",
                "Use lower temperature",
                "Increase training data quality",
                "Check tokenizer is trained",
            ],
        },
    ]
    
    print("\nCommon Errors & Solutions:\n")
    
    for error_info in common_errors:
        print(f"Error: {error_info['error']}")
        print(f"Causes:")
        for cause in error_info['causes']:
            print(f"  - {cause}")
        print(f"Solutions:")
        for solution in error_info['solutions']:
            print(f"  - {solution}")
        print()


def run_all_examples():
    """Run all examples"""
    print("\n" + "█"*70)
    print("AneryzAI LLM - Examples & Demonstrations")
    print("█"*70)
    
    examples = [
        ("Basic Generation", example_1_basic_generation),
        ("Language Detection", example_2_language_detection),
        ("Sampling Strategies", example_3_sampling_strategies),
        ("Batch Generation", example_4_batch_generation),
        ("Configuration", example_5_configuration),
        ("Training Concepts", example_6_training_concepts),
        ("Inference Tips", example_7_inference_tips),
        ("Language Features", example_8_language_specific),
        ("Model Inspection", example_9_model_inspection),
        ("Error Handling", example_10_error_handling),
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    try:
        choice = input("\nEnter example number (or 'all' to run all): ").strip()
        
        if choice.lower() == 'all':
            for name, example_func in examples:
                try:
                    example_func()
                except Exception as e:
                    print(f"\nNote: {e}")
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            idx = int(choice) - 1
            examples[idx][1]()
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"\n" + "█"*70)
    print("Examples Completed")
    print("█"*70 + "\n")


if __name__ == "__main__":
    run_all_examples()
