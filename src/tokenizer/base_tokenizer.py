"""
Base Tokenizer for AneryzAI LLM
Supports English and Indonesian languages
"""

import re
import json
from typing import List, Tuple, Dict, Set
from pathlib import Path
import numpy as np

class BaseTokenizer:
    """Base tokenizer with BPE-like algorithm for multiple languages"""
    
    def __init__(self, vocab_size: int = 32000, special_tokens: List[str] = None):
        """
        Initialize tokenizer
        Args:
            vocab_size: Target vocabulary size
            special_tokens: List of special tokens to include
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or [
            "<pad>", "<unk>", "<bos>", "<eos>",
            "<en>", "<id>", "<class>"
        ]
        
        # Initialize vocabulary
        self.word2idx = {}
        self.idx2word = {}
        self.merges = {}  # For BPE
        self.word_freq = {}  # Frequency count
        
        # Build initial special tokens vocabulary
        self._init_special_tokens()
        
    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        for i, token in enumerate(self.special_tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token
    
    def _preprocess_text(self, text: str, language: str = "en") -> str:
        """
        Preprocess text based on language
        Args:
            text: Raw text to preprocess
            language: Language code ("en" or "id")
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        if language == "en":
            # English preprocessing
            text = re.sub(r"http\S+", "<url>", text)  # Handle URLs
            text = re.sub(r"\b@\w+", "<mention>", text)  # Handle mentions
            text = re.sub(r"#\w+", "<hashtag>", text)  # Handle hashtags
            
        elif language == "id":
            # Indonesian preprocessing
            text = re.sub(r"http\S+", "<url>", text)
            text = re.sub(r"\b@\w+", "<mention>", text)
            text = re.sub(r"#\w+", "<hashtag>", text)
        
        return text
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a word into characters
        Args:
            word: Word to tokenize
        Returns:
            List of characters
        """
        return list(word) + ["</w>"]
    
    def train(self, texts: List[str], language: str = "en"):
        """
        Train tokenizer on texts using BPE algorithm
        Args:
            texts: List of training texts
            language: Language code
        """
        print(f"Training tokenizer on {len(texts)} texts (Language: {language})...")
        
        # Step 1: Preprocess and create initial vocabulary
        vocab = {}
        for text in texts:
            text = self._preprocess_text(text, language)
            # Split by whitespace and punctuation, keep punctuation
            words = re.findall(r"\w+|[^\w\s]", text)
            
            for word in words:
                word_tokens = self._tokenize_word(word)
                word_tuple = tuple(word_tokens)
                vocab[word_tuple] = vocab.get(word_tuple, 0) + 1
        
        self.word_freq = {' '.join(k): v for k, v in vocab.items()}
        
        # Step 2: BPE merging
        merges = {}
        for i in range(self.vocab_size - len(self.special_tokens)):
            # Find most frequent adjacent pair
            pairs = {}
            for word, freq in vocab.items():
                for j in range(len(word) - 1):
                    pair = (word[j], word[j + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            merges[best_pair] = i + len(self.special_tokens)
            
            # Merge best pair in vocabulary
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = self._merge_pair(word, best_pair)
                new_vocab[new_word] = freq
            vocab = new_vocab
            
            if (i + 1) % 1000 == 0:
                print(f"  Completed {i + 1} merges...")
        
        self.merges = merges
        
        # Build final vocabulary
        vocab_idx = len(self.special_tokens)
        for word in vocab:
            vocab_str = ' '.join(word)
            if vocab_str not in self.word2idx:
                self.word2idx[vocab_str] = vocab_idx
                self.idx2word[vocab_idx] = vocab_str
                vocab_idx += 1
        
        print(f"Tokenizer training completed. Vocabulary size: {len(self.word2idx)}")
    
    def _merge_pair(self, word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        """Merge a pair in a word"""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)
    
    def encode(self, text: str, language: str = "en", add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs
        Args:
            text: Text to encode
            language: Language code
            add_special_tokens: Whether to add BOS and EOS tokens
        Returns:
            List of token IDs
        """
        # Add language token
        tokens = []
        if add_special_tokens:
            lang_token = "<en>" if language == "en" else "<id>"
            tokens.append(self.word2idx[lang_token])
            tokens.append(self.word2idx["<bos>"])
        
        # Preprocess
        text = self._preprocess_text(text, language)
        
        # Tokenize
        words = re.findall(r"\w+|[^\w\s]", text)
        
        for word in words:
            word_tokens = self._tokenize_word(word)
            # Encode word tokens
            for token in word_tokens:
                token_id = self.word2idx.get(token, self.word2idx["<unk>"])
                tokens.append(token_id)
        
        if add_special_tokens:
            tokens.append(self.word2idx["<eos>"])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text
        Args:
            token_ids: List of token IDs
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.idx2word:
                token = self.idx2word[token_id]
                # Skip special tokens
                if token not in self.special_tokens:
                    tokens.append(token)
        
        # Join tokens
        text = ''.join(tokens).replace("</w>", " ").strip()
        return text
    
    def save(self, path: str):
        """Save tokenizer to disk"""
        save_data = {
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
            "merges": {str(k): v for k, v in self.merges.items()},
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
        }
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"Tokenizer saved to {path}")
    
    def load(self, path: str):
        """Load tokenizer from disk"""
        with open(path, 'r') as f:
            save_data = json.load(f)
        
        self.word2idx = save_data["word2idx"]
        self.idx2word = {int(k): v for k, v in save_data["idx2word"].items()}
        self.merges = save_data["merges"]
        self.special_tokens = save_data["special_tokens"]
        self.vocab_size = save_data["vocab_size"]
        print(f"Tokenizer loaded from {path}")
