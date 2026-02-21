"""
Transformer Model Architecture
Implements multi-head self-attention, feed-forward networks, and transformer layers
For AneryzAI LLM
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import numpy as np

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    Applied to query and key tensors before computing attention
    """
    
    def __init__(self, dim: int, max_length: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        
        # Pre-compute rotation matrices
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Pre-compute position indices
        t = torch.arange(max_length).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, dim)
        Returns:
            Tuple of (cos_emb, sin_emb) for applying rotation
        """
        seq_len = x.shape[1]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return cos, sin

    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to x"""
        # Reshape x for rotation
        x = x.view(*x.shape[:-1], -1, 2)
        # Apply rotation: [cos, -sin; sin, cos]
        x_rot = torch.stack([
            x[..., 0] * cos[..., 0] - x[..., 1] * sin[..., 0],
            x[..., 0] * sin[..., 0] + x[..., 1] * cos[..., 0],
        ], dim=-1)
        return x_rot.flatten(-2)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with causal masking for autoregressive models
    """
    
    def __init__(self, embedding_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_dropout = nn.Dropout(dropout_rate)
        
        # Scale factor for attention scores
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: (batch_size, seq_len, embedding_dim)
            attention_mask: (batch_size, seq_len) or (batch_size, 1, seq_len, seq_len)
            use_cache: Whether to return cached key-value pairs for generation
            past_key_value: Cached key-value from previous step
        
        Returns:
            Tuple of (output, cache) where cache is (key, value) if use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Use cached KV from previous step if available
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=1)
            v = torch.cat([past_value, v], dim=1)
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, embedding_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(2, 3)) / self.scale
        
        # Apply attention mask (causal mask for autoregressive)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Convert (batch_size, seq_len) to causal mask
                device = scores.device
                causal_mask = torch.triu(
                    torch.ones(seq_len, k.shape[2], device=device) * float('-inf'),
                    diagonal=1
                )
                scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
            else:
                scores = scores + attention_mask
        else:
            # Apply causal mask automatically
            device = scores.device
            causal_mask = torch.triu(
                torch.ones(seq_len, k.shape[2], device=device) * float('-inf'),
                diagonal=1
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape back to (batch_size, seq_len, embedding_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.embedding_dim)
        
        # Final projection
        output = self.output_proj(output)
        output = self.dropout(output)
        
        # Cache for next step if use_cache
        next_cache = (k, v) if use_cache else None
        
        return output, next_cache


class FeedForwardNetwork(nn.Module):
    """Feed-Forward Network (FFN) with GELU activation"""
    
    def __init__(self, embedding_dim: int, ffn_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, embedding_dim)
        Returns:
            (batch_size, seq_len, embedding_dim)
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    """Single Transformer Layer with multi-head attention and FFN"""
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout_rate: float = 0.1,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout_rate)
        self.ffn = FeedForwardNetwork(embedding_dim, ffn_dim, dropout_rate)
        
        self.norm1 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: (batch_size, seq_len, embedding_dim)
            attention_mask: Optional attention mask
            use_cache: Whether to cache key-value
            past_key_value: Cached key-value from previous step
        
        Returns:
            Tuple of (output, cache)
        """
        # Self-attention with residual connection
        norm_hidden = self.norm1(hidden_states)
        attn_output, cache = self.attention(
            norm_hidden,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value,
        )
        hidden_states = hidden_states + attn_output
        
        # Feed-forward with residual connection
        norm_hidden = self.norm2(hidden_states)
        ffn_output = self.ffn(norm_hidden)
        hidden_states = hidden_states + ffn_output
        
        return hidden_states, cache


class TransformerModel(nn.Module):
    """Complete Transformer Model for Language Modeling"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=0  # <pad> token
        )
        
        # Positional embedding
        self.position_embedding = nn.Embedding(
            config.max_sequence_length,
            config.embedding_dim
        )
        
        # RoPE (Rotary Position Embedding)
        self.rope = RotaryPositionalEmbedding(
            config.embedding_dim,
            max_length=config.max_sequence_length
        )
        
        self.embedding_dropout = nn.Dropout(config.dropout_rate)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout_rate=config.dropout_rate,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        # Tie embeddings with output projection for efficiency
        self.lm_head.weight = self.token_embedding.weight
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using normal distribution"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[list] = None,
        return_all_layers: bool = False,
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            use_cache: Whether to return cached key-values for generation
            past_key_values: List of cached key-values from previous forward passes
            return_all_layers: Whether to return hidden states from all layers
        
        Returns:
            Dictionary with:
                - logits: (batch_size, seq_len, vocab_size)
                - cache: List of cached key-values if use_cache=True
                - hidden_states: All layer outputs if return_all_layers=True
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embedding
        hidden_states = self.token_embedding(input_ids)
        
        # Position embedding
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(position_ids)
        hidden_states = hidden_states + position_emb
        
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Cache for all layers if use_cache
        new_cache = [] if use_cache else None
        all_hidden_states = [hidden_states] if return_all_layers else None
        
        # RoPE embeddings
        cos, sin = self.rope(hidden_states)
        
        # Transformer layers
        for i, layer in enumerate(self.layers):
            # Get past key-value if available
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, layer_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_kv,
            )
            
            if use_cache:
                new_cache.append(layer_cache)
            
            if return_all_layers:
                all_hidden_states.append(hidden_states)
        
        # Final layer norm
        hidden_states = self.final_norm(hidden_states)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        output = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        if use_cache:
            output["past_key_values"] = new_cache
        
        if return_all_layers:
            output["all_hidden_states"] = all_hidden_states
        
        return output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        num_beams: int = 1,
        repetition_penalty: float = 1.2,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively
        Args:
            input_ids: (batch_size, seq_len) - Starting tokens
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-K sampling parameter
            top_p: Top-P (nucleus) sampling parameter
            num_beams: Number of beams for beam search
            repetition_penalty: Penalty for repeating tokens
            use_cache: Whether to use KV cache for speed
        
        Returns:
            Generated token ids (batch_size, seq_len + max_new_tokens)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        past_key_values = None
        generated_ids = input_ids.clone()
        
        for step in range(max_new_tokens):
            # Get model output for last token
            if use_cache and step > 0:
                # Only pass the last token when using cache
                model_input = generated_ids[:, -1:] if step > 0 else generated_ids
            else:
                model_input = generated_ids
            
            output = self.forward(
                model_input,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
            
            logits = output["logits"][:, -1, :]  # Get last position logits
            
            if use_cache:
                past_key_values = output.get("past_key_values")
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply repetition penalty
            for i in range(batch_size):
                for token_id in generated_ids[i]:
                    logits[i, token_id] /= repetition_penalty
            
            # Top-K filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 0] = False  # Keep at least one token
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = float('-inf')
            
            # Sample or greedy select
            if num_beams == 1:
                # Sample from distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy selection
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        return generated_ids
