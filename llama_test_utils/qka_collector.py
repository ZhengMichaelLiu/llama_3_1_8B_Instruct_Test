"""
Q/K and Attention Matrix Collection with RoPE Applied

Author: Zheng Liu
Date: 2025/06/19
"""

import torch
from typing import Dict, Tuple, Optional, Any
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""
    # cos, sin: [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class QKAttentionCollector:
    """Collects Q/K matrices with RoPE applied and attention matrices from all layers."""

    def __init__(self, model_loader):
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.device = model_loader.device
        self.config = model_loader.config

        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads

    def collect_qk_attention_matrices(self, text: str, max_length: int = 3072) -> Tuple[Dict, Dict, Dict]:
        """Extract Q/K matrices with RoPE applied and attention matrices from all layers."""
        print(f"Tokenizing text (max_length={max_length})")

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        seq_len = inputs['input_ids'].shape[1]
        print(f"Sequence length: {seq_len}")

        # Storage for matrices
        qk_matrices: Dict[int, Dict[str, torch.Tensor]] = {}
        attention_matrices: Dict[int, torch.Tensor] = {}
        # Storage for captured RoPE embeddings
        rope_cache = {}

        # Hook to capture RoPE cos/sin from the model
        def create_rope_hook():
            def hook_fn(module, input, output):
                # output is (cos, sin)
                # We detach and move to CPU immediately to save GPU memory
                rope_cache['cos'] = output[0].detach().cpu()
                rope_cache['sin'] = output[1].detach().cpu()
            return hook_fn

        def create_q_hook(layer_idx):
            def hook_fn(module, input, output):
                # output: [batch, seq_len, hidden_size]
                bsz, seq_len, _ = output.shape
                # Reshape: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
                q_states = output.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                if layer_idx not in qk_matrices:
                    qk_matrices[layer_idx] = {}
                qk_matrices[layer_idx]['Q'] = q_states.detach().cpu()
            return hook_fn

        def create_k_hook(layer_idx):
            def hook_fn(module, input, output):
                bsz, seq_len, _ = output.shape
                # Reshape: [batch, seq_len, num_kv_heads, head_dim]
                k_states = output.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

                # Handle GQA - repeat K states to match Q heads for visualization/analysis
                if self.num_kv_heads != self.num_heads:
                    repeat_factor = self.num_heads // self.num_kv_heads
                    k_states = k_states.repeat_interleave(repeat_factor, dim=2)

                k_states = k_states.transpose(1, 2) # [batch, num_heads, seq_len, head_dim]
                if layer_idx not in qk_matrices:
                    qk_matrices[layer_idx] = {}
                qk_matrices[layer_idx]['K'] = k_states.detach().cpu()
            return hook_fn

        def create_attention_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is (attn_output, attn_weights, past_key_value)
                if len(output) > 1 and output[1] is not None:
                    # attn_weights: [batch, num_heads, seq_len, seq_len]
                    attention_matrices[layer_idx] = output[1].detach().cpu()
            return hook_fn

        # Register hooks
        hooks = []

        # 1. Hook the Rotary Embedding Module
        # LlamaForCausalLM -> model (LlamaModel) -> rotary_emb (LlamaRotaryEmbedding)
        if hasattr(self.model, "model") and hasattr(self.model.model, "rotary_emb"):
            hooks.append(self.model.model.rotary_emb.register_forward_hook(create_rope_hook()))
        else:
            print("WARNING: Could not find 'model.model.rotary_emb'. RoPE capture might fail.")

        # 2. Hook Attention Layers
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]

            # Hook Projections (Pre-RoPE)
            hooks.append(layer.self_attn.q_proj.register_forward_hook(create_q_hook(layer_idx)))
            hooks.append(layer.self_attn.k_proj.register_forward_hook(create_k_hook(layer_idx)))

            # Hook Attention Weights
            hooks.append(layer.self_attn.register_forward_hook(create_attention_hook(layer_idx)))

        # Forward pass
        print("Running forward pass...")
        with torch.no_grad():
            self.model(**inputs, output_attentions=True)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Apply RoPE to collected Q/K matrices
        print("Applying RoPE to Q/K matrices...")

        # Retrieve captured RoPE
        cos = rope_cache.get('cos')
        sin = rope_cache.get('sin')

        if cos is None or sin is None:
            raise RuntimeError("Failed to capture RoPE embeddings during forward pass.")

        q_matrices_rope = {}
        k_matrices_rope = {}

        for layer_idx in range(self.num_layers):
            if layer_idx in qk_matrices:
                Q = qk_matrices[layer_idx]['Q']
                K = qk_matrices[layer_idx]['K']

                # Apply RoPE using the captured cos/sin
                # Ensure Q/K are float32 for analysis if needed, or keep original dtype
                # cos/sin are already on CPU
                Q_rope, K_rope = apply_rotary_pos_emb(Q, K, cos, sin, unsqueeze_dim=1)

                q_matrices_rope[layer_idx] = Q_rope
                k_matrices_rope[layer_idx] = K_rope

        print(f"Collected Q/K matrices for {len(q_matrices_rope)} layers")
        print(f"Collected attention matrices for {len(attention_matrices)} layers")

        return q_matrices_rope, k_matrices_rope, attention_matrices