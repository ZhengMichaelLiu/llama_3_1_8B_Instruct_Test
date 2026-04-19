"""
Model Loading Module for Llama 3.1 8B

Author: Zheng Liu
Date: 2025/06/19
"""

import os
import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel

DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Models", "Llama_3_1_8B_Instruct"))

class ModelLoader:
    """Handles model and tokenizer loading with device management."""

    def __init__(self, model_name: str = DEFAULT_MODEL_PATH, torch_dtype: Optional[torch.dtype] = None):
        self.device = self._get_device()
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.config = None
        self._load_model()
        self._print_config()

    def _get_device(self):
        """Get optimal device based on availability."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Loading {self.model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Ensure pad_token is set (Llama usually defaults to None)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Tokenizer pad_token set to eos_token: {self.tokenizer.eos_token}")

        # Determine dtype if not provided
        if self.torch_dtype is None:
            if self.device.type == "cuda" or self.device.type == "mps":
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32

        if self.device.type == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"  # Required for output_attentions=True and hooks
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                attn_implementation="eager"  # Required for output_attentions=True and hooks
            ).to(self.device)

        self.model.eval()
        self.config = self.model.config

    def _print_config(self):
        """Print essential model configuration."""
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Layers: {self.config.num_hidden_layers}")
        print(f"Attention heads: {self.config.num_attention_heads}")
        print(f"KV heads: {self.config.num_key_value_heads}")
        print(f"Hidden size: {self.config.hidden_size}")
        print(f"Head dim: {self.config.hidden_size // self.config.num_attention_heads}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration dictionary."""
        return {
            'num_layers': self.config.num_hidden_layers,
            'num_heads': self.config.num_attention_heads,
            'num_kv_heads': self.config.num_key_value_heads,
            'head_dim': self.config.hidden_size // self.config.num_attention_heads,
            'hidden_size': self.config.hidden_size,
            'vocab_size': self.config.vocab_size,
            'device': self.device,
            'dtype': self.model.dtype
        }

    def cleanup(self):
        """Clean up model resources."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Model cleanup completed")
