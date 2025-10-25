from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import torch

@dataclass
class CompressionConfig:
    """
    Configuration for Streaming KVQuant Quantization (Sink/Outlier Aware).
    Removes parameters related to importance scoring and selective propagation.
    Adds parameters for chunking, streaming statistics, and mixed-precision quantization.
    """

    # Model configuration (保留)
    model_name: str = "meta-llama/Llama-2-7b-hf"
    max_position_embeddings: int = 4096
    num_hidden_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    # num_key_value_heads: int = 32 # Usually inferred from model config

    # Streaming and Chunking
    chunk_size: int = 256 # Size of token chunks for processing during prefill

    # Streaming Statistics (EMA Absmax)
    ema_decay: float = 0.99 # Decay factor for EMA Absmax updates

    # Outlier Detection
    outlier_threshold_abs: Optional[float] = 6.0 # Absolute threshold to mark channel/group as outlier (optional)
    outlier_threshold_relative: Optional[float] = 5.0 # Relative threshold (multiple of EMA Absmax) to mark as outlier (optional)

    # Attention Sink Handling
    attention_sink_size: int = 8 # Number of initial tokens to treat as Attention Sink

    # Quantization Settings (Per-channel Key, Group/Per-channel Value)
    key_bits_normal: int = 4 # Bits for normal Key channels
    key_bits_sink_outlier: int = 8 # Bits for Key channels corresponding to Sink tokens or marked as Outliers

    value_bits_normal: int = 4 # Bits for normal Value groups/channels
    value_bits_sink_outlier: int = 8 # Bits for Value groups/channels corresponding to Sink tokens or marked as Outliers

    value_quant_groups: int = -1 # Value quantization grouping. -1 for per-channel, 1 for per-tensor, >1 for group-wise.
                               # Example: Set to hidden_size // 64 for groups of 64.

    # Evaluation settings (保留)
    context_lengths: List[int] = field(default_factory=lambda: [4096, 8192, 16384, 32768])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8])

    def __post_init__(self):
        # Infer num_key_value_heads if possible (basic example)
        # In practice, load this from the actual model config
        if "7b" in self.model_name:
            self.num_key_value_heads = 32
        elif "13b" in self.model_name:
             self.num_key_value_heads = 40 # Example, adjust based on actual model
        else:
             self.num_key_value_heads = self.num_attention_heads

        # Validate value_quant_groups
        if self.value_quant_groups > 1 and self.hidden_size % self.value_quant_groups != 0:
            print(f"Warning: hidden_size ({self.hidden_size}) not divisible by value_quant_groups ({self.value_quant_groups}). Falling back to per-channel.")
            self.value_quant_groups = -1

        print(f"Initialized CompressionConfig:\n"
              f"  chunk_size={self.chunk_size}\n"
              f"  ema_decay={self.ema_decay}\n"
              f"  outlier_thresholds(abs/rel)={self.outlier_threshold_abs}/{self.outlier_threshold_relative}\n"
              f"  attention_sink_size={self.attention_sink_size}\n"
              f"  key_bits(norm/sink)={self.key_bits_normal}/{self.key_bits_sink_outlier}\n"
              f"  value_bits(norm/sink)={self.value_bits_normal}/{self.value_bits_sink_outlier}\n"
              f"  value_quant_groups={self.value_quant_groups}")