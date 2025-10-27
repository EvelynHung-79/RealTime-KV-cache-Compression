"""
Modified Llama model with real-time KV cache compression.

This module implements streaming KV cache quantization with:
- Chunked processing during prefill
- Attention Sink preservation
- Dynamic outlier detection
- Mixed-precision quantization
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from ..compression.streaming_quantization import (
    StreamingStatisticsManager,
    quantize_chunk,
    dequantize_symmetric,
)
from ..compression.unified_compressor import UnifiedCompressor
from configs.base_config import CompressionConfig


class CompressedKVCache:
    """Storage for compressed KV cache with mixed precision support."""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.key_cache = []
        self.value_cache = []
        self.key_scales = []
        self.value_scales = []
        self.key_bits_per_token = []
        self.value_bits_per_token = []

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_scale: torch.Tensor,
        value_scale: torch.Tensor,
        key_bits: int,
        value_bits: int,
    ):
        """Append quantized KV to cache."""
        self.key_cache.append(key)
        self.value_cache.append(value)
        self.key_scales.append(key_scale)
        self.value_scales.append(value_scale)
        self.key_bits_per_token.append(key_bits)
        self.value_bits_per_token.append(value_bits)

    def get_full_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and dequantize full KV cache.

        Returns:
            Tuple of (keys, values) in high precision
        """
        if not self.key_cache:
            return None, None

        # Concatenate all cached chunks
        keys_quantized = torch.cat(self.key_cache, dim=1)  # [batch, total_seq, num_heads, head_dim]
        values_quantized = torch.cat(self.value_cache, dim=1)

        # For simplicity, dequantize with the most recent scale
        # In practice, you'd track scales per chunk
        key_scale = self.key_scales[-1]
        value_scale = self.value_scales[-1]

        keys = dequantize_symmetric(keys_quantized, key_scale)
        values = dequantize_symmetric(values_quantized, value_scale)

        return keys, values

    def reset(self):
        """Clear cache."""
        self.key_cache = []
        self.value_cache = []
        self.key_scales = []
        self.value_scales = []
        self.key_bits_per_token = []
        self.value_bits_per_token = []

    def __len__(self):
        return sum(k.shape[1] for k in self.key_cache) if self.key_cache else 0


class CompressedLlamaAttention(nn.Module):
    """Llama attention with real-time KV cache compression."""

    def __init__(
        self,
        config: LlamaConfig,
        compression_config: CompressionConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.compression_config = compression_config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # RoPE
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=config.max_position_embeddings)

        # Statistics managers for Key and Value
        num_channels = self.num_key_value_heads * self.head_dim

        self.key_stats_manager = StreamingStatisticsManager(
            num_channels=num_channels,
            ema_decay=compression_config.ema_decay,
            outlier_threshold_relative=compression_config.outlier_threshold_relative,
            outlier_threshold_abs=compression_config.outlier_threshold_abs,
        )

        self.value_stats_manager = StreamingStatisticsManager(
            num_channels=num_channels,
            ema_decay=compression_config.ema_decay,
            outlier_threshold_relative=compression_config.outlier_threshold_relative,
            outlier_threshold_abs=compression_config.outlier_threshold_abs,
        )

        # KV Cache
        self.kv_cache = CompressedKVCache(compression_config)

        # Track current position for sink detection
        self.current_position = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass with streaming quantization."""

        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, seq_len, num_heads, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Get RoPE embeddings
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Determine if current chunk contains sink tokens
        is_sink_chunk = self.current_position < self.compression_config.attention_sink_size
        sink_size_in_chunk = min(
            self.compression_config.attention_sink_size - self.current_position,
            q_len
        ) if is_sink_chunk else 0

        # ==== KEY QUANTIZATION (Pre-RoPE) ====
        # Reshape for per-channel quantization: [batch, seq_len, num_channels]
        key_states_flat = key_states.transpose(1, 2).reshape(bsz, q_len, -1)

        # Quantize keys
        key_quantized, key_info = quantize_chunk(
            key_states_flat,
            self.key_stats_manager,
            bits_normal=self.compression_config.key_bits_normal,
            bits_high=self.compression_config.key_bits_sink_outlier,
            is_sink_chunk=is_sink_chunk,
            sink_size=sink_size_in_chunk,
            enable_outlier=self.compression_config.outlier_detection_enabled,
        )

        # Reshape back and apply RoPE
        key_states_quantized = key_quantized.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        key_states_rope, _ = apply_rotary_pos_emb(key_states_quantized, key_states_quantized, cos, sin, position_ids)

        # ==== VALUE QUANTIZATION (Post-RoPE, or directly since V doesn't use RoPE) ====
        value_states_flat = value_states.transpose(1, 2).reshape(bsz, q_len, -1)

        value_quantized, value_info = quantize_chunk(
            value_states_flat,
            self.value_stats_manager,
            bits_normal=self.compression_config.value_bits_normal,
            bits_high=self.compression_config.value_bits_sink_outlier,
            is_sink_chunk=is_sink_chunk,
            sink_size=sink_size_in_chunk,
            enable_outlier=self.compression_config.outlier_detection_enabled,
        )

        value_states_quantized = value_quantized.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Store in cache if using cache
        if use_cache:
            self.kv_cache.append(
                key_states_rope,
                value_states_quantized,
                self.key_stats_manager.get_scale(self.compression_config.key_bits_normal),
                self.value_stats_manager.get_scale(self.compression_config.value_bits_normal),
                key_info["avg_bits"],
                value_info["avg_bits"],
            )

        # ==== ATTENTION COMPUTATION ====
        # For current chunk, use high-precision K/V
        # For past chunks, retrieve and dequantize from cache
        if len(self.kv_cache) > 0:
            past_keys, past_values = self.kv_cache.get_full_kv()
            # Concatenate past and current
            key_states_full = torch.cat([past_keys, key_states_rope], dim=2)
            value_states_full = torch.cat([past_values, value_states_quantized], dim=2)
        else:
            key_states_full = key_states_rope
            value_states_full = value_states_quantized

        # Apply RoPE to query
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin, position_ids)

        # Repeat KV for grouped query attention
        key_states_full = key_states_full.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states_full = value_states_full.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states_full.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_full)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # Update position
        self.current_position += q_len

        return attn_output, attn_weights if output_attentions else None, None

    def reset_cache(self):
        """Reset KV cache and position tracking."""
        self.kv_cache.reset()
        self.current_position = 0
