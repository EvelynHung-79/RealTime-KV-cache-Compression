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
from transformers import LlamaForCausalLM, LlamaConfig, LlamaModel
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer as HFLlamaDecoderLayer,
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

    # def get_full_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Retrieve and dequantize full KV cache.

    #     Returns:
    #         Tuple of (keys, values) in high precision
    #     """
    #     if not self.key_cache:
    #         return None, None

    #     # Concatenate all cached chunks
    #     keys_quantized = torch.cat(self.key_cache, dim=1)  # [batch, total_seq, num_heads, head_dim]
    #     values_quantized = torch.cat(self.value_cache, dim=1)

    #     # For simplicity, dequantize with the most recent scale
    #     # In practice, you'd track scales per chunk
    #     key_scale = self.key_scales[-1]
    #     value_scale = self.value_scales[-1]

    #     keys = dequantize_symmetric(keys_quantized, key_scale)
    #     values = dequantize_symmetric(values_quantized, value_scale)

    #     return keys, values

    def get_full_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and dequantize full KV cache.

        Returns:
            Tuple of (keys, values) in high precision
        """
        if not self.key_cache:
            return None, None

        dequantized_keys = []
        dequantized_values = []

        # 逐一反量化 (Dequantize chunk by chunk)
        # 迭代 (iterate) 每個 chunk，並使用其對應儲存的 scale
        for k_chunk, v_chunk, k_scale, v_scale in zip(
            self.key_cache, self.value_cache, self.key_scales, self.value_scales
        ):            
            keys_dequant = dequantize_symmetric(k_chunk, k_scale)
            values_dequant = dequantize_symmetric(v_chunk, v_scale)
            
            dequantized_keys.append(keys_dequant)
            dequantized_values.append(values_dequant)

        # 串接 (concatenate) 所有反量化後的高精度 chunks
        # 注意：K/V cache 的 shape 是 [bsz, num_heads, seq_len, head_dim]
        # 我們是在 seq_len 維度 (dim=2) 上進行串接
        
        keys = torch.cat(dequantized_keys, dim=2)
        values = torch.cat(dequantized_values, dim=2)

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

        # 1. 取得量化後的 Key (用於存入快取)
        key_quantized, key_info = quantize_chunk(
            key_states_flat,
            self.key_stats_manager,
            bits_normal=self.compression_config.key_bits_normal,
            bits_high=self.compression_config.key_bits_sink_outlier,
            is_sink_chunk=is_sink_chunk,
            sink_size=sink_size_in_chunk,
            enable_outlier=self.compression_config.outlier_detection_enabled,
        )

        # Reshape back and apply RoPE to quantized keys (for cache)
        key_states_quantized_cache = key_quantized.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        key_states_rope_cache, _ = apply_rotary_pos_emb(key_states_quantized_cache, key_states_quantized_cache, cos, sin, position_ids)

        # 2. 取得高精度的 Key (用於當前計算)
        key_states_rope_high_precision, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin, position_ids)

        # ==== VALUE QUANTIZATION (Post-RoPE, or directly since V doesn't use RoPE) ====
        value_states_flat = value_states.transpose(1, 2).reshape(bsz, q_len, -1)

        # 1. 取得量化後的 Value (用於存入快取)
        value_quantized, value_info = quantize_chunk(
            value_states_flat,
            self.value_stats_manager,
            bits_normal=self.compression_config.value_bits_normal,
            bits_high=self.compression_config.value_bits_sink_outlier,
            is_sink_chunk=is_sink_chunk,
            sink_size=sink_size_in_chunk,
            enable_outlier=self.compression_config.outlier_detection_enabled,
        )

        value_states_quantized_cache = value_quantized.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # 2. 取得高精度的 Value (用於當前計算)
        #    *** 這是關鍵修改：我們直接使用原始的 value_states ***
        #    (value_states_high_precision = value_states)


        # Store in cache if using cache
        if use_cache:
            # *** 修改：存入量化後的版本 ***
            self.kv_cache.append(
                key_states_rope_cache,
                value_states_quantized_cache,
                self.key_stats_manager.get_scale(self.compression_config.key_bits_normal),
                self.value_stats_manager.get_scale(self.compression_config.value_bits_normal),
                key_info["avg_bits"],
                value_info["avg_bits"],
            )

        # ==== ATTENTION COMPUTATION ====
        # For current chunk, use high-precision K/V
        # For past chunks, retrieve and dequantize from cache
    
        past_keys, past_values = self.kv_cache.get_full_kv()

        if past_keys is not None:
            # Concatenate past (dequantized) and current (high-precision)
            key_states_full = torch.cat([past_keys, key_states_rope_high_precision], dim=2)
            value_states_full = torch.cat([past_values, value_states], dim=2)
        else:
            key_states_full = key_states_rope_high_precision
            value_states_full = value_states

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

# 2. 建立自定義的 Decoder Layer
class CompressedLlamaDecoderLayer(HFLlamaDecoderLayer):
    """
    繼承自 Hugging Face 的 LlamaDecoderLayer，
    但將 self.self_attn 替換為我們的 CompressedLlamaAttention
    """
    def __init__(
        self,
        config: LlamaConfig,
        compression_config: CompressionConfig,
        layer_idx: int,
    ):
        # 呼叫父類別 (HFLlamaDecoderLayer) 的 __init__
        # 這會初始化 self.input_layernorm, self.post_attention_layernorm, 
        # self.mlp, 和 *原始的* self.self_attn
        super().__init__(config, layer_idx=layer_idx)

        # *** 關鍵替換 ***
        # 用我們的壓縮版本覆蓋原始的 self.self_attn
        self.self_attn = CompressedLlamaAttention(
            config=config,
            compression_config=compression_config,
            layer_idx=layer_idx,
        )

# 3. 建立自定義的 LlamaForCausalLM
class CompressedLlamaForCausalLM(LlamaForCausalLM):
    """
    繼承自 LlamaForCausalLM，並在初始化時
    將所有 LlamaDecoderLayer 替換為 CompressedLlamaDecoderLayer
    """
    def __init__(self, config: LlamaConfig, compression_config: CompressionConfig):
        # 呼叫父類別 (LlamaForCausalLM) 的 __init__
        # 這會建立完整的 Llama 模型架構 (self.model, self.lm_head)
        super().__init__(config)

        # 儲存壓縮配置
        self.compression_config = compression_config
        
        # 建立統一的壓縮管理器
        # 這個管理器將持有所有層的統計數據
        self.compressor = UnifiedCompressor(compression_config)

        # *** 關鍵替換 ***
        # 迭代模型中的每一層 (self.model.layers)
        new_layers = nn.ModuleList()
        for layer_idx, layer in enumerate(self.model.layers):
            # 建立我們自定義的 Decoder Layer
            compressed_layer = CompressedLlamaDecoderLayer(
                config=config,
                compression_config=compression_config,
                layer_idx=layer_idx,
            )
            
            # 向 compressor 註冊這一層的
            # key 和 value 統計管理器 (stats_manager)
            if hasattr(compressed_layer.self_attn, 'key_stats_manager'):
                self.compressor.register_stats_manager(
                    layer_idx, "key", compressed_layer.self_attn.key_stats_manager
                )
            if hasattr(compressed_layer.self_attn, 'value_stats_manager'):
                self.compressor.register_stats_manager(
                    layer_idx, "value", compressed_layer.self_attn.value_stats_manager
                )
            
            new_layers.append(compressed_layer)

        # 用我們的新層列表替換舊的層列表
        self.model.layers = new_layers

    def reset_compression_state(self):
        """
_summary_
        重置所有層的壓縮統計數據
        """
        self.compressor.reset_compression_state()
        
        # 同時重置每一層 Attention 的快取和位置
        for layer in self.model.layers:
            if hasattr(layer.self_attn, 'reset_cache'):
                layer.self_attn.reset_cache()

    def get_compression_stats(self) -> Dict:
        """
        獲取聚合後的壓縮統計數據
        """
        return self.compressor.get_overall_compression_stats()

    # 讓 from_pretrained 也能接收 compression_config
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        compression_config: CompressionConfig,
        **kwargs,
    ):
        """
        覆蓋 from_pretrained 方法，以確保 compression_config 被正確傳遞
        """
        
        # 首先，使用父類別的方法載入模型
        # 注意：此時載入的是 *未經修改* 的 LlamaForCausalLM
        # 我們傳遞 device_map="auto" 和其他 kwargs
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

        # 接下來，我們需要手動將其轉換為我們的 CompressedLlamaForCausalLM
        # 這是一個常見的技巧：載入權重，然後替換模組

        # 1. 建立一個新的 CompressedLlamaForCausalLM 實例
        #    它會使用載入的 config，並初始化我們的壓縮層
        compressed_model = cls(model.config, compression_config)

        # 2. 將載入的權重複製到新模型中
        compressed_model.load_state_dict(model.state_dict())
        
        # 3. 確保 device_map 被正確處理
        #    如果使用了 device_map，我們需要重新分配模型
        if "device_map" in kwargs:
             # from_pretrained 已經處理了 device_map，
             # 我們的 compressed_model 繼承了 LlamaForCausalLM，
             # 並且在 __init__ 中呼叫了 super().__init__，
             # 但替換層的操作可能會破壞 device_map。
             # 最安全的方式是再次呼叫 device_map 相關的函數，
             # 或者在 __init__ 中更巧妙地處理
             
             # 簡單起見，我們假設 load_state_dict 之後，
             # 權重已經在正確的設備上。
             # 如果遇到 device 問題，可能需要使用 accelerate.dispatch_model
             pass

        # 刪除臨時載入的原始模型
        del model
        
        return compressed_model