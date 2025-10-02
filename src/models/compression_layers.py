import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math

class CompressedKVCache:
    """
    Compressed KV Cache container that stores mixed-precision quantized key-value pairs
    """

    def __init__(self, max_batch_size: int, max_seq_len: int, head_dim: int):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.cache_data = {}
        self.compression_info = {}

    def store_compressed_kv(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance_scores: torch.Tensor,
        precision_labels: torch.Tensor,
        selection_mask: torch.Tensor
    ):
        """Store compressed KV data for specific layer"""
        self.cache_data[layer_idx] = {
            'keys': keys.detach(),
            'values': values.detach(),
            'importance_scores': importance_scores.detach(),
            'precision_labels': precision_labels.detach(),
            'selection_mask': selection_mask.detach(),
            'compressed_seq_len': keys.shape[1]
        }

    def get_compressed_kv(self, layer_idx: int) -> Optional[Dict]:
        """Retrieve compressed KV data for specific layer"""
        return self.cache_data.get(layer_idx)

    def clear_cache(self):
        """Clear all cached data"""
        self.cache_data.clear()
        self.compression_info.clear()

class QuantizedLinear(nn.Module):
    """
    Linear layer with mixed-precision weights based on importance
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        precision_bits: int = 8
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.precision_bits = precision_bits

        # Initialize full-precision weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None

        # Quantization parameters
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('weight_zero_point', torch.zeros(1))

    def quantize_weights(self, num_bits: int):
        """Quantize weights to specified bit width"""
        with torch.no_grad():
            w_min = self.weight.min()
            w_max = self.weight.max()

            # Compute scale and zero point
            qmin = 0
            qmax = (2 ** num_bits) - 1
            scale = (w_max - w_min) / (qmax - qmin)
            zero_point = qmin - w_min / scale

            # Quantize and dequantize
            quantized = torch.round(self.weight / scale + zero_point)
            quantized = torch.clamp(quantized, qmin, qmax)
            dequantized = (quantized - zero_point) * scale

            self.weight.data = dequantized
            self.weight_scale.data = scale
            self.weight_zero_point.data = zero_point

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class AdaptiveQuantization(nn.Module):
    """
    Adaptive quantization layer that applies different precisions based on importance
    """

    def __init__(self, feature_dim: int, num_bits_options: list = [2, 4, 8]):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_bits_options = num_bits_options

        # Store quantization parameters for different precisions
        self.quantization_params = {}
        for bits in num_bits_options:
            self.quantization_params[bits] = {
                'scale': torch.ones(1),
                'zero_point': torch.zeros(1)
            }

    def compute_quantization_params(
        self, 
        tensor: torch.Tensor, 
        num_bits: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute quantization scale and zero point"""
        t_min = tensor.min()
        t_max = tensor.max()

        if t_max == t_min:
            return torch.tensor(1.0, device=tensor.device), torch.tensor(0.0, device=tensor.device)

        qmin = 0
        qmax = (2 ** num_bits) - 1
        scale = (t_max - t_min) / (qmax - qmin)
        zero_point = qmin - t_min / scale

        return scale, zero_point

    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        num_bits: int,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        """Apply quantization with given parameters"""
        qmin = 0
        qmax = (2 ** num_bits) - 1

        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        dequantized = (quantized - zero_point) * scale

        return dequantized

    def forward(
        self,
        tensor: torch.Tensor,
        precision_labels: torch.Tensor
    ) -> torch.Tensor:
        """Apply adaptive quantization based on precision labels"""

        output = torch.zeros_like(tensor)

        for precision_level, num_bits in enumerate(self.num_bits_options):
            mask = (precision_labels == precision_level)

            if mask.any():
                # Get tensor subset
                masked_tensor = tensor[mask]

                # Compute quantization parameters
                scale, zero_point = self.compute_quantization_params(masked_tensor, num_bits)

                # Apply quantization
                quantized_tensor = self.quantize_tensor(masked_tensor, num_bits, scale, zero_point)

                # Store back
                output[mask] = quantized_tensor

        return output

class ImportanceGuidedAttention(nn.Module):
    """
    Attention layer with importance-guided token selection
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        layer_idx: int = 0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_idx = layer_idx
        self.dropout = dropout

        # Standard attention layers
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Adaptive quantization
        self.adaptive_quant = AdaptiveQuantization(self.head_dim)

        # Importance scoring components
        self.importance_scorer = nn.Linear(hidden_size, 1)

    def compute_importance_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute token importance scores"""

        batch_size, seq_len, _ = query.shape

        # Compute attention between query and key
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # If prompt mask is provided, focus on prompt attention
        if prompt_mask is not None:
            prompt_attn = attn_weights[:, :, prompt_mask].sum(dim=-1)  # [batch, seq_len]
            importance_base = prompt_attn
        else:
            # Use average attention as importance
            importance_base = attn_weights.mean(dim=-1)  # [batch, seq_len]

        # Apply learned importance scoring
        importance_features = query.mean(dim=1)  # [batch, hidden_size]
        importance_bias = self.importance_scorer(importance_features).squeeze(-1)  # [batch]

        # Combine base importance with learned bias
        importance_scores = importance_base + importance_bias.unsqueeze(1)

        return importance_scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        use_compression: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states) 
        value = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        compression_info = {}

        if use_compression:
            # Compute importance scores (use first head for simplicity)
            importance_scores = self.compute_importance_scores(
                query[:, 0], key[:, 0], prompt_mask
            )

            # Apply dynamic precision assignment (simplified)
            precision_labels = torch.zeros_like(importance_scores, dtype=torch.long)
            precision_labels[importance_scores > 0.7] = 2  # 8-bit
            precision_labels[(importance_scores >= 0.3) & (importance_scores <= 0.7)] = 1  # 4-bit
            # Rest remain 0 (2-bit)

            # Apply adaptive quantization to K, V
            compressed_key = self.adaptive_quant(
                key.transpose(1, 2).contiguous().view(batch_size, seq_len, -1),
                precision_labels
            ).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            compressed_value = self.adaptive_quant(
                value.transpose(1, 2).contiguous().view(batch_size, seq_len, -1),
                precision_labels  
            ).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Token selection based on importance
            selection_threshold = importance_scores.quantile(0.5)  # Keep top 50%
            selection_mask = importance_scores >= selection_threshold

            # Apply selection
            num_selected = selection_mask.sum(dim=1).max().item()
            if num_selected < seq_len:
                # Select tokens
                selected_indices = torch.topk(importance_scores, num_selected, dim=1)[1]

                # Gather selected tokens
                batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_selected)

                compressed_key = compressed_key[batch_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.head_dim), :, selected_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.head_dim)]
                compressed_value = compressed_value[batch_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.head_dim), :, selected_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.head_dim)]

                # Update sequence length
                seq_len = num_selected

            compression_info = {
                'importance_scores': importance_scores,
                'precision_labels': precision_labels,
                'selection_mask': selection_mask,
                'compression_ratio': num_selected / importance_scores.shape[1]
            }

            key = compressed_key
            value = compressed_value

        # Standard attention computation
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.shape[-1] != attn_weights.shape[-1]:
                # Adjust attention mask for compressed sequence
                attention_mask = attention_mask[:, :, :attn_weights.shape[-1]]
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, compression_info

class CompressedTransformerLayer(nn.Module):
    """
    Transformer layer with integrated compression
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        layer_idx: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size

        # Attention with compression
        self.self_attn = ImportanceGuidedAttention(
            hidden_size, num_heads, dropout, layer_idx
        )

        # Layer normalization
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # Feed forward network
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        use_compression: bool = True
    ) -> Tuple[torch.Tensor, Dict]:

        # Self attention with compression
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, attn_weights, compression_info = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            use_compression=use_compression
        )

        hidden_states = residual + self.dropout(attn_output)

        # Feed forward network
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # SwiGLU activation
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden_states = F.silu(gate) * up
        hidden_states = self.down_proj(hidden_states)

        hidden_states = residual + self.dropout(hidden_states)

        return hidden_states, compression_info