import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from configs.base_config import CompressionConfig # 引用更新後的 Config

class StreamingStatisticsManager(nn.Module):
    """
    Manages running statistics (EMA Absmax) for Key and Value quantization
    and detects outliers based on configured thresholds.
    Designed to be part of the main model for state persistence.
    """
    def __init__(self, config: CompressionConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads # Use num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Buffers for EMA Absmax
        # Key: Per-channel (Layer, Head, Channel)
        self.register_buffer('key_ema_absmax', torch.zeros(self.num_layers, self.num_kv_heads, self.head_dim))
        # Value: Group-wise or Per-channel
        value_stat_shape = self._get_value_stat_shape()
        self.register_buffer('value_ema_absmax', torch.zeros(value_stat_shape))

        # Buffers for Outlier status (can be boolean or float for weighted updates)
        self.register_buffer('key_is_outlier', torch.zeros(self.num_layers, self.num_kv_heads, self.head_dim, dtype=torch.bool))
        self.register_buffer('value_is_outlier', torch.zeros(value_stat_shape, dtype=torch.bool))

        self.decay = config.ema_decay
        self.abs_thresh = config.outlier_threshold_abs
        self.rel_thresh = config.outlier_threshold_relative

        self.initialized = nn.Parameter(torch.tensor(False), requires_grad=False) # Track if stats are initialized

    def _get_value_stat_shape(self):
        if self.config.value_quant_groups == -1: # Per-channel
            return (self.num_layers, self.num_kv_heads, self.head_dim)
        elif self.config.value_quant_groups == 1: # Per-tensor (per layer, per head)
            return (self.num_layers, self.num_kv_heads, 1)
        else: # Group-wise
             num_groups = self.head_dim // (self.head_dim // self.config.value_quant_groups) # Ensure divisibility checked in config
             # Shape: (Layer, Head, Group) - assuming groups are within head_dim
             # Adjust if grouping spans heads or other dimensions
             return (self.num_layers, self.num_kv_heads, num_groups)


    @torch.no_grad()
    def update(self, layer_idx: int, tensor_type: str, chunk_tensor: torch.Tensor):
        """
        Update EMA Absmax statistics and detect outliers for the given chunk.
        chunk_tensor shape: [batch, num_kv_heads, chunk_len, head_dim] (after potential reshape)
        """
        if chunk_tensor.numel() == 0:
            return

        is_key = tensor_type == 'key'
        ema_buffer = self.key_ema_absmax if is_key else self.value_ema_absmax
        outlier_buffer = self.key_is_outlier if is_key else self.value_is_outlier
        stat_shape = ema_buffer.shape # e.g., (L, H, D) or (L, H, G)

        # 1. Calculate current chunk's absmax per channel/group
        abs_chunk = chunk_tensor.abs()

        if is_key or self.config.value_quant_groups == -1: # Per-channel
            # Max across batch and chunk_len: result shape [num_kv_heads, head_dim]
            current_absmax = abs_chunk.amax(dim=(0, 2))
            target_ema = ema_buffer[layer_idx] # Shape [H, D]
            target_outlier = outlier_buffer[layer_idx] # Shape [H, D]
        elif self.config.value_quant_groups == 1: # Per-tensor (per head)
            current_absmax = abs_chunk.amax(dim=(0, 2, 3), keepdim=True) # Shape [H, 1]
            target_ema = ema_buffer[layer_idx] # Shape [H, 1]
            target_outlier = outlier_buffer[layer_idx] # Shape [H, 1]
        else: # Group-wise
             # Reshape chunk: [batch, H, chunk_len, num_groups, group_size]
             num_groups = stat_shape[2]
             group_size = self.head_dim // num_groups
             grouped_chunk = abs_chunk.view(chunk_tensor.shape[0], self.num_kv_heads, chunk_tensor.shape[2], num_groups, group_size)
             # Max across batch, chunk_len, group_size: result shape [H, num_groups]
             current_absmax = grouped_chunk.amax(dim=(0, 2, 4))
             target_ema = ema_buffer[layer_idx] # Shape [H, G]
             target_outlier = outlier_buffer[layer_idx] # Shape [H, G]


        # 2. Update EMA Absmax
        if not self.initialized:
             # Initialize with the first chunk's absmax
             updated_ema = current_absmax
             if layer_idx == self.num_layers -1 and tensor_type == 'value': # Rough check for last update
                 self.initialized.data = torch.tensor(True)
        else:
            updated_ema = self.decay * target_ema + (1.0 - self.decay) * current_absmax

        # Ensure no zeros if starting from zeros
        updated_ema = torch.maximum(updated_ema, torch.tensor(1e-5, device=updated_ema.device))
        target_ema.copy_(updated_ema)


        # 3. Detect Outliers (only if initialized and thresholds are set)
        if self.initialized and (self.abs_thresh is not None or self.rel_thresh is not None):
            is_outlier_abs = (current_absmax > self.abs_thresh) if self.abs_thresh is not None else False
            is_outlier_rel = (current_absmax > self.rel_thresh * target_ema) if self.rel_thresh is not None else False # Compare with PREVIOUS EMA
            new_outliers = torch.logical_or(is_outlier_abs, is_outlier_rel)

            # Update outlier status - maybe make it decay over time? For now, just set.
            target_outlier.copy_(new_outliers)


    def get_stats(self, layer_idx: int, tensor_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (ema_absmax, is_outlier) for the layer/type"""
        is_key = tensor_type == 'key'
        ema_buffer = self.key_ema_absmax if is_key else self.value_ema_absmax
        outlier_buffer = self.key_is_outlier if is_key else self.value_is_outlier
        return ema_buffer[layer_idx], outlier_buffer[layer_idx]

    def reset(self):
        """Reset statistics and outlier status."""
        self.key_ema_absmax.zero_()
        self.value_ema_absmax.zero_()
        self.key_is_outlier.zero_()
        self.value_is_outlier.zero_()
        self.initialized.data = torch.tensor(False)

# --- Quantization Functions ---

@torch.no_grad()
def calculate_scale(absmax: torch.Tensor, bits: int, epsilon=1e-5) -> torch.Tensor:
    """Calculates scale for symmetric quantization."""
    qmax = (2**(bits - 1)) - 1
    # Add epsilon to prevent scale being zero
    scale = torch.clamp(absmax, min=epsilon) / qmax
    return scale

@torch.no_grad()
def quantize_symmetric(tensor: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """Performs symmetric quantization."""
    qmax = (2**(bits - 1)) - 1
    qmin = -(2**(bits - 1))

    # Ensure scale has compatible dimensions for broadcasting
    # tensor: [..., feature_dim], scale: [feature_dim] or [num_groups] or [1]
    scale = scale.unsqueeze(0).unsqueeze(-2) # Adjust based on tensor and scale shapes needed

    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)

    # Determine integer type based on bits
    if bits <= 8:
        return quantized.to(torch.int8)
    else: # Should not happen often for KV Cache, maybe INT16 if needed
        return quantized.to(torch.int16)


@torch.no_grad()
def dequantize_symmetric(quantized_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Performs symmetric dequantization."""
    # Ensure scale has compatible dimensions
    scale = scale.unsqueeze(0).unsqueeze(-2) # Adjust based on tensor and scale shapes needed
    return quantized_tensor.float() * scale

@torch.no_grad()
def quantize_chunk(
    chunk_tensor: torch.Tensor, #[B, H, C_len, D]
    layer_idx: int,
    tensor_type: str,
    token_indices: torch.Tensor, #[C_len] global indices of tokens in chunk
    stats_manager: StreamingStatisticsManager,
    config: CompressionConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes a chunk considering Sink/Outlier status.
    Returns (quantized_chunk, scale)
    """
    is_key = tensor_type == 'key'
    normal_bits = config.key_bits_normal if is_key else config.value_bits_normal
    special_bits = config.key_bits_sink_outlier if is_key else config.value_bits_sink_outlier

    ema_absmax, is_outlier = stats_manager.get_stats(layer_idx, tensor_type)
    # ema_absmax shape: [H, D] or [H, G] or [H, 1]
    # is_outlier shape: same as ema_absmax

    # Determine bits per channel/group based on outlier status
    if ema_absmax.shape == is_outlier.shape:
        bits_per_stat = torch.where(is_outlier, special_bits, normal_bits) # Shape [H, D] or [H, G] or [H, 1]
    else: # Fallback if shapes mismatch (shouldn't happen)
        bits_per_stat = torch.full_like(ema_absmax, normal_bits, dtype=torch.long)

    # Expand bits_per_stat to match chunk_tensor's feature dimension if needed
    if config.value_quant_groups > 1 and not is_key: # Group-wise Value
        num_groups = ema_absmax.shape[-1]
        group_size = config.head_dim // num_groups
        # Expand bits from [H, G] to [H, D] for applying scale
        bits_expanded = bits_per_stat.unsqueeze(-1).repeat(1, 1, group_size).view(config.num_kv_heads, config.head_dim)
        # Expand absmax similarly for scale calculation
        absmax_expanded = ema_absmax.unsqueeze(-1).repeat(1, 1, group_size).view(config.num_kv_heads, config.head_dim)
    else: # Per-channel or Per-tensor
        bits_expanded = bits_per_stat # Already [H, D] or [H, 1]
        absmax_expanded = ema_absmax

    # Calculate scale based on potentially mixed bits (use higher bit qmax for safety if mixing)
    # Simpler: Calculate scale based on absmax and TARGET bits (normal/special) separately
    # This requires quantizing different parts separately or complex masking.

    # -- Simplified approach: Calculate scale based on absmax and assign bits later --
    # Calculate scale using the MAX potential bits (special_bits) to avoid clipping normal values prematurely
    # This might slightly reduce precision for normal values if special_bits > normal_bits significantly.
    # Alternative: Calculate two scales (normal, special) and use masking during quantization.
    # Let's try the simpler single-scale approach first, based on absmax only.
    scale = calculate_scale(absmax_expanded, special_bits) # Shape [H, D] or [H, 1]

    # Quantize the whole chunk using the calculated scale
    # Need to apply different bit clipping based on Sink/Outlier status

    # Identify Sink tokens within the chunk
    is_sink_token = token_indices < config.attention_sink_size # Shape [C_len]

    # Combine Sink/Outlier conditions
    # bits_expanded: [H, D] or [H, 1], bits to use if NOT sink
    # final_bits_tensor: [H, C_len, D] or [H, C_len, 1]
    final_bits = bits_expanded.unsqueeze(1).repeat(1, chunk_tensor.shape[2], 1) # Repeat bits for chunk_len
    # Override bits for sink tokens
    final_bits[:, is_sink_token, :] = special_bits

    # Perform quantization with dynamic clipping based on final_bits
    qmax = (2**(final_bits.float() - 1)) - 1
    qmin = -(2**(final_bits.float() - 1))
    
    # Scale shape needs adjustment for broadcasting with chunk_tensor [B, H, C_len, D]
    if scale.shape[-1] == 1: # Per-tensor (per head) scale [H, 1] -> [1, H, 1, 1]
        scale_bc = scale.unsqueeze(0).unsqueeze(2)
    else: # Per-channel scale [H, D] -> [1, H, 1, D]
        scale_bc = scale.unsqueeze(0).unsqueeze(2)
    
    quantized_values = torch.clamp(torch.round(chunk_tensor / scale_bc), qmin, qmax)

    # Determine storage type (use smallest common type, e.g., int8 if max bits <= 8)
    max_bits = max(config.key_bits_normal, config.key_bits_sink_outlier,
                   config.value_bits_normal, config.value_bits_sink_outlier)
    dtype = torch.int8 if max_bits <= 8 else torch.int16

    quantized_chunk = quantized_values.to(dtype)

    # Return scale used for this chunk (needed for potential dequantization)
    # Return the expanded scale matching the feature dim [H, D] or [H, 1]
    return quantized_chunk, scale # scale shape: [H, D] or [H, 1]