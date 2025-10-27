"""
Streaming quantization module for real-time KV cache compression.

This module implements:
1. StreamingStatisticsManager: Manages EMA-based statistics and outlier detection
2. Symmetric quantization/dequantization functions
3. Chunk-based quantization with Sink and Outlier awareness
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class StreamingStatisticsManager:
    """Manages streaming statistics for per-channel/group-wise quantization.

    This class maintains exponential moving average (EMA) of absolute maximum values
    for each channel/group and implements outlier detection logic.

    Args:
        num_channels: Number of channels to track
        ema_decay: Decay factor for EMA (default: 0.9)
        outlier_threshold_relative: Relative threshold for outlier detection (default: 3.0)
        outlier_threshold_abs: Absolute threshold for outlier detection (optional)
        device: Device to store statistics on
    """

    def __init__(
        self,
        num_channels: int,
        ema_decay: float = 0.9,
        outlier_threshold_relative: float = 3.0,
        outlier_threshold_abs: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_channels = num_channels
        self.ema_decay = ema_decay
        self.outlier_threshold_relative = outlier_threshold_relative
        self.outlier_threshold_abs = outlier_threshold_abs
        self.device = device

        # Initialize EMA absmax statistics
        self.ema_absmax = torch.zeros(num_channels, device=device)
        self.is_initialized = False

        # Track outlier status
        self.outlier_mask = torch.zeros(num_channels, dtype=torch.bool, device=device)

        # Statistics for monitoring
        self.update_count = 0
        self.outlier_history = []

    def update(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update statistics with new tensor and detect outliers.

        Args:
            tensor: Input tensor of shape [batch, seq_len, num_channels] or [seq_len, num_channels]

        Returns:
            Tuple of (ema_absmax, outlier_mask)
        """
        # Compute absmax for this chunk across all dimensions except channels
        if tensor.dim() == 3:
            # [batch, seq_len, num_channels]
            chunk_absmax = tensor.abs().amax(dim=(0, 1))
        elif tensor.dim() == 2:
            # [seq_len, num_channels]
            chunk_absmax = tensor.abs().amax(dim=0)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {tensor.dim()}D")

        # Initialize or update EMA
        if not self.is_initialized:
            self.ema_absmax = chunk_absmax.clone()
            self.is_initialized = True
        else:
            self.ema_absmax = (
                self.ema_decay * self.ema_absmax + 
                (1 - self.ema_decay) * chunk_absmax
            )

        # Detect outliers
        self.outlier_mask = self._detect_outliers(chunk_absmax)

        # Update statistics
        self.update_count += 1
        self.outlier_history.append(self.outlier_mask.sum().item())

        return self.ema_absmax, self.outlier_mask

    def _detect_outliers(self, chunk_absmax: torch.Tensor) -> torch.Tensor:
        """Detect outlier channels based on thresholds.

        Args:
            chunk_absmax: Absolute maximum values for current chunk

        Returns:
            Boolean tensor indicating outlier channels
        """
        outlier_mask = torch.zeros_like(chunk_absmax, dtype=torch.bool)

        # Relative threshold: current value >> historical EMA
        if self.outlier_threshold_relative is not None and self.is_initialized:
            relative_outliers = (
                chunk_absmax > self.outlier_threshold_relative * self.ema_absmax
            )
            outlier_mask |= relative_outliers

        # Absolute threshold: current value > fixed threshold
        if self.outlier_threshold_abs is not None:
            absolute_outliers = chunk_absmax > self.outlier_threshold_abs
            outlier_mask |= absolute_outliers

        return outlier_mask

    def get_scale(self, bits: int) -> torch.Tensor:
        """Calculate quantization scale based on current EMA statistics.

        Args:
            bits: Number of bits for quantization

        Returns:
            Scale tensor for each channel
        """
        # Avoid division by zero
        safe_absmax = torch.clamp(self.ema_absmax, min=1e-8)

        # Calculate scale: absmax / (2^(bits-1) - 1)
        max_int = 2 ** (bits - 1) - 1
        scale = safe_absmax / max_int

        return scale

    def reset(self):
        """Reset all statistics."""
        self.ema_absmax.zero_()
        self.is_initialized = False
        self.outlier_mask.zero_()
        self.update_count = 0
        self.outlier_history = []

    def get_statistics(self) -> Dict:
        """Get current statistics for monitoring."""
        return {
            "ema_absmax_mean": self.ema_absmax.mean().item(),
            "ema_absmax_max": self.ema_absmax.max().item(),
            "outlier_count": self.outlier_mask.sum().item(),
            "outlier_ratio": self.outlier_mask.float().mean().item(),
            "update_count": self.update_count,
        }


def calculate_scale(absmax: torch.Tensor, bits: int) -> torch.Tensor:
    """Calculate quantization scale from absolute maximum values.

    Args:
        absmax: Absolute maximum values
        bits: Number of quantization bits

    Returns:
        Quantization scale
    """
    safe_absmax = torch.clamp(absmax, min=1e-8)
    max_int = 2 ** (bits - 1) - 1
    scale = safe_absmax / max_int
    return scale


def quantize_symmetric(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Symmetric quantization with per-channel scaling.

    Args:
        tensor: Input tensor to quantize
        scale: Per-channel scale values
        bits: Number of quantization bits

    Returns:
        Quantized tensor (still in float format for compatibility)
    """
    # Ensure scale has correct shape for broadcasting
    if scale.dim() == 1 and tensor.dim() >= 2:
        # Reshape scale to broadcast correctly
        scale = scale.view(1, -1) if tensor.dim() == 2 else scale.view(1, 1, -1)

    # Quantize
    max_int = 2 ** (bits - 1) - 1
    min_int = -(2 ** (bits - 1))

    quantized = torch.clamp(
        torch.round(tensor / scale),
        min=min_int,
        max=max_int
    )

    return quantized


def dequantize_symmetric(
    quantized_tensor: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize tensor back to floating point.

    Args:
        quantized_tensor: Quantized tensor
        scale: Per-channel scale values

    Returns:
        Dequantized tensor
    """
    # Ensure scale has correct shape for broadcasting
    if scale.dim() == 1 and quantized_tensor.dim() >= 2:
        scale = scale.view(1, -1) if quantized_tensor.dim() == 2 else scale.view(1, 1, -1)

    return quantized_tensor * scale


def quantize_chunk(
    tensor: torch.Tensor,
    stats_manager: StreamingStatisticsManager,
    bits_normal: int,
    bits_high: int,
    is_sink_chunk: bool = False,
    sink_size: int = 0,
    enable_outlier: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """Quantize a chunk with Sink and Outlier awareness.

    This function:
    1. Updates streaming statistics
    2. Detects outliers
    3. Applies mixed-precision quantization based on token position and outlier status

    Args:
        tensor: Input tensor [seq_len, num_channels] or [batch, seq_len, num_channels]
        stats_manager: Statistics manager for this layer/key-value
        bits_normal: Bit width for normal tokens
        bits_high: Bit width for sink/outlier tokens
        is_sink_chunk: Whether this chunk contains sink tokens
        sink_size: Number of sink tokens in this chunk
        enable_outlier: Whether to enable outlier detection

    Returns:
        Tuple of (quantized_tensor, info_dict)
    """
    # Update statistics and detect outliers
    ema_absmax, outlier_mask = stats_manager.update(tensor)

    # Initialize quantized tensor
    quantized = torch.zeros_like(tensor)

    # Get scales for different bit widths
    scale_normal = stats_manager.get_scale(bits_normal)
    scale_high = stats_manager.get_scale(bits_high)

    # Track which tokens/channels use which precision
    info = {
        "sink_tokens": 0,
        "outlier_channels": outlier_mask.sum().item(),
        "normal_tokens": tensor.shape[-2] if tensor.dim() >= 2 else 0,
    }

    # Handle different input dimensions
    if tensor.dim() == 3:
        batch_size, seq_len, num_channels = tensor.shape
    elif tensor.dim() == 2:
        seq_len, num_channels = tensor.shape
        batch_size = 1
        tensor = tensor.unsqueeze(0)
        quantized = quantized.unsqueeze(0)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {tensor.dim()}D")

    # Process sink tokens with high precision
    if is_sink_chunk and sink_size > 0:
        sink_tokens = tensor[:, :sink_size, :]
        quantized[:, :sink_size, :] = quantize_symmetric(
            sink_tokens, scale_high, bits_high
        )
        info["sink_tokens"] = sink_size
        start_idx = sink_size
    else:
        start_idx = 0

    # Process remaining tokens
    if start_idx < seq_len:
        remaining_tokens = tensor[:, start_idx:, :]

        if enable_outlier and outlier_mask.any():
            # Split into outlier and normal channels
            normal_mask = ~outlier_mask

            # Quantize outlier channels with high precision
            outlier_data = remaining_tokens[:, :, outlier_mask]
            quantized[:, start_idx:, outlier_mask] = quantize_symmetric(
                outlier_data, scale_high[outlier_mask], bits_high
            )

            # Quantize normal channels with normal precision
            normal_data = remaining_tokens[:, :, normal_mask]
            quantized[:, start_idx:, normal_mask] = quantize_symmetric(
                normal_data, scale_normal[normal_mask], bits_normal
            )
        else:
            # All remaining tokens use normal precision
            quantized[:, start_idx:, :] = quantize_symmetric(
                remaining_tokens, scale_normal, bits_normal
            )

    # Remove batch dimension if it was added
    if batch_size == 1 and tensor.dim() == 3:
        quantized = quantized.squeeze(0)

    # Calculate average bits used
    total_elements = seq_len * num_channels
    high_precision_elements = (
        sink_size * num_channels + 
        (seq_len - sink_size) * outlier_mask.sum().item() if enable_outlier else 0
    )
    normal_precision_elements = total_elements - high_precision_elements

    avg_bits = (
        (high_precision_elements * bits_high + normal_precision_elements * bits_normal) /
        total_elements
    )
    info["avg_bits"] = avg_bits

    return quantized, info
