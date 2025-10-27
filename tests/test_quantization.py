"""Test streaming quantization module."""

import torch
import pytest
from src.compression.streaming_quantization import (
    StreamingStatisticsManager,
    calculate_scale,
    quantize_symmetric,
    dequantize_symmetric,
    quantize_chunk,
)


def test_streaming_statistics_manager():
    """Test StreamingStatisticsManager basic functionality."""
    num_channels = 128
    manager = StreamingStatisticsManager(num_channels, ema_decay=0.9)

    # Create test tensor
    tensor = torch.randn(16, num_channels)

    # Update statistics
    ema_absmax, outlier_mask = manager.update(tensor)

    assert ema_absmax.shape == (num_channels,)
    assert outlier_mask.shape == (num_channels,)
    assert manager.is_initialized
    assert manager.update_count == 1


def test_outlier_detection():
    """Test outlier detection logic."""
    num_channels = 10
    manager = StreamingStatisticsManager(
        num_channels, 
        ema_decay=0.9,
        outlier_threshold_relative=3.0
    )

    # Normal values
    normal_tensor = torch.randn(8, num_channels)
    manager.update(normal_tensor)

    # Create tensor with outliers in some channels
    outlier_tensor = torch.randn(8, num_channels)
    outlier_tensor[:, 0] *= 10  # Create outlier in channel 0

    _, outlier_mask = manager.update(outlier_tensor)

    # Channel 0 should be detected as outlier
    assert outlier_mask[0].item() == True


def test_quantize_dequantize():
    """Test quantization and dequantization."""
    tensor = torch.randn(16, 128)

    # Calculate scale
    absmax = tensor.abs().max(dim=0)[0]
    scale = calculate_scale(absmax, bits=4)

    # Quantize
    quantized = quantize_symmetric(tensor, scale, bits=4)

    # Dequantize
    dequantized = dequantize_symmetric(quantized, scale)

    # Check shape preservation
    assert dequantized.shape == tensor.shape

    # Check quantization error is bounded
    max_error = (tensor - dequantized).abs().max()
    assert max_error < scale.max() * 2  # Error should be within 2 * scale


def test_quantize_chunk():
    """Test chunk-based quantization with Sink and Outlier awareness."""
    num_channels = 128
    seq_len = 16

    manager = StreamingStatisticsManager(num_channels, ema_decay=0.9)
    tensor = torch.randn(seq_len, num_channels)

    # Test with sink tokens
    quantized, info = quantize_chunk(
        tensor,
        manager,
        bits_normal=4,
        bits_high=8,
        is_sink_chunk=True,
        sink_size=4,
        enable_outlier=True,
    )

    assert quantized.shape == tensor.shape
    assert info["sink_tokens"] == 4
    assert "avg_bits" in info
    assert 4 <= info["avg_bits"] <= 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
