"""Test compression integration."""

import torch
import pytest
from configs.base_config import CompressionConfig
from src.compression.unified_compressor import UnifiedCompressor
from src.compression.streaming_quantization import StreamingStatisticsManager


def test_compression_config():
    """Test CompressionConfig validation."""
    # Valid config
    config = CompressionConfig(
        chunk_size=512,
        ema_decay=0.9,
        key_bits_normal=4,
        value_bits_normal=4,
    )
    assert config.chunk_size == 512

    # Invalid chunk size
    with pytest.raises(ValueError):
        CompressionConfig(chunk_size=-1)

    # Invalid EMA decay
    with pytest.raises(ValueError):
        CompressionConfig(ema_decay=1.5)


def test_unified_compressor():
    """Test UnifiedCompressor functionality."""
    config = CompressionConfig()
    compressor = UnifiedCompressor(config)

    # Register some stats managers
    for layer_idx in range(3):
        for kv_type in ["key", "value"]:
            manager = StreamingStatisticsManager(num_channels=128)
            compressor.register_stats_manager(layer_idx, kv_type, manager)

    # Check registration
    assert len(compressor.stats_managers) == 6  # 3 layers * 2 types

    # Get stats
    stats = compressor.get_overall_compression_stats()
    assert "avg_key_bits" in stats
    assert "avg_value_bits" in stats


def test_compression_stats_calculation():
    """Test compression statistics calculation."""
    config = CompressionConfig(
        key_bits_normal=4,
        value_bits_normal=4,
    )
    compressor = UnifiedCompressor(config)

    # Add manager and update statistics
    manager = StreamingStatisticsManager(num_channels=128)
    compressor.register_stats_manager(0, "key", manager)

    # Update with some data
    tensor = torch.randn(16, 128)
    manager.update(tensor)

    # Get stats
    stats = compressor.get_overall_compression_stats()
    assert stats["compression_ratio_key"] > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
