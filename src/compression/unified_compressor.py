"""
Unified compressor interface for managing compression state.
"""

import torch
from typing import Dict, Optional
from .streaming_quantization import StreamingStatisticsManager


class UnifiedCompressor:
    """Unified interface for managing compression statistics and state.

    This class provides a centralized way to manage statistics managers
    for all layers and reset compression state.
    """

    def __init__(self, compression_config):
        """Initialize unified compressor.

        Args:
            compression_config: CompressionConfig instance
        """
        self.config = compression_config
        self.stats_managers = {}  # Will be populated by model layers

    def register_stats_manager(
        self,
        layer_idx: int,
        kv_type: str,
        stats_manager: StreamingStatisticsManager
    ):
        """Register a statistics manager for a specific layer and KV type.

        Args:
            layer_idx: Layer index
            kv_type: Either 'key' or 'value'
            stats_manager: StreamingStatisticsManager instance
        """
        key = f"layer_{layer_idx}_{kv_type}"
        self.stats_managers[key] = stats_manager

    def reset_compression_state(self):
        """Reset all compression statistics."""
        for stats_manager in self.stats_managers.values():
            stats_manager.reset()

    def get_overall_compression_stats(self) -> Dict:
        """Get aggregated compression statistics across all layers.

        Returns:
            Dictionary containing overall compression statistics
        """
        if not self.stats_managers:
            return {}

        total_outliers = 0
        total_channels = 0
        all_stats = []

        for key, stats_manager in self.stats_managers.items():
            stats = stats_manager.get_statistics()
            stats["manager_key"] = key
            all_stats.append(stats)

            total_outliers += stats["outlier_count"]
            total_channels += stats_manager.num_channels

        # Calculate average bits based on configuration
        key_bits_avg = self._estimate_average_bits(
            self.config.key_bits_normal,
            self.config.key_bits_sink_outlier,
            total_outliers,
            total_channels,
        )

        value_bits_avg = self._estimate_average_bits(
            self.config.value_bits_normal,
            self.config.value_bits_sink_outlier,
            total_outliers,
            total_channels,
        )

        return {
            "total_outliers": total_outliers,
            "total_channels": total_channels,
            "outlier_ratio": total_outliers / max(total_channels, 1),
            "avg_key_bits": key_bits_avg,
            "avg_value_bits": value_bits_avg,
            "compression_ratio_key": 16.0 / key_bits_avg,  # Assuming FP16 baseline
            "compression_ratio_value": 16.0 / value_bits_avg,
            "layer_stats": all_stats,
        }

    def _estimate_average_bits(
        self,
        bits_normal: int,
        bits_high: int,
        num_outliers: int,
        total_channels: int,
    ) -> float:
        """Estimate average bits per element.

        Args:
            bits_normal: Normal bit width
            bits_high: High precision bit width
            num_outliers: Number of outlier channels
            total_channels: Total number of channels

        Returns:
            Estimated average bits
        """
        if total_channels == 0:
            return bits_normal

        outlier_ratio = num_outliers / total_channels
        avg_bits = (
            outlier_ratio * bits_high + 
            (1 - outlier_ratio) * bits_normal
        )

        return avg_bits
