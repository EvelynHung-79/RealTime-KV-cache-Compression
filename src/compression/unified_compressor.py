import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
import time
import psutil
import os

# Import new quantization components
from .streaming_quantization import StreamingStatisticsManager, RealTimeQuantizer, quantize_chunk, calculate_scale, dequantize_symmetric
from configs.base_config import CompressionConfig

class RealTimePrefillCompressor:
    """
    Unified Real-time Prefill KV Cache Compression System using Streaming KVQuant.
    Manages the statistics manager. Quantization logic is now mainly in the Attention layer.
    """

    def __init__(self, config: CompressionConfig, model_config=None): # Pass full model config if needed
        self.config = config
        self.model_config = model_config # Store HuggingFace model config if passed

        # Initialize core component: Statistics Manager
        # Pass the CompressionConfig to the manager
        self.stats_manager = StreamingStatisticsManager(config)

        # Compression state (simplified)
        self.layer_quant_stats = {} # Store stats per layer if needed for analysis

    def get_stats_manager(self) -> StreamingStatisticsManager:
        """Provide access to the statistics manager."""
        return self.stats_manager

    def reset_compression_state(self):
        """Reset streaming statistics for a new sequence."""
        if hasattr(self, 'stats_manager'):
            self.stats_manager.reset()
        self.layer_quant_stats = {}
        print("Streaming statistics reset.")

    def update_layer_analysis_stats(self, layer_idx, stats):
        """Optionally store per-layer stats during forward pass for later analysis."""
        self.layer_quant_stats[layer_idx] = stats

    def get_overall_compression_stats(self) -> Dict:
        """Get overall statistics about the quantization process."""
        # TODO: Implement reporting based on stats_manager state
        # e.g., percentage of outlier channels detected, avg bits used (if tracked)
        # This no longer calculates compression ratio as token count is unchanged.
        # Focus on reporting estimated memory based on bit counts.

        total_params_k = self.config.num_hidden_layers * self.config.num_key_value_heads * (self.config.hidden_size // self.config.num_attention_heads)
        total_params_v = total_params_k # Assuming same size

        # Rough estimation - needs sequence length context
        # A better approach is to track actual bits used per token/channel in stats_manager if needed
        avg_bits_k = (config.key_bits_normal + config.key_bits_sink_outlier) / 2 # Very rough
        avg_bits_v = (config.value_bits_normal + config.value_bits_sink_outlier) / 2 # Very rough

        original_bits = 16 # Assuming FP16 baseline

        estimated_ratio = (avg_bits_k * total_params_k + avg_bits_v * total_params_v) / (original_bits * (total_params_k + total_params_v))
        estimated_savings = 1.0 - estimated_ratio

        stats = {
            'num_layers_processed': self.config.num_hidden_layers, # Assuming all layers processed
            'estimated_avg_bits_key': avg_bits_k,
            'estimated_avg_bits_value': avg_bits_v,
            'estimated_memory_ratio': estimated_ratio,
            'estimated_memory_savings': estimated_savings,
            'layer_specific_stats_collected': len(self.layer_quant_stats) > 0,
            # Add counts of outliers if tracked in stats_manager
        }
        # You could iterate through self.layer_quant_stats for more detailed averages
        return stats

    def estimate_memory_usage(self) -> Dict[str, float]:
        """Estimate current memory usage (remains the same)."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
            'gpu_memory_cached_mb': torch.cuda.memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0
        }

# Removed CompressionHook as integration is now deeper within Attention layer