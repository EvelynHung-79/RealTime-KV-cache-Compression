import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

class CompressionMetrics:
    """Calculate and track compression-related metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.layer_stats = []
        self.total_original_memory = 0
        self.total_compressed_memory = 0
        self.total_processing_time = 0

    def update_layer_stats(
        self,
        layer_idx: int,
        original_size: int,
        compressed_size: int,
        processing_time: float,
        importance_scores: torch.Tensor,
        precision_distribution: Dict[str, int]
    ):
        """Update statistics for a single layer"""

        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
        memory_savings = 1.0 - compression_ratio

        layer_stat = {
            'layer_idx': layer_idx,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'memory_savings': memory_savings,
            'processing_time': processing_time,
            'avg_importance': importance_scores.mean().item(),
            'importance_std': importance_scores.std().item(),
            'precision_distribution': precision_distribution
        }

        self.layer_stats.append(layer_stat)
        self.total_original_memory += original_size
        self.total_compressed_memory += compressed_size
        self.total_processing_time += processing_time

    def get_overall_metrics(self) -> Dict[str, float]:
        """Calculate overall compression metrics"""

        if not self.layer_stats:
            return {}

        # Overall compression
        overall_compression_ratio = (
            self.total_compressed_memory / self.total_original_memory 
            if self.total_original_memory > 0 else 1.0
        )
        overall_memory_savings = 1.0 - overall_compression_ratio

        # Layer-wise averages
        avg_layer_compression = np.mean([s['compression_ratio'] for s in self.layer_stats])
        avg_layer_savings = np.mean([s['memory_savings'] for s in self.layer_stats])
        avg_processing_time = np.mean([s['processing_time'] for s in self.layer_stats])

        # Importance score statistics
        avg_importance = np.mean([s['avg_importance'] for s in self.layer_stats])
        avg_importance_std = np.mean([s['importance_std'] for s in self.layer_stats])

        # Precision distribution across all layers
        total_high = sum(s['precision_distribution'].get('high', 0) for s in self.layer_stats)
        total_medium = sum(s['precision_distribution'].get('medium', 0) for s in self.layer_stats)
        total_low = sum(s['precision_distribution'].get('low', 0) for s in self.layer_stats)
        total_tokens = total_high + total_medium + total_low

        precision_ratios = {
            'high_ratio': total_high / total_tokens if total_tokens > 0 else 0,
            'medium_ratio': total_medium / total_tokens if total_tokens > 0 else 0,
            'low_ratio': total_low / total_tokens if total_tokens > 0 else 0
        }

        return {
            'overall_compression_ratio': overall_compression_ratio,
            'overall_memory_savings': overall_memory_savings,
            'avg_layer_compression_ratio': avg_layer_compression,
            'avg_layer_memory_savings': avg_layer_savings,
            'total_processing_time': self.total_processing_time,
            'avg_layer_processing_time': avg_processing_time,
            'avg_importance_score': avg_importance,
            'avg_importance_std': avg_importance_std,
            'num_layers_processed': len(self.layer_stats),
            **precision_ratios
        }

    def get_layer_wise_breakdown(self) -> List[Dict]:
        """Get detailed layer-wise breakdown"""
        return self.layer_stats.copy()

class PerformanceTimer:
    """Timer utility for measuring performance"""

    def __init__(self):
        self.start_times = {}
        self.durations = {}

    def start(self, name: str):
        """Start timing an operation"""
        self.start_times[name] = time.time()

    def stop(self, name: str) -> float:
        """Stop timing and return duration"""
        if name not in self.start_times:
            return 0.0

        duration = time.time() - self.start_times[name]
        self.durations[name] = duration
        del self.start_times[name]
        return duration

    def get_duration(self, name: str) -> float:
        """Get recorded duration"""
        return self.durations.get(name, 0.0)

    def get_all_durations(self) -> Dict[str, float]:
        """Get all recorded durations"""
        return self.durations.copy()

def calculate_throughput(
    num_tokens: int, 
    total_time: float,
    batch_size: int = 1
) -> Dict[str, float]:
    """Calculate throughput metrics"""

    if total_time <= 0:
        return {'tokens_per_second': 0, 'samples_per_second': 0}

    tokens_per_second = num_tokens / total_time
    samples_per_second = batch_size / total_time

    return {
        'tokens_per_second': tokens_per_second,
        'samples_per_second': samples_per_second,
        'time_per_token': total_time / num_tokens if num_tokens > 0 else 0,
        'time_per_sample': total_time / batch_size
    }

def calculate_compression_efficiency(
    original_memory: float,
    compressed_memory: float, 
    quality_score: float,
    processing_overhead: float = 0.0
) -> Dict[str, float]:
    """Calculate compression efficiency metrics"""

    compression_ratio = compressed_memory / original_memory if original_memory > 0 else 1.0
    memory_savings = 1.0 - compression_ratio

    # Quality-adjusted compression (penalize quality loss)
    quality_adjusted_savings = memory_savings * quality_score

    # Efficiency considering processing overhead
    net_efficiency = quality_adjusted_savings - processing_overhead

    return {
        'compression_ratio': compression_ratio,
        'memory_savings': memory_savings,
        'quality_score': quality_score,
        'quality_adjusted_savings': quality_adjusted_savings,
        'processing_overhead': processing_overhead,
        'net_efficiency': net_efficiency,
        'efficiency_score': net_efficiency / memory_savings if memory_savings > 0 else 0
    }