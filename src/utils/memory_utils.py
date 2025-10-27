"""Memory monitoring utilities."""

import torch
import psutil
from typing import Dict

def monitor_memory() -> Dict[str, float]:
    """Monitor current memory usage."""
    stats = {}

    # CPU memory
    process = psutil.Process()
    stats['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024

    # GPU memory
    if torch.cuda.is_available():
        stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        stats['gpu_memory_free_mb'] = (torch.cuda.get_device_properties(0).total_memory - 
                                       torch.cuda.memory_allocated()) / 1024 / 1024

    return stats

def get_memory_stats() -> str:
    """Get formatted memory statistics."""
    stats = monitor_memory()
    output = f"CPU Memory: {stats['cpu_memory_mb']:.2f} MB\n"
    if 'gpu_memory_allocated_mb' in stats:
        output += f"GPU Memory Allocated: {stats['gpu_memory_allocated_mb']:.2f} MB\n"
        output += f"GPU Memory Reserved: {stats['gpu_memory_reserved_mb']:.2f} MB\n"
        output += f"GPU Memory Free: {stats['gpu_memory_free_mb']:.2f} MB"
    return output
