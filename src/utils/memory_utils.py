import torch
import psutil
import threading
import time
from typing import Dict, List
import numpy as np

class MemoryMonitor:
    """Monitor memory usage during experiments"""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.memory_history = []
        self.gpu_memory_history = []
        self.monitor_thread = None

    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.memory_history = []
            self.gpu_memory_history = []
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            # CPU memory
            process = psutil.Process()
            cpu_memory = process.memory_info().rss / (1024 * 1024)  # MB
            self.memory_history.append({
                'timestamp': time.time(),
                'cpu_memory_mb': cpu_memory
            })

            # GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
                self.gpu_memory_history.append({
                    'timestamp': time.time(),
                    'gpu_allocated_mb': gpu_memory,
                    'gpu_reserved_mb': gpu_reserved
                })

            time.sleep(self.interval)

    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage statistics"""
        if not self.memory_history:
            return {}

        cpu_memories = [m['cpu_memory_mb'] for m in self.memory_history]

        stats = {
            'peak_cpu_memory_mb': max(cpu_memories),
            'avg_cpu_memory_mb': np.mean(cpu_memories),
            'min_cpu_memory_mb': min(cpu_memories)
        }

        if self.gpu_memory_history:
            gpu_allocated = [m['gpu_allocated_mb'] for m in self.gpu_memory_history]
            gpu_reserved = [m['gpu_reserved_mb'] for m in self.gpu_memory_history]

            stats.update({
                'peak_gpu_allocated_mb': max(gpu_allocated),
                'peak_gpu_reserved_mb': max(gpu_reserved),
                'avg_gpu_allocated_mb': np.mean(gpu_allocated),
                'avg_gpu_reserved_mb': np.mean(gpu_reserved)
            })

        return stats

def get_model_memory_footprint(model) -> Dict[str, float]:
    """Calculate model memory footprint"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    return {
        'parameters_mb': param_size / (1024 * 1024),
        'buffers_mb': buffer_size / (1024 * 1024),
        'total_model_mb': (param_size + buffer_size) / (1024 * 1024)
    }

def estimate_kv_cache_memory(
    batch_size: int,
    seq_len: int, 
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    precision_bytes: int = 2  # fp16
) -> float:
    """Estimate KV cache memory usage"""
    head_dim = hidden_size // num_heads
    kv_cache_size = 2 * batch_size * seq_len * num_layers * hidden_size * precision_bytes
    return kv_cache_size / (1024 * 1024)  # MB

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()