"""Metrics for evaluating compression performance."""

import torch
import torch.nn as nn
from typing import Dict
import numpy as np

class CompressionMetrics:
    """Track compression-related metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.total_key_bits = 0
        self.total_value_bits = 0
        self.total_elements = 0
        self.outlier_counts = []
        self.perplexities = []

    def update(self, compression_stats: Dict):
        """Update metrics with compression statistics."""
        if 'avg_key_bits' in compression_stats:
            self.total_key_bits += compression_stats['avg_key_bits']
        if 'avg_value_bits' in compression_stats:
            self.total_value_bits += compression_stats['avg_value_bits']
        if 'total_outliers' in compression_stats:
            self.outlier_counts.append(compression_stats['total_outliers'])
        self.total_elements += 1

    def compute(self) -> Dict:
        """Compute final metrics."""
        if self.total_elements == 0:
            return {}

        avg_key_bits = self.total_key_bits / self.total_elements
        avg_value_bits = self.total_value_bits / self.total_elements

        # Compression ratios (assuming FP16 baseline)
        key_compression_ratio = 16.0 / avg_key_bits if avg_key_bits > 0 else 1.0
        value_compression_ratio = 16.0 / avg_value_bits if avg_value_bits > 0 else 1.0
        overall_compression_ratio = (key_compression_ratio + value_compression_ratio) / 2

        # Memory savings
        key_memory_saving = (1 - 1/key_compression_ratio) * 100
        value_memory_saving = (1 - 1/value_compression_ratio) * 100
        overall_memory_saving = (key_memory_saving + value_memory_saving) / 2

        return {
            'avg_key_bits': avg_key_bits,
            'avg_value_bits': avg_value_bits,
            'key_compression_ratio': key_compression_ratio,
            'value_compression_ratio': value_compression_ratio,
            'overall_compression_ratio': overall_compression_ratio,
            'key_memory_saving_pct': key_memory_saving,
            'value_memory_saving_pct': value_memory_saving,
            'overall_memory_saving_pct': overall_memory_saving,
            'avg_outlier_count': np.mean(self.outlier_counts) if self.outlier_counts else 0,
            'perplexity': np.mean(self.perplexities) if self.perplexities else None,
        }

def calculate_perplexity(model, tokenizer, texts, device="cuda"):
    """Calculate perplexity on a set of texts.

    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of text strings
        device: Device to run on

    Returns:
        Average perplexity
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = encodings.input_ids.to(device)

            # Forward pass
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

            # Reset compression state between texts
            if hasattr(model, 'reset_compression_state'):
                model.reset_compression_state()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity
