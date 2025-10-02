import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np

class DynamicPrecisionQuantizer:
    """
    Dynamic precision assignment and quantization based on token importance scores
    Implements multi-bit quantization: 8-bit (HIGH), 4-bit (MID), 2-bit (LOW)
    """

    def __init__(self, config):
        self.config = config
        self.theta_h = config.theta_h  # High precision threshold
        self.theta_m = config.theta_m  # Medium precision threshold
        self.high_bits = config.high_precision_bits
        self.medium_bits = config.medium_precision_bits
        self.low_bits = config.low_precision_bits

    def assign_precision_levels(
        self, 
        importance_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Assign precision levels based on importance scores

        Args:
            importance_scores: [batch, seq_len] importance scores

        Returns:
            precision_labels: [batch, seq_len] precision assignments (0=LOW, 1=MID, 2=HIGH)
            precision_stats: dict with count statistics
        """
        batch_size, seq_len = importance_scores.shape

        # Initialize precision labels (0=LOW by default)
        precision_labels = torch.zeros_like(importance_scores, dtype=torch.long)

        # Assign precision levels based on thresholds
        high_mask = importance_scores >= self.theta_h
        medium_mask = (importance_scores >= self.theta_m) & (importance_scores < self.theta_h)

        precision_labels[high_mask] = 2    # HIGH precision
        precision_labels[medium_mask] = 1  # MEDIUM precision
        # LOW precision tokens remain 0

        # Calculate statistics
        total_tokens = batch_size * seq_len
        precision_stats = {
            'high_count': high_mask.sum().item(),
            'medium_count': medium_mask.sum().item(), 
            'low_count': total_tokens - high_mask.sum().item() - medium_mask.sum().item(),
            'high_ratio': high_mask.sum().item() / total_tokens,
            'medium_ratio': medium_mask.sum().item() / total_tokens,
            'low_ratio': (total_tokens - high_mask.sum().item() - medium_mask.sum().item()) / total_tokens
        }

        return precision_labels, precision_stats

    def get_quantization_params(
        self, 
        tensor: torch.Tensor, 
        num_bits: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute quantization parameters (scale and zero_point) for uniform quantization

        Args:
            tensor: input tensor to quantize
            num_bits: target bit width

        Returns:
            scale: quantization scale factor
            zero_point: quantization zero point
        """
        # Compute min/max along feature dimension
        t_min = tensor.min()
        t_max = tensor.max()

        # Avoid division by zero
        if t_max == t_min:
            scale = torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype)
            zero_point = torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
            return scale, zero_point

        # Compute quantization parameters
        qmin = 0
        qmax = (2 ** num_bits) - 1

        scale = (t_max - t_min) / (qmax - qmin)
        zero_point = qmin - t_min / scale

        return scale, zero_point

    def quantize_tensor(
        self, 
        tensor: torch.Tensor, 
        num_bits: int,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply uniform quantization to tensor

        Args:
            tensor: input tensor
            num_bits: quantization bit width
            scale: quantization scale
            zero_point: quantization zero point

        Returns:
            quantized_tensor: quantized tensor
        """
        qmin = 0
        qmax = (2 ** num_bits) - 1

        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)

        # Dequantize back to original range
        dequantized = (quantized - zero_point) * scale

        return dequantized

    def apply_mixed_precision_quantization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor, 
        precision_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Apply mixed precision quantization based on precision labels

        Args:
            key_states: [batch, seq_len, head_dim] key tensor
            value_states: [batch, seq_len, head_dim] value tensor  
            precision_labels: [batch, seq_len] precision assignments

        Returns:
            quantized_keys: quantized key states
            quantized_values: quantized value states
            quant_info: quantization information
        """
        batch_size, seq_len, head_dim = key_states.shape
        device = key_states.device

        # Initialize output tensors
        quantized_keys = torch.zeros_like(key_states)
        quantized_values = torch.zeros_like(value_states)

        # Track quantization info
        quant_info = {
            'scales': {},
            'zero_points': {},
            'bit_assignments': precision_labels.detach().cpu().numpy()
        }

        # Process each precision level separately
        for precision_level in [0, 1, 2]:  # LOW, MID, HIGH
            mask = (precision_labels == precision_level)

            if not mask.any():
                continue

            # Get corresponding bit width
            if precision_level == 0:    # LOW
                num_bits = self.low_bits
            elif precision_level == 1:  # MID  
                num_bits = self.medium_bits
            else:                       # HIGH
                num_bits = self.high_bits

            # Extract tokens with this precision level
            if mask.any():
                # For each token position with this precision
                for b in range(batch_size):
                    for s in range(seq_len):
                        if mask[b, s]:
                            # Quantize key
                            k_token = key_states[b:b+1, s:s+1, :]
                            k_scale, k_zp = self.get_quantization_params(k_token, num_bits)
                            quantized_keys[b, s, :] = self.quantize_tensor(
                                key_states[b, s, :], num_bits, k_scale, k_zp
                            )

                            # Quantize value  
                            v_token = value_states[b:b+1, s:s+1, :]
                            v_scale, v_zp = self.get_quantization_params(v_token, num_bits)
                            quantized_values[b, s, :] = self.quantize_tensor(
                                value_states[b, s, :], num_bits, v_scale, v_zp
                            )

        return quantized_keys, quantized_values, quant_info

    def estimate_memory_savings(
        self, 
        original_tensor: torch.Tensor,
        precision_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Estimate memory savings from mixed precision quantization

        Args:
            original_tensor: original tensor (assumed fp16)
            precision_labels: precision assignments

        Returns:
            memory_info: dictionary with memory statistics
        """
        total_elements = original_tensor.numel()

        # Count elements by precision
        high_elements = (precision_labels == 2).sum().item() * original_tensor.shape[-1]
        medium_elements = (precision_labels == 1).sum().item() * original_tensor.shape[-1]
        low_elements = (precision_labels == 0).sum().item() * original_tensor.shape[-1]

        # Calculate memory usage (in bytes)
        # Original: 16 bits = 2 bytes per element
        original_memory = total_elements * 2

        compressed_memory = (
            high_elements * (self.high_bits / 8) +
            medium_elements * (self.medium_bits / 8) + 
            low_elements * (self.low_bits / 8)
        )

        compression_ratio = compressed_memory / original_memory
        memory_savings = 1.0 - compression_ratio

        return {
            'original_memory_mb': original_memory / (1024 * 1024),
            'compressed_memory_mb': compressed_memory / (1024 * 1024), 
            'compression_ratio': compression_ratio,
            'memory_savings': memory_savings,
            'high_elements_ratio': high_elements / total_elements,
            'medium_elements_ratio': medium_elements / total_elements,
            'low_elements_ratio': low_elements / total_elements
        }