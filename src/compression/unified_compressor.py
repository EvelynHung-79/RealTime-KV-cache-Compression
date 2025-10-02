import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
import time
import psutil
import os

from .token_importance import PromptGuidedImportanceScorer, LayerWiseImportanceTracker
from .dynamic_quantization import DynamicPrecisionQuantizer
from .selective_propagation import SelectiveTokenPropagator

class RealTimePrefillCompressor:
    """
    Unified Real-time Prefill KV Cache Compression System

    Integrates three core components:
    1. Prompt-guided Token Importance Scoring
    2. Dynamic Precision Assignment & Quantization  
    3. Selective Token Propagation
    """

    def __init__(self, config, model_config=None):
        self.config = config
        self.model_config = model_config

        # Initialize core components
        self.importance_tracker = LayerWiseImportanceTracker(config)
        self.quantizer = DynamicPrecisionQuantizer(config)  
        self.propagator = SelectiveTokenPropagator(config)

        # Compression state
        self.compression_stats = {}
        self.layer_states = {}

    def identify_prompt_tokens(
        self, 
        input_ids: torch.Tensor,
        special_tokens: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Identify prompt tokens from input sequence
        Simple heuristic: assume first part is prompt, or use special tokens

        Args:
            input_ids: [batch, seq_len] input token ids
            special_tokens: list of special token ids that separate prompt

        Returns:
            prompt_indices: [prompt_len] indices of prompt tokens
        """
        seq_len = input_ids.shape[1]

        # Simple heuristic: first 20% of sequence as prompt
        # In practice, this should be determined by task-specific logic
        prompt_length = max(1, min(seq_len // 5, 128))
        prompt_indices = torch.arange(prompt_length, device=input_ids.device)

        return prompt_indices

    def extract_attention_weights(
        self, 
        attention_output: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Extract attention weights from transformer layer output
        Note: This is a placeholder - actual implementation depends on model architecture

        Args:
            attention_output: attention layer output
            layer_idx: current layer index

        Returns:
            attention_weights: [batch, heads, seq_len, seq_len] attention matrix
        """
        # This is a simplified placeholder
        # In practice, you need to modify the transformer model to return attention weights
        # or hook into the attention computation

        if hasattr(attention_output, 'attentions'):
            return attention_output.attentions
        else:
            # Fallback: create dummy attention weights for testing
            batch_size = 1  # Assuming single batch for simplicity
            num_heads = self.config.num_attention_heads
            seq_len = attention_output.shape[1] if len(attention_output.shape) > 1 else 64

            # Create random attention weights for testing
            dummy_attention = torch.randn(
                batch_size, num_heads, seq_len, seq_len,
                device=attention_output.device
            )
            return F.softmax(dummy_attention, dim=-1)

    def compress_layer_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor, 
        attention_weights: torch.Tensor,
        input_ids: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Apply real-time compression to KV cache for a single layer

        Args:
            key_states: [batch, seq_len, head_dim] key tensor
            value_states: [batch, seq_len, head_dim] value tensor
            attention_weights: [batch, heads, seq_len, seq_len] attention weights
            input_ids: [batch, seq_len] input token ids
            layer_idx: current layer index

        Returns:
            compressed_keys: compressed key states
            compressed_values: compressed value states
            compression_info: detailed compression statistics
        """
        start_time = time.time()

        # Step 1: Identify prompt tokens
        prompt_indices = self.identify_prompt_tokens(input_ids)

        # Step 2: Compute token importance scores
        importance_scores = self.importance_tracker.update_scores(
            layer_idx, attention_weights, prompt_indices
        )

        # Step 3: Assign dynamic precision levels
        precision_labels, precision_stats = self.quantizer.assign_precision_levels(
            importance_scores
        )

        # Step 4: Apply mixed precision quantization  
        quantized_keys, quantized_values, quant_info = self.quantizer.apply_mixed_precision_quantization(
            key_states, value_states, precision_labels
        )

        # Step 5: Selective token propagation
        selected_keys, selected_values, selected_scores, selected_labels, propagation_info = self.propagator.apply_token_selection(
            quantized_keys, quantized_values, importance_scores, precision_labels, layer_idx, input_ids
        )

        # Calculate compression metrics
        original_memory = key_states.numel() + value_states.numel()  # Total elements
        compressed_memory = selected_keys.numel() + selected_values.numel()
        compression_ratio = compressed_memory / original_memory if original_memory > 0 else 0

        processing_time = time.time() - start_time

        # Compile compression info
        compression_info = {
            'layer_idx': layer_idx,
            'processing_time': processing_time,
            'original_shape': key_states.shape,
            'compressed_shape': selected_keys.shape,
            'compression_ratio': compression_ratio,
            'memory_savings': 1.0 - compression_ratio,
            'importance_stats': {
                'mean_score': importance_scores.mean().item(),
                'std_score': importance_scores.std().item(),
                'min_score': importance_scores.min().item(),
                'max_score': importance_scores.max().item()
            },
            'precision_stats': precision_stats,
            'quantization_info': quant_info,
            'propagation_info': propagation_info
        }

        # Store layer state for tracking
        self.layer_states[layer_idx] = compression_info

        return selected_keys, selected_values, compression_info

    def get_overall_compression_stats(self) -> Dict:
        """Get overall compression statistics across all layers"""

        if not self.layer_states:
            return {}

        total_layers = len(self.layer_states)
        total_time = sum(state['processing_time'] for state in self.layer_states.values())
        avg_compression = sum(state['compression_ratio'] for state in self.layer_states.values()) / total_layers
        avg_memory_savings = sum(state['memory_savings'] for state in self.layer_states.values()) / total_layers

        # Aggregate precision statistics
        total_high = sum(state['precision_stats']['high_count'] for state in self.layer_states.values())
        total_medium = sum(state['precision_stats']['medium_count'] for state in self.layer_states.values()) 
        total_low = sum(state['precision_stats']['low_count'] for state in self.layer_states.values())
        total_tokens = total_high + total_medium + total_low

        # Calculate cumulative compression (multiplicative effect across layers)
        cumulative_compression = 1.0
        for layer_state in self.layer_states.values():
            if 'propagation_info' in layer_state:
                prop_info = layer_state['propagation_info']
                if 'selection_stats' in prop_info:
                    layer_ratio = prop_info.get('propagation_ratio', 1.0)
                    cumulative_compression *= layer_ratio

        return {
            'total_layers_processed': total_layers,
            'total_processing_time': total_time,
            'avg_processing_time_per_layer': total_time / total_layers,
            'avg_compression_ratio': avg_compression,
            'avg_memory_savings': avg_memory_savings,
            'cumulative_compression': cumulative_compression,
            'overall_memory_savings': 1.0 - cumulative_compression,
            'precision_distribution': {
                'high_ratio': total_high / total_tokens if total_tokens > 0 else 0,
                'medium_ratio': total_medium / total_tokens if total_tokens > 0 else 0,  
                'low_ratio': total_low / total_tokens if total_tokens > 0 else 0
            }
        }

    def reset_compression_state(self):
        """Reset compression state for new sequence"""
        self.layer_states = {}
        self.importance_tracker.layer_scores = {}

    def estimate_memory_usage(self) -> Dict[str, float]:
        """Estimate current memory usage"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size  
            'gpu_memory_mb': torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
            'gpu_memory_cached_mb': torch.cuda.memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0
        }


class CompressionHook:
    """Hook for integrating compression into transformer models"""

    def __init__(self, compressor: RealTimePrefillCompressor):
        self.compressor = compressor
        self.hooks = []

    def register_hooks(self, model):
        """Register hooks on transformer layers"""

        for layer_idx, layer in enumerate(model.model.layers):
            hook = layer.register_forward_hook(
                lambda module, input, output, l=layer_idx: self._compression_hook(
                    module, input, output, l
                )
            )
            self.hooks.append(hook)

    def _compression_hook(self, module, input, output, layer_idx):
        """Forward hook for applying compression"""

        # This is a simplified hook - actual implementation needs to:
        # 1. Extract key/value states from the layer
        # 2. Get attention weights
        # 3. Apply compression
        # 4. Replace the KV cache with compressed version

        # For now, just track that the layer was processed
        print(f"Processing layer {layer_idx} with compression")

        return output

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []