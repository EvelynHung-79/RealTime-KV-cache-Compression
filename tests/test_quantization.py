import pytest
import torch
from configs.base_config import CompressionConfig
from src.compression.dynamic_quantization import DynamicPrecisionQuantizer

class TestDynamicPrecisionQuantizer:

    @pytest.fixture
    def config(self):
        return CompressionConfig(
            model_name="meta-llama/Llama-2-7b-hf",
            theta_h=0.7,
            theta_m=0.3,
            high_precision_bits=8,
            medium_precision_bits=4,
            low_precision_bits=2
        )

    @pytest.fixture
    def quantizer(self, config):
        return DynamicPrecisionQuantizer(config)

    @pytest.fixture
    def sample_tensors(self):
        batch_size, seq_len, head_dim = 2, 12, 64
        key_states = torch.randn(batch_size, seq_len, head_dim)
        value_states = torch.randn(batch_size, seq_len, head_dim)

        # Create synthetic importance scores
        importance_scores = torch.rand(batch_size, seq_len)

        return {
            'key_states': key_states,
            'value_states': value_states,
            'importance_scores': importance_scores
        }

    def test_assign_precision_levels(self, quantizer, sample_tensors):
        precision_labels, stats = quantizer.assign_precision_levels(sample_tensors['importance_scores'])

        # Check shape and dtype
        assert precision_labels.shape == sample_tensors['importance_scores'].shape
        assert precision_labels.dtype == torch.long

        # Check stats sums match total tokens
        total_tokens = sample_tensors['importance_scores'].numel()
        calculated_total = stats['high_count'] + stats['medium_count'] + stats['low_count']
        assert calculated_total == total_tokens

        # Check ratio sums approximately 1
        ratio_sum = stats['high_ratio'] + stats['medium_ratio'] + stats['low_ratio']
        assert abs(ratio_sum - 1.0) < 1e-6

    def test_apply_mixed_precision_quantization(self, quantizer, sample_tensors):
        precision_labels, stats = quantizer.assign_precision_levels(sample_tensors['importance_scores'])

        quantized_keys, quantized_values, quant_info = quantizer.apply_mixed_precision_quantization(
            sample_tensors['key_states'],
            sample_tensors['value_states'],
            precision_labels
        )

        # Check output shapes
        assert quantized_keys.shape == sample_tensors['key_states'].shape
        assert quantized_values.shape == sample_tensors['value_states'].shape

        # Check quantized tensors are finite
        assert torch.isfinite(quantized_keys).all()
        assert torch.isfinite(quantized_values).all()

        # Check quantization info contains bit assignments
        assert 'bit_assignments' in quant_info

    def test_estimate_memory_savings(self, quantizer, sample_tensors):
        precision_labels = torch.randint(0, 3, sample_tensors['importance_scores'].shape)

        memory_info = quantizer.estimate_memory_savings(
            sample_tensors['key_states'],
            precision_labels
        )

        # Check expected keys exist
        expected_keys = [
            'original_memory_mb', 'compressed_memory_mb', 'compression_ratio',
            'memory_savings', 'high_elements_ratio', 'medium_elements_ratio', 'low_elements_ratio'
        ]
        for key in expected_keys:
            assert key in memory_info

        # Memory savings should be positive
        assert 0 <= memory_info['memory_savings'] <= 1

if __name__ == "__main__":
    pytest.main([__file__, '-v'])