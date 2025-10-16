import pytest
import torch
from configs.base_config import CompressionConfig
from src.compression.token_importance import PromptGuidedImportanceScorer
from src.compression.dynamic_quantization import DynamicPrecisionQuantizer
from src.compression.selective_propagation import SelectiveTokenPropagator

class TestCompressionComponents:

    @pytest.fixture
    def config(self):
        return CompressionConfig(
            model_name="meta-llama/Llama-2-7b-hf",
            num_hidden_layers=4,  # Smaller for testing
            alpha=0.4,
            beta=0.3,
            gamma=0.3,
            theta_h=0.7,
            theta_m=0.3
        )

    @pytest.fixture
    def sample_data(self):
        batch_size, seq_len, hidden_size = 1, 10, 64
        num_heads = 8

        # Mock attention weights
        attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(attention_weights, dim=-1)

        # Mock key/value states
        key_states = torch.randn(batch_size, seq_len, hidden_size)
        value_states = torch.randn(batch_size, seq_len, hidden_size)

        # Mock prompt indices
        prompt_indices = torch.arange(3)  # First 3 tokens as prompt

        return {
            'attention_weights': attention_weights,
            'key_states': key_states, 
            'value_states': value_states,
            'prompt_indices': prompt_indices,
            'seq_len': seq_len,
            'batch_size': batch_size
        }

    def test_importance_scorer(self, config, sample_data):
        scorer = PromptGuidedImportanceScorer(config)

        importance_scores = scorer.compute_importance_scores(
            sample_data['attention_weights'],
            sample_data['prompt_indices'],
            layer_idx=0
        )

        assert importance_scores.shape == (sample_data['batch_size'], sample_data['seq_len'])
        assert torch.isfinite(importance_scores).all()
        assert not torch.isnan(importance_scores).any()

    def test_dynamic_quantizer(self, config, sample_data):
        quantizer = DynamicPrecisionQuantizer(config)

        # Generate mock importance scores
        importance_scores = torch.randn(sample_data['batch_size'], sample_data['seq_len'])

        # Test precision assignment
        precision_labels, precision_stats = quantizer.assign_precision_levels(importance_scores)

        assert precision_labels.shape == importance_scores.shape
        assert precision_labels.dtype == torch.long
        assert all(label in [0, 1, 2] for label in precision_labels.unique().tolist())

        # Test quantization
        quantized_keys, quantized_values, quant_info = quantizer.apply_mixed_precision_quantization(
            sample_data['key_states'], sample_data['value_states'], precision_labels
        )

        assert quantized_keys.shape == sample_data['key_states'].shape
        assert quantized_values.shape == sample_data['value_states'].shape

    def test_selective_propagator(self, config, sample_data):
        propagator = SelectiveTokenPropagator(config)

        # Generate mock importance scores and precision labels
        importance_scores = torch.randn(sample_data['batch_size'], sample_data['seq_len'])
        precision_labels = torch.randint(0, 3, (sample_data['batch_size'], sample_data['seq_len']))

        # Test token selection
        selected_keys, selected_values, selected_scores, selected_labels, prop_info = propagator.apply_token_selection(
            sample_data['key_states'], sample_data['value_states'],
            importance_scores, precision_labels, layer_idx=0
        )

        # Check that selection reduces sequence length
        assert selected_keys.shape[1] <= sample_data['seq_len']
        assert selected_values.shape[1] <= sample_data['seq_len']
        assert selected_keys.shape[1] == selected_values.shape[1]

        # Check propagation ratios
        layer_ratio = propagator.get_layer_propagation_ratio(0)
        assert 0.0 < layer_ratio <= 1.0

if __name__ == "__main__":
    pytest.main([__file__])