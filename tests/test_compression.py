import pytest
import torch
import sys
import os

# ===== 路徑設置 =====
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.base_config import CompressionConfig
from src.compression.streaming_quantization import StreamingStatisticsManager, quantize_chunk
from src.models.modified_llama import CompressedLlamaAttention # 用於獲取 shape 等信息

# --- Fixtures ---

@pytest.fixture
def test_config():
    # 使用與 test_quantization 相同的配置
    return CompressionConfig(
        model_name="test-model",
        num_hidden_layers=2,
        hidden_size=128, # head_dim = 32
        num_attention_heads=4,
        chunk_size=16,
        ema_decay=0.9,
        outlier_threshold_relative=5.0,
        attention_sink_size=4,
        key_bits_normal=4,
        key_bits_sink_outlier=8,
        value_bits_normal=4,
        value_bits_sink_outlier=8,
        value_quant_groups=-1
    )

@pytest.fixture
def stats_manager(test_config):
    return StreamingStatisticsManager(test_config)

@pytest.fixture
def sample_tensors(test_config):
    batch_size = 1
    seq_len = test_config.chunk_size * 2 # Simulate multiple chunks
    hidden_size = test_config.hidden_size
    num_heads = test_config.num_key_value_heads
    head_dim = hidden_size // test_config.num_attention_heads

    # Mock high-precision states (simulating output of K/V proj)
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim) * 2.0
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim) * 2.0
    token_indices = torch.arange(seq_len)

    return key_states, value_states, token_indices

# --- Test Cases ---

class TestStreamingQuantIntegration:

    def test_chunk_processing_loop(self, test_config, stats_manager, sample_tensors):
        """
        模擬 Attention 層中的分塊處理循環，檢查統計更新和量化輸出。
        """
        full_key_states, full_value_states, full_token_indices = sample_tensors
        layer_idx = 0
        chunk_size = test_config.chunk_size

        all_quantized_keys = []
        all_key_scales = []
        all_quantized_values = []
        all_value_scales = []

        # 模擬逐塊處理
        for i in range(0, full_key_states.shape[2], chunk_size):
            key_chunk = full_key_states[:, :, i:i+chunk_size, :]
            value_chunk = full_value_states[:, :, i:i+chunk_size, :] # Assume Value processed post-RoPE, use sample tensor for shape test
            token_indices_chunk = full_token_indices[i:i+chunk_size]

            # 1. 更新 Key 統計 (Pre-RoPE)
            stats_manager.update(layer_idx, 'key', key_chunk)
            # 2. 量化 Key
            quant_key_chunk, key_scale = quantize_chunk(
                key_chunk, layer_idx, 'key', token_indices_chunk, stats_manager, test_config
            )

            # (模擬 RoPE 應用)
            # 假設 value_chunk 是 RoPE 之後的

            # 3. 更新 Value 統計 (Post-RoPE)
            stats_manager.update(layer_idx, 'value', value_chunk) # Use value_chunk for test
            # 4. 量化 Value
            quant_value_chunk, value_scale = quantize_chunk(
                value_chunk, layer_idx, 'value', token_indices_chunk, stats_manager, test_config
            )

            # 檢查輸出
            assert quant_key_chunk.shape == key_chunk.shape
            assert quant_value_chunk.shape == value_chunk.shape
            assert quant_key_chunk.dtype == torch.int8
            assert quant_value_chunk.dtype == torch.int8
            # 檢查 scale 形狀 (Per-channel)
            assert key_scale.shape == (test_config.num_key_value_heads, stats_manager.head_dim)
            assert value_scale.shape == (test_config.num_key_value_heads, stats_manager.head_dim)

            all_quantized_keys.append(quant_key_chunk)
            all_key_scales.append(key_scale) # 注意：scale 在每個 chunk 可能基於更新的 EMA 而略有不同
            all_quantized_values.append(quant_value_chunk)
            all_value_scales.append(value_scale)

        # 檢查統計數據是否已更新（非零）
        final_key_ema, _ = stats_manager.get_stats(layer_idx, 'key')
        final_value_ema, _ = stats_manager.get_stats(layer_idx, 'value')
        assert final_key_ema.abs().sum() > 0
        assert final_value_ema.abs().sum() > 0

        # 可以添加更多檢查，例如拼接後的 shape 等

    def test_sink_outlier_handling_integration(self, test_config, stats_manager, sample_tensors):
        """
        測試量化函數是否能在 Sink 和 Outlier 條件下正確選擇位元數。
        （依賴 test_quantization.py 中對 quantize_chunk 的詳細測試）
        """
        key_states, _, token_indices = sample_tensors
        layer_idx = 0
        tensor_type = 'key'
        chunk_size = test_config.chunk_size

        # 處理第一個 chunk (包含 Sink)
        key_chunk_1 = key_states[:, :, :chunk_size, :]
        token_indices_1 = token_indices[:chunk_size]
        stats_manager.update(layer_idx, tensor_type, key_chunk_1)
        quant_chunk_1, _ = quantize_chunk(
            key_chunk_1, layer_idx, tensor_type, token_indices_1, stats_manager, test_config
        )

        # 檢查 Sink 部分 (前 sink_size 個 token) 是否可能用到 8bit 範圍
        sink_size = test_config.attention_sink_size
        qmax_sink = (2**(test_config.key_bits_sink_outlier - 1)) - 1
        assert torch.max(torch.abs(quant_chunk_1[:, :, :sink_size, :])) <= qmax_sink

        # 處理第二個 chunk (不含 Sink)
        key_chunk_2 = key_states[:, :, chunk_size:chunk_size*2, :]
        token_indices_2 = token_indices[chunk_size:chunk_size*2]

        # 手動標記一個 outlier channel
        stats_manager.key_is_outlier[layer_idx, 0, 0] = True
        stats_manager.update(layer_idx, tensor_type, key_chunk_2) # 更新統計數據
        quant_chunk_2, _ = quantize_chunk(
            key_chunk_2, layer_idx, tensor_type, token_indices_2, stats_manager, test_config
        )

        # 檢查 Outlier channel 是否用了 8bit 範圍
        outlier_channel_data = quant_chunk_2[:, 0, :, 0] # Head 0, Channel 0
        assert torch.max(torch.abs(outlier_channel_data)) <= qmax_sink

        # 檢查 Normal channel 是否用了 4bit 範圍
        normal_channel_data = quant_chunk_2[:, 1, :, 1] # Head 1, Channel 1
        qmax_normal = (2**(test_config.key_bits_normal - 1)) - 1
        assert torch.max(torch.abs(normal_channel_data)) <= qmax_normal