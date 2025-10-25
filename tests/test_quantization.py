import pytest
import torch
import sys
import os

# ===== 路徑設置 =====
# 確保可以找到 src 目錄下的模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.base_config import CompressionConfig
from src.compression.streaming_quantization import (
    StreamingStatisticsManager,
    calculate_scale,
    quantize_symmetric,
    dequantize_symmetric,
    quantize_chunk
)

# --- Fixtures ---

@pytest.fixture
def base_config():
    # 使用一個簡化的配置進行測試
    return CompressionConfig(
        model_name="test-model",
        num_hidden_layers=2,
        hidden_size=128, # head_dim = 128 / 4 = 32
        num_attention_heads=4,
        chunk_size=16,
        ema_decay=0.9,
        outlier_threshold_relative=5.0,
        attention_sink_size=4,
        key_bits_normal=4,
        key_bits_sink_outlier=8,
        value_bits_normal=4,
        value_bits_sink_outlier=8,
        value_quant_groups=-1 # Per-channel value quant
    )

@pytest.fixture
def stats_manager(base_config):
    manager = StreamingStatisticsManager(base_config)
    # 手動設置為已初始化，以便測試 outlier 檢測
    manager.initialized.data = torch.tensor(True)
    return manager

@pytest.fixture
def sample_chunk_data(base_config):
    batch_size = 1
    num_heads = base_config.num_key_value_heads # 假設 num_kv_heads = num_attention_heads
    chunk_len = base_config.chunk_size
    head_dim = base_config.hidden_size // base_config.num_attention_heads
    # 創建一個高精度 chunk tensor [B, H, C_len, D]
    chunk_tensor = torch.randn(batch_size, num_heads, chunk_len, head_dim) * 2.0 # 增加數值範圍
    # 創建 token 索引
    token_indices = torch.arange(chunk_len)
    return chunk_tensor, token_indices

# --- Test Cases ---

class TestStreamingStatisticsManager:

    def test_initialization(self, stats_manager, base_config):
        assert stats_manager.key_ema_absmax.shape == (base_config.num_hidden_layers, base_config.num_key_value_heads, stats_manager.head_dim)
        assert stats_manager.value_ema_absmax.shape == (base_config.num_hidden_layers, base_config.num_key_value_heads, stats_manager.head_dim) # because value_quant_groups=-1
        assert not stats_manager.key_is_outlier.any()
        assert not stats_manager.value_is_outlier.any()
        assert not stats_manager.initialized.item() # 初始化後應為 False

    def test_update_ema(self, stats_manager, sample_chunk_data, base_config):
        chunk_tensor, _ = sample_chunk_data
        layer_idx = 0
        tensor_type = 'key'

        initial_absmax = stats_manager.get_stats(layer_idx, tensor_type)[0].clone()
        assert torch.all(initial_absmax == 0) # Initially zero

        stats_manager.update(layer_idx, tensor_type, chunk_tensor)
        first_update_absmax = stats_manager.get_stats(layer_idx, tensor_type)[0]

        # 第一次更新 (initialized=False)，EMA 應該等於 current absmax
        current_absmax = chunk_tensor.abs().amax(dim=(0, 2))
        assert torch.allclose(first_update_absmax, torch.maximum(current_absmax, torch.tensor(1e-5)))

        # 第二次更新
        chunk_tensor_2 = torch.randn_like(chunk_tensor) * 0.5
        stats_manager.update(layer_idx, tensor_type, chunk_tensor_2)
        second_update_absmax = stats_manager.get_stats(layer_idx, tensor_type)[0]
        current_absmax_2 = chunk_tensor_2.abs().amax(dim=(0, 2))
        expected_ema = base_config.ema_decay * first_update_absmax + (1.0 - base_config.ema_decay) * current_absmax_2
        expected_ema = torch.maximum(expected_ema, torch.tensor(1e-5))
        assert torch.allclose(second_update_absmax, expected_ema)

    def test_outlier_detection(self, stats_manager, sample_chunk_data, base_config):
        chunk_tensor, _ = sample_chunk_data
        layer_idx = 0
        tensor_type = 'key'

        # 初始更新以設置 EMA
        stats_manager.update(layer_idx, tensor_type, chunk_tensor * 0.1) # 用小值初始化
        initial_ema, initial_outlier = stats_manager.get_stats(layer_idx, tensor_type)
        assert not initial_outlier.any()

        # 創建一個包含 outlier 的 chunk
        outlier_chunk = chunk_tensor.clone()
        outlier_value = initial_ema[0, 0].item() * (base_config.outlier_threshold_relative + 1.0) # 確保超過相對閾值
        outlier_chunk[0, 0, 0, 0] = outlier_value # 在第一個通道製造 outlier

        stats_manager.update(layer_idx, tensor_type, outlier_chunk)
        _, updated_outlier = stats_manager.get_stats(layer_idx, tensor_type)

        # 檢查對應通道是否被標記為 outlier
        assert updated_outlier[0, 0].item() is True
        # 檢查其他通道是否未被標記 (除非隨機值也觸發了)
        assert not updated_outlier[1:, :].any()
        assert not updated_outlier[0, 1:].any()

    def test_reset(self, stats_manager, sample_chunk_data):
        chunk_tensor, _ = sample_chunk_data
        layer_idx = 0
        stats_manager.update(layer_idx, 'key', chunk_tensor)
        stats_manager.update(layer_idx, 'value', chunk_tensor)
        stats_manager.initialized.data = torch.tensor(True) # 手動設為 True

        assert stats_manager.key_ema_absmax.abs().sum() > 0
        assert stats_manager.value_ema_absmax.abs().sum() > 0
        assert stats_manager.initialized.item() is True

        stats_manager.reset()

        assert torch.all(stats_manager.key_ema_absmax == 0)
        assert torch.all(stats_manager.value_ema_absmax == 0)
        assert not stats_manager.key_is_outlier.any()
        assert not stats_manager.value_is_outlier.any()
        assert stats_manager.initialized.item() is False


class TestQuantizationFunctions:

    def test_calculate_scale(self):
        absmax = torch.tensor([1.0, 4.0, 8.0])
        bits = 4
        # qmax = 2**(4-1) - 1 = 7
        expected_scale = absmax / 7.0
        scale = calculate_scale(absmax, bits)
        assert torch.allclose(scale, expected_scale)
        # Test epsilon
        absmax_zero = torch.tensor([0.0])
        scale_zero = calculate_scale(absmax_zero, 4, epsilon=1e-5)
        assert scale_zero > 0

    def test_quantize_dequantize_symmetric(self):
        tensor = torch.tensor([-4.0, -1.5, 0.0, 1.5, 4.0])
        absmax = torch.tensor([4.0])
        bits = 4
        qmax = 7
        qmin = -8 # 注意 INT4 的範圍通常是非對稱的，但對稱量化 clip 到 [-7, 7]
        scale = calculate_scale(absmax, bits) # scale = 4.0 / 7.0

        quantized = quantize_symmetric(tensor, scale, bits)
        expected_quantized = torch.round(tensor / scale).clamp(qmin, qmax).to(torch.int8) # [-7, -3, 0, 3, 7]
        # 由於 clip 到 [-7, 7] in quantize_symmetric
        expected_quantized_clipped = torch.round(tensor / scale).clamp(-7, 7).to(torch.int8) # [-7, -3, 0, 3, 7]
        # 注意: quantize_symmetric 實現中 clip 範圍應為 [-qmax, qmax] for symmetric
        # If bits=4, qmax=7, clip range should be [-7, 7]
        # Let's adjust expected based on [-7, 7] clipping
        # tensor / scale approx: [-7, -2.625, 0, 2.625, 7]
        # rounded: [-7, -3, 0, 3, 7]
        # clamped [-7, 7]: [-7, -3, 0, 3, 7]
        assert torch.equal(quantized, expected_quantized_clipped)

        dequantized = dequantize_symmetric(quantized, scale)
        # Should be close to original, but with quantization error
        assert torch.allclose(dequantized, tensor, atol=scale.item()/2) # Allow half-scale error

    def test_quantize_chunk_normal(self, stats_manager, sample_chunk_data, base_config):
        chunk_tensor, token_indices = sample_chunk_data
        layer_idx = 0
        tensor_type = 'key'
        # Update stats first
        stats_manager.update(layer_idx, tensor_type, chunk_tensor)

        quantized_chunk, scale = quantize_chunk(
            chunk_tensor, layer_idx, tensor_type, token_indices, stats_manager, base_config
        )

        assert quantized_chunk.dtype == torch.int8 # Because max bits is 8
        assert quantized_chunk.shape == chunk_tensor.shape
        assert scale.shape == stats_manager.get_stats(layer_idx, tensor_type)[0].shape # Match stat shape

        # Dequantize and check error
        dequantized_chunk = dequantize_symmetric(quantized_chunk, scale)
        # Error check (atol depends on scale and bits) - use mean absolute error
        mae = torch.mean(torch.abs(dequantized_chunk - chunk_tensor))
        # Expect MAE to be relatively small, roughly proportional to scale
        avg_scale = scale.mean().item()
        assert mae < avg_scale # Heuristic check

    def test_quantize_chunk_sink(self, stats_manager, sample_chunk_data, base_config):
        chunk_tensor, token_indices = sample_chunk_data # chunk_len=16, sink_size=4
        layer_idx = 0
        tensor_type = 'key'
        stats_manager.update(layer_idx, tensor_type, chunk_tensor)

        quantized_chunk, scale = quantize_chunk(
            chunk_tensor, layer_idx, tensor_type, token_indices, stats_manager, base_config
        )

        # Calculate expected ranges
        qmax_normal = (2**(base_config.key_bits_normal - 1)) - 1 # 4 bits -> 7
        qmax_sink = (2**(base_config.key_bits_sink_outlier - 1)) - 1 # 8 bits -> 127

        # Check ranges
        sink_part = quantized_chunk[:, :, :base_config.attention_sink_size, :]
        normal_part = quantized_chunk[:, :, base_config.attention_sink_size:, :]

        assert torch.max(torch.abs(sink_part)) <= qmax_sink
        if normal_part.numel() > 0:
             assert torch.max(torch.abs(normal_part)) <= qmax_normal
             # Check if sink actually used higher range (if values large enough)
             # This requires specific input, hard to guarantee with random

    def test_quantize_chunk_outlier(self, stats_manager, sample_chunk_data, base_config):
        chunk_tensor, token_indices = sample_chunk_data
        layer_idx = 0
        tensor_type = 'key'

        # Force an outlier status on a channel
        stats_manager.key_is_outlier[layer_idx, 0, 0] = True # Mark channel 0 of head 0 as outlier
        stats_manager.update(layer_idx, tensor_type, chunk_tensor) # Update EMA

        quantized_chunk, scale = quantize_chunk(
            chunk_tensor, layer_idx, tensor_type, token_indices, stats_manager, base_config
        )

        qmax_normal = (2**(base_config.key_bits_normal - 1)) - 1 # 7
        qmax_outlier = (2**(base_config.key_bits_sink_outlier - 1)) - 1 # 127

        # Check ranges for the specific channel
        outlier_channel_data = quantized_chunk[:, 0, :, 0] # All tokens in chunk for head 0, channel 0
        normal_channel_data = quantized_chunk[:, 1, :, 1] # Example normal channel

        # Sink tokens might also be 8-bit, exclude them from normal check
        sink_size = base_config.attention_sink_size
        outlier_channel_normal_tokens = outlier_channel_data[:, sink_size:]
        normal_channel_normal_tokens = normal_channel_data[:, sink_size:]


        # Outlier channel (even non-sink tokens) should use outlier bits
        if outlier_channel_normal_tokens.numel() > 0:
            assert torch.max(torch.abs(outlier_channel_normal_tokens)) <= qmax_outlier

        # Normal channel (non-sink tokens) should use normal bits
        if normal_channel_normal_tokens.numel() > 0:
             assert torch.max(torch.abs(normal_channel_normal_tokens)) <= qmax_normal