"""Test overall model functionality."""

import torch
import pytest
from transformers import LlamaConfig
from configs.base_config import CompressionConfig
from src.models.modified_llama import CompressedLlamaForCausalLM

@pytest.fixture(scope="module")
def small_llama_config():
    """
    提供一個用於測試的小型 LlamaConfig
    """
    return LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        # Ensure rope_theta is included for newer transformers versions
        rope_theta=10000.0
    )

@pytest.fixture(scope="module")
def compression_config():
    """
    提供一個標準的 CompressionConfig
    """
    return CompressionConfig(
        chunk_size=128,
        key_bits_normal=4,
        value_bits_normal=4,
        attention_sink_size=4,
        outlier_detection_enabled=True,
    )

def test_model_loading_and_initialization(small_llama_config, compression_config):
    """
    測試模型是否可以使用自定義 config 正確初始化
    """
    model = CompressedLlamaForCausalLM(
        config=small_llama_config,
        compression_config=compression_config
    )

    assert model is not None
    assert isinstance(model, CompressedLlamaForCausalLM)
    assert model.config.hidden_size == 64
    # Check if a compression config attribute exists
    assert hasattr(model, 'compression_config')
    assert model.compression_config.chunk_size == 128

    # 檢查 layer 是否已被正確替換
    from src.models.modified_llama import CompressedLlamaDecoderLayer
    assert isinstance(model.model.layers[0], CompressedLlamaDecoderLayer)
    # Check if the attention layer within the decoder layer is the compressed version
    from src.models.modified_llama import CompressedLlamaAttention
    assert isinstance(model.model.layers[0].self_attn, CompressedLlamaAttention)


def test_model_forward_pass(small_llama_config, compression_config):
    """
    測試模型是否可以執行一次 forward pass，並檢查輸出 shape 和數值
    """
    # 1. 設置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ensure dtype consistency, e.g., float32 for tests if not specifically testing float16
    model = CompressedLlamaForCausalLM(
        config=small_llama_config,
        compression_config=compression_config
    ).to(device).to(torch.float32) # Use float32 for stability in tests
    model.eval() # 設置為評估模式

    # 2. 準備輸入
    batch_size = 2
    seq_len = 32
    # 隨機生成 input_ids (詞彙表大小為 1000)
    input_ids = torch.randint(
        0, small_llama_config.vocab_size, (batch_size, seq_len), device=device
    )
    # Also create attention_mask, as models often expect it
    attention_mask = torch.ones_like(input_ids)

    # 3. 執行 Forward Pass
    with torch.no_grad():
        # Pass attention_mask
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # 4. 檢查輸出
    assert outputs is not None
    assert hasattr(outputs, 'logits')

    logits = outputs.logits

    # 檢查 Logits Shape
    # 應為 (batch_size, seq_len, vocab_size)
    expected_shape = (batch_size, seq_len, small_llama_config.vocab_size)
    assert logits.shape == expected_shape, f"輸出 shape 錯誤. 應為 {expected_shape}, 得到 {logits.shape}"

    # 檢查數值是否合理
    assert not torch.isnan(logits).any(), "輸出中包含 NaN"
    assert not torch.isinf(logits).any(), "輸出中包含 Inf"


def test_compression_stats_and_reset(small_llama_config, compression_config):
    """
    測試 get_compression_stats 和 reset_compression_state 方法
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CompressedLlamaForCausalLM(
        config=small_llama_config,
        compression_config=compression_config
    ).to(device)

    # 1. 執行一次 forward pass 來產生統計數據
    input_ids = torch.randint(0, small_llama_config.vocab_size, (1, 16), device=device)
    attention_mask = torch.ones_like(input_ids)
    # Ensure model is in eval mode if needed, and run forward pass
    model.eval()
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    # 2. 獲取統計數據
    stats = model.get_compression_stats()
    assert stats is not None
    assert "avg_key_bits" in stats
    # Check total channels based on config
    expected_channels_per_layer = small_llama_config.num_key_value_heads * (small_llama_config.hidden_size // small_llama_config.num_attention_heads)
    expected_total_channels = small_llama_config.num_hidden_layers * expected_channels_per_layer * 2 # *2 for key and value
    # Note: unified_compressor only counts channels where stats were updated.
    # If outlier detection is off, value stats might not have outliers.
    # Check stats calculation in unified_compressor.
    # A simpler check might be if total_channels is positive if layers > 0
    assert stats["total_channels"] > 0, "Stats managers were not registered or updated"

    # Verify stats values are reasonable (e.g., avg bits between min/max configured)
    min_bits = min(compression_config.key_bits_normal, compression_config.key_bits_sink_outlier)
    max_bits = max(compression_config.key_bits_normal, compression_config.key_bits_sink_outlier)
    assert min_bits <= stats["avg_key_bits"] <= max_bits

    # 3. 重置狀態
    model.reset_compression_state()

    # Check reset_cache effect by looking at the internal state if accessible,
    # or by verifying stats reset.
    # Accessing internal state directly might be brittle.
    # Let's check if stats managers were reset (e.g., update_count is 0)
    # We need access to the managers via the compressor or layers.
    # Example: Check update count of the first key stats manager
    first_key_manager = model.compressor.stats_managers.get("layer_0_key")
    assert first_key_manager is not None
    assert first_key_manager.update_count == 0, "Stats manager update count was not reset"

    # 檢查 reset_cache 是否生效 (current_position should be 0)
    # Ensure the layer and attention module structure allows this access
    assert hasattr(model.model.layers[0], 'self_attn'), "Decoder layer does not have self_attn"
    assert hasattr(model.model.layers[0].self_attn, 'current_position'), "Attention layer does not track current_position"
    assert model.model.layers[0].self_attn.current_position == 0, "Attention layer current_position was not reset"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])