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
    assert model.compression_config.chunk_size == 128
    
    # 檢查 layer 是否已被正確替換
    from src.models.modified_llama import CompressedLlamaDecoderLayer
    assert isinstance(model.model.layers[0], CompressedLlamaDecoderLayer)


def test_model_forward_pass(small_llama_config, compression_config):
    """
    測試模型是否可以執行一次 forward pass，並檢查輸出 shape 和數值
    """
    # 1. 設置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CompressedLlamaForCausalLM(
        config=small_llama_config,
        compression_config=compression_config
    ).to(device)
    model.eval() # 設置為評估模式

    # 2. 準備輸入
    batch_size = 2
    seq_len = 32
    # 隨機生成 input_ids (詞彙表大小為 1000)
    input_ids = torch.randint(
        0, small_llama_config.vocab_size, (batch_size, seq_len), device=device
    )

    # 3. 執行 Forward Pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

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
    model = CompressedLlamaForCausalLM(
        config=small_llama_config,
        compression_config=compression_config
    )
    
    # 1. 執行一次 forward pass 來產生統計數據
    input_ids = torch.randint(0, small_llama_config.vocab_size, (1, 16))
    model(input_ids)

    # 2. 獲取統計數據
    stats = model.get_compression_stats()
    assert stats is not None
    assert "avg_key_bits" in stats
    assert stats["total_channels"] > 0 # 確保 stats_managers 被註冊

    # 3. 重置狀態
    model.reset_compression_state()
    stats_after_reset = model.get_compression_stats()
    
    # 注意：outlier_ratio 在重置後可能仍為 0，但 update_count 應該歸零
    # 檢查 UnifiedCompressor 的 get_overall_compression_stats 實作
    # 我們的實作 
    # 和 streaming_quantization 
    # 表明 reset 會重置 update_count 和 outlier_history
    
    # 檢查 reset_cache 是否生效
    assert model.model.layers[0].self_attn.current_position == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])