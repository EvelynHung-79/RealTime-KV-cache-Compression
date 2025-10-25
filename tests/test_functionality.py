import sys
import os
import json
import torch
import pytest # 引入 pytest 以便可能使用其功能，雖然此腳本主要是執行性的

# ===== 路徑設置 =====
# 確保可以找到 src 和 configs 目錄下的模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.modified_llama import create_compressed_llama_model
from src.evaluation.longbench_eval import LongBenchEvaluator
from configs.base_config import CompressionConfig
from transformers import AutoTokenizer

# --- 測試配置 ---
# 使用較小的模型或確保有足夠資源運行 Llama-2-7b
# 注意：真實模型路徑需要指向您下載的模型
MODEL_PATH = "meta-llama/Llama-2-7b-hf" # 或者您的本地路徑 "models/llama2-7b"
# MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # 備選：使用更小的模型進行快速測試

# 檢查模型路徑是否存在 (如果是本地路徑)
# if not os.path.isdir(MODEL_PATH):
#     pytest.skip(f"Model path '{MODEL_PATH}' not found, skipping functionality test.", allow_module_level=True)

# ===== 1️⃣ CUDA 環境檢查 =====
if torch.cuda.is_available():
    device = "cuda"
    print("✅ CUDA is available. Running on GPU.")
else:
    device = "cpu"
    print("⚠️ CUDA not available. Running on CPU. This might be slow.")

# ===== 2️⃣ 模型設定 (使用新的 CompressionConfig) =====
try:
    # 創建新的 CompressionConfig 實例
    config = CompressionConfig(
        model_name=MODEL_PATH,
        chunk_size=256,
        ema_decay=0.99,
        # outlier_threshold_abs=6.0, # 可以先不設絕對閾值
        outlier_threshold_relative=5.0,
        attention_sink_size=8,
        key_bits_normal=4,
        key_bits_sink_outlier=8,
        value_bits_normal=4,
        value_bits_sink_outlier=8,
        value_quant_groups=-1 # Per-channel value quant
    )
except Exception as e:
     pytest.fail(f"Failed to initialize CompressionConfig: {e}")


# ===== 3️⃣ 初始化模型與 tokenizer =====
print(f"🚀 Loading compressed model from: {MODEL_PATH}")
try:
    # 注意：create_compressed_llama_model 現在需要 compression_config 作為參數
    model = create_compressed_llama_model(MODEL_PATH, compression_config=config, device=device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Model and tokenizer loaded successfully.")
except Exception as e:
    pytest.fail(f"Failed to load model or tokenizer: {e}")


# ===== 4️⃣ 簡單測試模型是否可生成 =====
def test_simple_generation():
    input_text = "The capital of France is Paris. Paris is known for its beautiful architecture, museums, and culture."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    print(f"\n🧪 Testing single text generation...\nInput: {input_text}")
    generated_text = "Generation failed"
    try:
        with torch.no_grad():
            # 重置統計數據
            if hasattr(model, 'reset_compression_state'):
                 model.reset_compression_state()

            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False, # 使用確定性生成以方便測試
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Output: {generated_text}")
            assert len(generated_text) > len(input_text) # 基本檢查：是否生成了新內容
    except Exception as e:
        pytest.fail(f"Simple generation failed: {e}")
    finally:
        print(f"Raw Output: {generated_text}") # 打印原始輸出以供調試

    # 獲取新的統計數據 (內容已改變)
    compression_stats = {}
    if hasattr(model, 'get_compression_stats'):
        compression_stats = model.get_compression_stats()
        print(f"\n📊 Compression Stats: {compression_stats}")
        # 可以加入對 stats 內容的基本斷言，例如 key 是否存在
        assert isinstance(compression_stats, dict)
    else:
        print("\n⚠️ Model does not have get_compression_stats method.")

    print("-" * 80)

# ===== 5️⃣ LongBench 評測 (縮減版測試) =====
# 注意：運行此測試需要下載 LongBench 數據集並可能消耗較長時間和資源
@pytest.mark.skip(reason="LongBench evaluation can be time-consuming and requires dataset.")
def test_longbench_evaluation_short():
    print("\n🏁 Starting LongBench Evaluation (short test)...")

    # 定義輸出目錄
    TEST_OUTPUT_DIR = "experiments/results/test_functionality_output_pytest"
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    try:
        # 初始化 evaluator
        evaluator = LongBenchEvaluator(model, tokenizer, config, output_dir=TEST_OUTPUT_DIR)

        # --- 單一任務快速測試 ---
        print(f"\n🎯 Evaluating single task (narrativeqa), results saved to {TEST_OUTPUT_DIR}...")
        single_task_result = evaluator.evaluate_task(
            'narrativeqa',
            max_samples=2, # 使用非常少的樣本以加速
            max_new_tokens=10 # 使用非常少的新 token
        )

        print("\n--- Single Task Result (narrativeqa) ---")
        print(json.dumps(single_task_result, indent=2, ensure_ascii=False))
        # 基本斷言：結果應該包含預期的 key
        assert 'task_name' in single_task_result
        assert 'num_samples' in single_task_result
        assert 'quality_metrics' in single_task_result
        assert 'compression_metrics' in single_task_result # 評估器聚合後的指標
        assert single_task_result['num_samples'] == 2

    except Exception as e:
        pytest.fail(f"LongBench evaluation failed: {e}")
    finally:
        print(f"\n✅ Evaluation test completed. Detailed results (if run) saved in: {TEST_OUTPUT_DIR}\n")

# 可以添加更多測試，例如測試 reset_compression_state 的效果等

# 如果希望直接運行此文件進行測試 (非 pytest)
if __name__ == "__main__":
    print("Running basic generation test...")
    test_simple_generation()
    # print("\nRunning LongBench short test (if uncommented)...")
    # test_longbench_evaluation_short() # 取消註釋以運行評估測試