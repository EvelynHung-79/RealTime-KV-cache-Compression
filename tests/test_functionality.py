import sys
import os
import json
import torch

# ===== 路徑設置 =====
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.modified_llama import create_compressed_llama_model
from src.evaluation.longbench_eval import LongBenchEvaluator
from configs.base_config import CompressionConfig
from transformers import AutoTokenizer


# ===== 1️⃣ CUDA 環境檢查 =====
if torch.cuda.is_available():
    device = "cuda"
    print("✅ CUDA is available. Running on GPU.")
else:
    device = "cpu"
    print("⚠️  CUDA not available. Running on CPU. This might be slow.")


# ===== 2️⃣ 模型設定 =====
MODEL_PATH = "models/llama2-7b"

if not os.path.isdir(MODEL_PATH):
    print(f"❌ Error: Model path '{MODEL_PATH}' not found.")
    sys.exit(1)

config = CompressionConfig(
    model_name=MODEL_PATH,
    alpha=0.5,   # Prompt attention weight
    beta=0.3,    # Position bias weight
    gamma=0.2,   # Context relevance weight
    theta_h=0.6, # High precision threshold (保守一點)
    theta_m=0.2, # Medium precision threshold (保守一點)
    early_layer_ratio=0.9,
    middle_layer_ratio=0.8,
    later_layer_ratio=0.7
)

# ===== 3️⃣ 初始化模型與 tokenizer =====
print("🚀 Loading compressed model ...")
model = create_compressed_llama_model(MODEL_PATH, config, device=device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("✅ Model and tokenizer loaded successfully.")


# ===== 4️⃣ 簡單測試模型是否可生成 =====
input_text = "The capital of France is Paris. Paris is known for its beautiful architecture, museums, and culture. Can you tell me more about it."
inputs = tokenizer(input_text, return_tensors="pt").to(device)

print(f"\n🧪 Testing single text generation...\nInput: {input_text}")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output: {generated_text}")

compression_stats = model.get_compression_stats()
print(f"\n💾 Memory savings (approx): {compression_stats.get('overall_memory_savings', 0)*100:.1f}%")
print("-" * 80)

# ===== 5️⃣ LongBench 評測 =====
print("\n🏁 Starting LongBench Evaluation...")

# 初始化 evaluator
evaluator = LongBenchEvaluator(model, tokenizer, config, output_dir="./longbench_results")

# --- 單一任務快速測試 ---
print("\n🎯 Evaluating single task (narrativeqa)...")
single_task_result = evaluator.evaluate_task('narrativeqa', max_samples=3, max_new_tokens=50)

print("\n--- Single Task Result (narrativeqa) ---")
print(json.dumps(single_task_result, indent=2, ensure_ascii=False))

# --- 多任務小規模驗證 ---
print("\n🔥 Running small-scale multi-task benchmark (narrativeqa + qasper)...")
overall_results = evaluator.evaluate_all_tasks(tasks=['narrativeqa', 'qasper'], max_samples_per_task=2)

print("\n--- Overall Results ---")
print(json.dumps(overall_results, indent=2, ensure_ascii=False))

print("\n✅ Evaluation completed. Detailed results saved in ./longbench_results/\n")