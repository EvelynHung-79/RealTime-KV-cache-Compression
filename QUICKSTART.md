# 快速入門指南

## 安裝步驟

### 1. 解壓專案
```bash
unzip real-time-prefill-kv-cache-compression.zip
cd real-time-prefill-kv-cache-compression
```

### 2. 設置環境
```bash
# 方法 A: 使用提供的腳本
bash scripts/setup_environment.sh

# 方法 B: 手動設置
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

pip install -r requirements.txt
# pip install -e .
```

### 3. 下載模型（可選）
```bash
python scripts/download_model.py
```

## 基本使用

### 運行簡單實驗
```bash
python experiments/run_compression_experiment.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --chunk_size 512 \
    --key_bits_normal 4 \
    --value_bits_normal 4 \
    --attention_sink_size 4 \
    --outlier_detection_enabled \
    --max_samples 10
```

### 運行消融實驗
```bash
python experiments/ablation_study.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --output_dir ablation_results/
```

### 運行超參數調優
```bash
python experiments/hyperparameter_tuning.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --n_trials 20
```

## 在 Python 中使用

```python
from src.models.modified_llama import CompressedLlamaForCausalLM
from configs.base_config import CompressionConfig
from transformers import AutoTokenizer
import torch

# 配置壓縮參數
config = CompressionConfig(
    chunk_size=512,
    ema_decay=0.9,
    attention_sink_size=4,
    key_bits_normal=4,
    value_bits_normal=4,
    key_bits_sink_outlier=8,
    value_bits_sink_outlier=8,
    outlier_threshold_relative=3.0,
    outlier_detection_enabled=True,
)

# 載入模型
model = CompressedLlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    compression_config=config,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 生成文本
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))

# 查看壓縮統計
stats = model.get_compression_stats()
print(f"平均 Key 位元數: {stats['avg_key_bits']:.2f}")
print(f"平均 Value 位元數: {stats['avg_value_bits']:.2f}")
print(f"壓縮比: {stats['compression_ratio_key']:.2f}x")
```

## 運行測試

```bash
# 運行所有測試
pytest tests/ -v

# 運行特定測試
pytest tests/test_quantization.py -v
pytest tests/test_compression.py -v
```

## 專案結構

```
real-time-prefill-kv-cache-compression/
├── README.md                  # 專案總覽
├── QUICKSTART.md              # 本文檔
├── requirements.txt           # 依賴套件
├── setup.py                   # 安裝設定
├── configs/                   # 配置模組
│   └── base_config.py         # 核心配置
├── src/                       # 源代碼
│   ├── models/                # 模型實現
│   │   ├── modified_llama.py  # 壓縮版 Llama
│   │   └── compression_layers.py
│   ├── compression/           # 量化核心
│   │   ├── streaming_quantization.py
│   │   └── unified_compressor.py
│   ├── utils/                 # 工具函數
│   └── evaluation/            # 評估模組
├── experiments/               # 實驗腳本
│   ├── run_compression_experiment.py
│   ├── ablation_study.py
│   └── hyperparameter_tuning.py
├── tests/                     # 單元測試
└── scripts/                   # 輔助腳本
```

## 配置參數說明

### 核心參數
- `chunk_size`: 串流處理的數據塊大小（推薦：512-2048）
- `ema_decay`: 指數移動平均的衰減係數（推薦：0.9-0.99）
- `attention_sink_size`: Attention Sink tokens 數量（推薦：4-8）

### 量化參數
- `key_bits_normal`: 常規 Key 的量化位元數（推薦：3-4）
- `value_bits_normal`: 常規 Value 的量化位元數（推薦：3-4）
- `key_bits_sink_outlier`: Sink/Outlier Key 的位元數（推薦：8）
- `value_bits_sink_outlier`: Sink/Outlier Value 的位元數（推薦：8）

### Outlier 檢測
- `outlier_threshold_relative`: 相對閾值倍數（推薦：2.5-3.5）
- `outlier_detection_enabled`: 是否啟用 Outlier 檢測

## 常見問題

### Q: 記憶體不足怎麼辦？
A: 嘗試：
1. 減小 `chunk_size`
2. 使用較低的量化位元數
3. 使用 `torch.float16` 或 `bfloat16`
4. 啟用 `device_map="auto"` 進行模型分片

### Q: 如何評估不同配置？
A: 使用提供的消融實驗腳本：
```bash
python experiments/ablation_study.py
```

### Q: 如何自定義量化策略？
A: 修改 `src/compression/streaming_quantization.py` 中的量化函數，
或創建新的 `CompressionConfig`。

## 下一步

1. 運行基礎實驗驗證設置
2. 嘗試不同的配置參數
3. 在你的數據集上評估
4. 根據結果調整超參數
5. 實現自定義功能（如融合 CUDA kernel）

## 技術支持

如有問題，請：
1. 查看 README.md 獲取詳細文檔
2. 查看測試文件了解使用範例
3. 提交 Issue 或聯繫作者

## 引用

如果本專案對你的研究有幫助，請引用：

```bibtex
@misc{streaming-kvquant-2025,
  title={Streaming KVQuant with Sink and Outlier Awareness for Real-time Prefill KV Cache Compression},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/real-time-prefill-kv-cache-compression}
}
```

---

**祝研究順利！** 🚀
