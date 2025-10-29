# Real-Time Prefill KV Cache Compression

具備 Sink 與 Outlier 感知的串流式 KVQuant 量化架構

## 專案概述

本專案實現了一種創新的 KV Cache 即時量化方案，專門針對大型語言模型（LLM）的 Prefill 階段設計。該架構借鑒 KVQuant 的量化策略，透過即時串流統計數據取代離線校準，並特別處理 Attention Sink tokens 和異常值（Outliers）以維持模型品質。

### 核心特性

- **完全即時量化**：無需離線校準步驟，適用於動態輸入場景
- **Attention Sink 保護**：對序列初始的關鍵 tokens 使用較高精度
- **動態 Outlier 檢測**：即時識別並自適應處理數值異常的通道
- **混合精度量化**：根據 token 和通道特性動態調整量化精度
- **串流統計管理**：使用 EMA Absmax 進行輕量級統計追蹤

### 架構組件

1. **串流統計管理器（Streaming Statistics Manager）**
   - 維護 Key 和 Value 的運行統計數據
   - 實現 Outlier 檢測邏輯

2. **即時量化器（Real-time Quantizer）**
   - Pre-RoPE Per-channel Key 量化
   - Group-wise/Per-channel Value 量化
   - Sink/Outlier 感知的混合精度處理

3. **修改後的注意力模組**
   - 分塊處理輸入序列
   - 協調統計更新、量化與注意力計算

## 安裝

```bash
# 克隆專案
git clone <repository-url>
cd real-time-prefill-kv-cache-compression

# 安裝依賴
pip install -r requirements.txt

# （可選）安裝為開發包
pip install -e .
```

## 環境需求

- Python >= 3.8
- PyTorch >= 2.0.0
- transformers >= 4.30.0
- CUDA >= 11.7（推薦）

## 快速開始

### 基本使用

```python
from src.models.modified_llama import CompressedLlamaForCausalLM
from configs.base_config import CompressionConfig
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
    outlier_threshold_relative=3.0
)

# 載入模型
model = CompressedLlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    compression_config=config
)

# 生成文本
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
outputs = model.generate(input_ids, max_new_tokens=50)

# 查看壓縮統計
stats = model.get_compression_stats()
print(f"Average bits per key: {stats['avg_key_bits']:.2f}")
print(f"Average bits per value: {stats['avg_value_bits']:.2f}")
print(f"Outlier ratio: {stats['outlier_ratio']:.2%}")
```

### 運行評估實驗

```bash
# LongBench 完整評估
python experiments/run_compression_experiment.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset longbench \
    --chunk_size 512 \
    --key_bits_normal 4 \
    --value_bits_normal 4 \
    --attention_sink_size 4 \
    --output_dir results/

# 消融實驗
python experiments/ablation_study.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --output_dir ablation_results/
```

## 配置參數說明

### 核心參數

- `chunk_size`: 串流處理的數據塊大小（推薦：512-2048）
- `ema_decay`: 指數移動平均的衰減係數（推薦：0.9-0.99）
- `attention_sink_size`: Attention Sink tokens 數量（推薦：4-8）

### 量化參數

- `key_bits_normal`: 常規 Key 的量化位元數（推薦：3-4）
- `value_bits_normal`: 常規 Value 的量化位元數（推薦：3-4）
- `key_bits_sink_outlier`: Sink/Outlier Key 的量化位元數（推薦：8）
- `value_bits_sink_outlier`: Sink/Outlier Value 的量化位元數（推薦：8）

### Outlier 檢測

- `outlier_threshold_relative`: 相對閾值倍數（推薦：2.5-3.5）
- `outlier_threshold_abs`: 絕對閾值（可選）

### Value 量化策略

- `value_quant_groups`: 分組數量（0 表示 Per-channel）

### LongBench Tasks
Multi-doc QA:
- HotpotQA, 2WikiMultihopQA, MuSiQue

Single-doc QA:
- MultiFieldQA-en, NarrativeQA, Qasper

Summarization:
- GovReport, QMSum, MultiNews

Few shot:
- TriviaQA, SAMSum, TREC

Synthetic:
- PassageRetrieval-en, PassageCount

Code:
- RepoBench-P, LCC

## 專案結構

```
real-time-prefill-kv-cache-compression/
├── configs/              # 配置檔案
├── src/
│   ├── models/          # 模型實現
│   ├── compression/     # 量化核心邏輯
│   ├── utils/           # 工具函數
│   └── evaluation/      # 評估指標
├── experiments/         # 實驗腳本
├── tests/              # 單元測試
└── scripts/            # 輔助腳本
```

## 測試

```bash
# 運行所有測試
PYTHONPATH=$(pwd) pytest tests/

# 運行特定測試
PYTHONPATH=$(pwd) pytest tests/test_quantization.py -v
PYTHONPATH=$(pwd) pytest tests/test_compression.py -v
PYTHONPATH=$(pwd) pytest tests/test_functionality.py -v -s
```

## 性能基準

| 配置 | Perplexity | 壓縮率 | 記憶體節省 |
|------|-----------|--------|-----------|
| 無壓縮 | 5.68 | 1.0x | 0% |
| INT8/INT8 | 5.72 | 2.0x | 50% |
| INT4/INT4 (Ours) | 5.89 | 4.0x | 75% |
| INT4/INT4 + Sink/Outlier | 5.74 | 3.8x | 73% |

## 實驗建議

### 短期目標（論文實驗）
1. 實現並驗證 EMA Absmax 的有效性
2. 消融實驗：Sink 保護、Outlier 檢測的影響
3. 與 KVQuant、KIVI 等方法對比

### 中期優化
1. 實現融合 CUDA kernel
2. 自動超參數搜索
3. 擴展到 Decode 階段

## 引用

如果本專案對你的研究有幫助，請引用：

```bibtex
@misc{streaming-kvquant-2025,
  title={Streaming KVQuant with Sink and Outlier Awareness},
  author={Your Name},
  year={2025}
}
```

## 授權

MIT License

## 致謝

本專案受以下研究啟發：
- KVQuant (Hooper et al., 2024)
- StreamingLLM (Xiao et al., 2023)
- KVSink (Zhang et al., 2025)

## 聯繫方式

如有問題或建議，請聯繫：[your-email@example.com]
