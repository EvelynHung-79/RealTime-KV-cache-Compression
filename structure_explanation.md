# 專案架構與檔案說明 (串流式 KVQuant 量化版)

本文件說明 Real-time Prefill KV Cache Compression 專案的架構，採用基於 KVQuant 思想的即時串流 KV Cache 量化方法，並特別處理 Attention Sink 與 Outliers。此版本**不包含**選擇性傳播 (Selective Propagation) 或複雜的 Token 重要性評分。

## Repository Structure (更新後)

```

real-time-prefill-kv-cache-compression/
├── README.md
├── requirements.txt
├── setup.py \# (如果有的話)
├── configs/
│   ├── **init**.py
│   └── base\_config.py \# 更新：包含串流、量化位元、Sink/Outlier 參數
├── src/
│   ├── **init**.py
│   ├── models/
│   │   ├── **init**.py
│   │   ├── modified\_llama.py \# 更新：實現分塊處理、串流量化、Sink/Outlier 感知
│   │   └── compression\_layers.py \# (可選，可能包含輔助層如 QuantizedLinear)
│   ├── compression/
│   │   ├── **init**.py
│   │   ├── streaming\_quantization.py \# 新增：包含串流統計管理器和量化函數
│   │   └── unified\_compressor.py \# 更新：主要負責持有和重置統計管理器
│   ├── utils/
│   │   ├── **init**.py
│   │   ├── memory\_utils.py
│   │   ├── eval\_utils.py
│   │   └── data\_utils.py
│   └── evaluation/
│       ├── **init**.py
│       ├── longbench\_eval.py \# 更新：需適配新的統計數據獲取方式
│       ├── metrics.py \# 更新：需調整指標計算（基於位元數而非 Token 數）
│       └── benchmark\_runner.py \# 更新：測試配置和指標需調整
├── experiments/
│   ├── **init**.py
│   ├── run\_compression\_experiment.py \# 更新：使用新配置參數和流程
│   ├── ablation\_study.py \# 更新：需基於新參數重寫消融實驗
│   └── hyperparameter\_tuning.py \# 更新：需基於新參數重寫搜索空間和目標函數
├── tests/ \# 更新：需重寫測試以驗證新架構
│   ├── **init**.py
│   ├── test\_compression.py \# 需重寫，測試串流、量化整合
│   └── test\_quantization.py \# 需重寫，測試 streaming\_quantization.py
└── scripts/
│   └── download\_model.py \# (原 download\_models.sh 已改為 .py)
│   └── setup\_environment.sh

```

## Execution Flow (更新後)

```

run\_compression\_experiment.py
↓
src/models/modified\_llama.py (create\_compressed\_llama\_model)
→ 創建壓縮版 LLaMA 模型 (內含 StreamingStatisticsManager)
↓
src/models/modified\_llama.py (CompressedLlamaAttention.forward - Prefill 階段)
→ 將輸入 hidden\_states 分塊 (Chunking)
→ 對每個 Chunk:
→ 計算 K, V (高精度)
→ **(Pre-RoPE)** 更新 Key 的串流統計 (EMA Absmax) 並檢測 Outlier (調用 StreamingStatisticsManager)
→ **(Pre-RoPE)** 計算 Key 量化參數 (考慮 Sink/Outlier 狀態)
→ **(Pre-RoPE)** 量化 Key Chunk (調用 streaming\_quantization 函數)
→ 將量化 Key 存入 KV Cache
→ 應用 RoPE (使用高精度或臨時反量化 K/V)
→ **(Post-RoPE)** 更新 Value 的串流統計並檢測 Outlier
→ **(Post-RoPE)** 計算 Value 量化參數 (考慮 Sink/Outlier 狀態)
→ **(Post-RoPE)** 量化 Value Chunk
→ 將量化 Value 存入 KV Cache
→ 使用 **高精度** Q, K\_rope, V\_rope 計算 Attention Output Chunk
→ 合併所有 Attention Output Chunks
→ 傳遞高精度 Hidden States 到下一層 (所有 Tokens)
↓
(逐層重複 Attention 模組流程)
↓
src/evaluation/longbench\_eval.py
→ 執行 LongBench 評估 (使用帶量化 KV Cache 的模型)
↓
src/utils/
→ memory\_utils.py (監控記憶體使用)
→ eval\_utils.py (計算評估指標, 如 F1, ROUGE)
→ data\_utils.py (載入和預處理資料)

```

## Directory & File Descriptions (更新後)

### 1. `configs/`
配置文件目錄，管理所有實驗參數。

- **`base_config.py`**
  - **核心配置**: 定義 `CompressionConfig` dataclass。
  - **新增參數**: 包含串流處理 (`chunk_size`)、串流統計 (`ema_decay`)、Outlier 檢測 (`outlier_threshold_abs`, `outlier_threshold_relative`)、Attention Sink (`attention_sink_size`) 以及 Key/Value 在不同情況下的量化位元數 (`key_bits_normal`, `key_bits_sink_outlier`, `value_bits_normal`, `value_bits_sink_outlier`)、Value 分組 (`value_quant_groups`) 等參數。
  - **移除參數**: 移除了與重要性評分 (alpha, beta, gamma, theta\_h/m) 和選擇性傳播 (ratios) 相關的參數。

- **`model_configs/`**
  - (可選) 存放特定基礎模型（如 llama2-7b）的固有參數。

### 2. `src/compression/`
核心壓縮模組，實現即時 KV cache 串流量化。

- **`streaming_quantization.py`** (新增)
  - **`StreamingStatisticsManager`**:
    - 管理 Key (Per-channel) 和 Value (Group-wise/Per-channel) 的 EMA Absmax 運行統計量。
    - 內建 Outlier 檢測邏輯，標記異常通道/分組。
    - 提供接口更新統計、獲取統計、查詢 Outlier 狀態、重置狀態。
  - **量化函數**:
    - `calculate_scale`: 計算對稱量化的 scale。
    - `quantize_symmetric`: 執行對稱量化。
    - `dequantize_symmetric`: 執行反量化。
    - `quantize_chunk`: 核心函數，根據 Sink/Outlier 狀態選擇合適位元數，調用量化函數處理數據區塊。

- **`unified_compressor.py`**
  - **職責簡化**: 主要負責在模型初始化時創建 `StreamingStatisticsManager` 實例。
  - **`reset_compression_state`**: 提供重置 `StreamingStatisticsManager` 狀態的接口。
  - **`get_overall_compression_stats`**: 提供獲取量化相關統計數據（如 Outlier 比例、估計記憶體節省）的接口。

### 3. `src/models/`
模型架構修改與整合。

- **`modified_llama.py`**
  - **`CompressedLlamaAttention`**:
    - **核心修改**: 實現 Prefill 階段的**分塊 (Chunking)** 處理邏輯。
    - **串流量化整合**: 在處理每個 chunk 時，調用 `StreamingStatisticsManager` 更新統計並檢測 Outlier，調用 `streaming_quantization.py` 中的函數執行 Key (Pre-RoPE, Per-channel) 和 Value (Post-RoPE, Group-wise/Simplified) 的量化。
    - **Sink/Outlier 感知**: 將 token 索引傳遞給量化函數，以實現對 Sink tokens 和 Outlier 通道/分組的特殊精度處理。
    - **KV Cache 管理**: 負責將量化後的 K/V chunk 存入快取 (可能需要自定義快取結構或修改 `past_key_value` 的處理)。
    - **注意力計算**: 確保使用高精度（或反量化後）的 Q, K, V 進行注意力計算。
    - **階段區分**: 包含簡單邏輯區分 Prefill 和 Decode 階段，Decode 階段直接使用（並更新）量化快取。
  - **`CompressedLlamaDecoderLayer`**: 封裝 `CompressedLlamaAttention`，確保參數正確傳遞。
  - **`CompressedLlamaForCausalLM`**:
    - 在初始化時創建並持有 `StreamingStatisticsManager` 實例。
    - 將 `StreamingStatisticsManager` 實例和 `CompressionConfig` 注入到每一層的 `CompressedLlamaDecoderLayer`。
    - 提供 `reset_compression_state` 接口調用管理器的重置方法。

- **`compression_layers.py`**
  - (可選) 可能包含一些輔助層，例如用於模型權重量化的 `QuantizedLinear`。`CompressedKVCache` 的部分邏輯可能已被整合進 `CompressedLlamaAttention`。

### 4. `src/evaluation/`
評估與基準測試模組。

- **`longbench_eval.py`**
  - **評估器**: 提供 `LongBenchEvaluator` 類別。
  - **更新**: 需要調整從 `model.get_compression_stats()` 獲取和聚合統計數據的方式，以匹配新架構的輸出。

- **`metrics.py`**
  - **指標計算**: 實作 F1、ROUGE-L 等任務指標。
  - **`CompressionMetrics`**: 需要重構，不再計算基於 token 數量的壓縮率，而是收集和報告量化相關指標（如平均位元數、Outlier 比例、估計記憶體節省）。

- **`benchmark_runner.py`**
  - **基準測試**: 提供 `CompressionBenchmark` 類別。
  - **更新**: 需要更新測試配置的生成方式和結果分析邏輯，以適應新的超參數和評估指標。

### 5. `src/utils/`
通用工具函數。

- **`memory_utils.py`**: 提供 `MemoryMonitor` 類別，功能不變。
- **`eval_utils.py`**: 包含 logging、任務指標計算函數，功能不變。
- **`data_utils.py`**: 數據加載與預處理，功能不變。

### 6. `experiments/`
實驗執行腳本。

- **`run_compression_experiment.py`**
  - **主實驗腳本**: 串接所有模組執行量化流程與評估。
  - **更新**: 使用新的配置參數，調用更新後的模型和評估流程，記錄新的統計指標。

- **`ablation_study.py`**
  - **消融研究**: 自動化測試不同參數設定的影響。
  - **更新**: 需要基於新的配置參數（如 chunk\_size, ema\_decay, bits, sink\_size）重新設計消融實驗。

- **`hyperparameter_tuning.py`**
  - **超參數自動搜尋**: 尋找最佳參數組合。
  - **更新**: 需要基於新的配置參數重新定義搜索空間和目標函數。

### 7. `tests/`
單元測試與功能測試。

- **`test_compression.py`**: (需重寫) 測試 `modified_llama.py` 中分塊、量化、統計更新的整合邏輯。
- **`test_quantization.py`**: (需重寫) 測試 `streaming_quantization.py` 中的統計更新、Outlier 檢測、量化函數在不同情況下（Sink, Outlier, Normal）的正確性。

### 8. `scripts/`
自動化腳本。

- **`setup_environment.sh`**: 環境設置，功能不變。
- **`download_model.py`**: 下載模型，功能不變。