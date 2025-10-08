***
# 專案架構與檔案說明

## Repository Structure

```
real-time-prefill-kv-cache-compression/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── __init__.py
│   ├── base_config.py
│   └── model_configs/
│       ├── __init__.py
│       ├── llama2_7b.py
│       └── llama2_13b.py
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── modified_llama.py
│   │   └── compression_layers.py
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── token_importance.py
│   │   ├── dynamic_quantization.py
│   │   ├── selective_propagation.py
│   │   └── unified_compressor.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── memory_utils.py
│   │   ├── eval_utils.py
│   │   └── data_utils.py
│   └── evaluation/
│       ├── __init__.py
│       ├── longbench_eval.py
│       ├── metrics.py
│       └── benchmark_runner.py
├── experiments/
│   ├── __init__.py
│   ├── run_compression_experiment.py
│   ├── ablation_study.py
│   └── hyperparameter_tuning.py
├── tests/
│   ├── __init__.py
│   ├── test_compression.py
│   ├── test_importance_scoring.py
│   └── test_quantization.py
└── scripts/
    ├── download_models.sh
    └── setup_environment.sh
```

## Execution Flow

```
run_compression_experiment.py
    ↓
src/models/modified_llama.py
    → 創建壓縮版 LLaMA 模型
    → 註冊 hook 和壓縮器
    ↓
src/compression/unified_compressor.py
    → 協調整個壓縮流程
    ↓
    → src/compression/token_importance.py
        → 計算 token 重要性分數
    ↓
    → src/compression/dynamic_quantization.py
        → 執行動態精度分配
    ↓
    → src/compression/selective_propagation.py
        → 執行選擇性傳播
    ↓
src/evaluation/longbench_eval.py
    → 執行 LongBench 評估
    ↓
src/utils/
    → memory_utils.py (監控記憶體使用)
    → eval_utils.py (計算評估指標)
    → data_utils.py (載入和預處理資料)
```

## Directory & File Descriptions

### 1. `configs/`
配置文件目錄，管理所有實驗參數

- **`base_config.py`**
  - 儲存所有壓縮相關的超參數（layer 傳播比例、精度閾值、重要性權重等）
  - 供主程式與各模組讀取，確保所有流程參數一致
  - 定義 `CompressionConfig` 類別

- **`model_configs/`**
  - **`llama2_7b.py`**: LLaMA2-7B 專用配置
  - **`llama2_13b.py`**: LLaMA2-13B 專用配置

### 2. `src/compression/`
核心壓縮模組，實現即時 KV cache 壓縮

- **`token_importance.py`**
  - 實作 token 重要性評分公式（prompt-guided 三項式）
  - 提供 `PromptGuidedImportanceScorer` 類別
  - 可根據 config 調整權重 (α, β, γ)

- **`dynamic_quantization.py`**
  - 根據重要性分數分配不同量化精度（2/4/8 bit）
  - 提供 `DynamicPrecisionQuantizer` 類別
  - 執行混合精度量化

- **`selective_propagation.py`**
  - 根據重要性分數與層級預算選擇要傳播的 tokens
  - 提供 `SelectiveTokenPropagator` 類別
  - 支援 per-layer ratio 與 drop 機制

- **`unified_compressor.py`**
  - 整合上述三個模組，實現完整壓縮流程
  - 提供 `RealTimePrefillCompressor` 類別
  - 高階 API：`compress_layer_kv_cache()`

### 3. `src/models/`
模型架構修改與整合

- **`modified_llama.py`**
  - 修改 LLaMA2 模型以支援 KV cache 壓縮
  - 提供 `CompressedLlamaAttention` 和 `CompressedLlamaForCausalLM`
  - 在每個 transformer layer 插入壓縮流程

- **`compression_layers.py`**
  - 自訂壓縮相關的神經網路層
  - 提供 `CompressedKVCache`、`QuantizedLinear` 等工具

### 4. `src/evaluation/`
評估與基準測試模組

- **`longbench_eval.py`**
  - LongBench 資料集評估器
  - 提供 `LongBenchEvaluator` 類別
  - 支援 13 個長文本任務的評估

- **`metrics.py`**
  - 實作各種評估指標（F1、ROUGE-L、EM 等）
  - 提供 `CompressionMetrics` 類別
  - 計算壓縮率、記憶體節省等指標

- **`benchmark_runner.py`**
  - 批次實驗執行與結果收集
  - 提供 `CompressionBenchmark` 類別
  - 支援多配置比較與視覺化

### 5. `src/utils/`
通用工具函數

- **`data_utils.py`**
  - 資料載入、預處理、tokenization
  - 提供 `LongBenchDataLoader` 和 `DataCollator`
  - 支援 LongBench 資料集處理

- **`memory_utils.py`**
  - 記憶體監控工具
  - 提供 `MemoryMonitor` 類別
  - 追踪 GPU/CPU 記憶體使用

- **`eval_utils.py`**
  - 評估相關的輔助函數
  - 提供 logging、metrics 計算等工具
  - 包含 `PerformanceTimer` 類別

### 6. `experiments/`
實驗執行腳本

- **`run_compression_experiment.py`**
  - **主實驗腳本**，串接所有模組執行完整壓縮流程與評估
  - 支援自訂超參數、任務選擇、baseline 比較
  - 產生實驗報告與統計資料

- **`ablation_study.py`**
  - 自動化消融研究腳本
  - 測試不同組件對結果的影響
  - 包含 5 種消融實驗類型

- **`hyperparameter_tuning.py`**
  - 超參數自動搜尋
  - 支援 Random Search、Bayesian Optimization、Evolutionary Search
  - 找出最佳壓縮/品質平衡點

### 7. `tests/`
單元測試與功能測試

- **`test_compression.py`**
  - 測試整體壓縮流程
  - 驗證 `unified_compressor.py`

- **`test_importance_scoring.py`**
  - 測試 token 重要性評分
  - 驗證三項式公式計算

- **`test_quantization.py`**
  - 測試動態量化模組
  - 驗證精度分配與壓縮效果

### 8. `scripts/`
自動化腳本

- **`setup_environment.sh`**
  - 自動安裝依賴、建立虛擬環境
  - 設定 PYTHONPATH

- **`download_models.sh`**
  - 自動下載 LLaMA2 模型與 tokenizer
  - 處理 Hugging Face 權限
