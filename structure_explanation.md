# 資料夾與檔案說明

以下依據 repo 結構，逐一解釋每個資料夾與檔案的功能、彼此關聯，以及如何使用這些 code 來完成你的 real-time prefill KV cache compression 方案。

***

## 1. `configs/`
- **base_config.py**
  - 儲存所有壓縮相關的超參數（如 layer 傳播比例、精度閾值、重要性權重等）。
  - 供主程式與各模組讀取，確保所有流程參數一致。

***

## 2. `src/compression/`
- **token_importance.py**
  - 實作 token 重要性評分公式（prompt-guided三項式），可根據 config 調整權重。
  - 提供 API 給壓縮主流程呼叫。
- **dynamic_quantization.py**
  - 根據重要性分數分配不同量化精度（2/4/8 bit），並執行量化。
  - 與 token_importance.py 串接，根據分數自動選擇精度。
- **selective_propagation.py**
  - 根據重要性分數與層級預算，選擇要傳播到下一層的 tokens。
  - 支援 per-layer ratio 與 drop 機制。
- **compression_layers.py**
  - 將上述三個模組整合，實現 layer-wise KV cache 壓縮主流程。
  - 提供一個 `compress_kv_cache()` 介面給主程式呼叫。
- **unified_compressor.py**
  - 將所有壓縮步驟包裝成一個高階類別，方便主程式或 benchmark runner 直接調用。

***

## 3. `src/models/`
- **modified_llama.py**
  - 對 LLaMA2 模型進行 hook，支援在每層插入壓縮流程。
  - 提供 KV cache 讀寫、壓縮、回傳等功能。

***

## 4. `src/evaluation/`
- **longbench_eval.py**
  - 載入 LongBench 資料集，執行推理並計算各項指標（F1、ROUGE-L、EM等）。
  - 支援多任務、長文本測試。
- **metrics.py**
  - 實作各種評估指標計算函數。
- **benchmark_runner.py**
  - 負責整合模型、壓縮器、資料集，批次執行實驗並收集結果。
  - 可指定 baseline、ablation、不同超參數組合。

***

## 5. `src/utils/`
- **data_utils.py**
  - 資料預處理、tokenization、分批載入等工具。
  - 提供給 longbench_eval.py 與主程式使用。

***

## 6. `experiments/`
- **run_compression_experiment.py**
  - 主實驗腳本，串接所有模組，執行完整壓縮流程與評估。
  - 可指定模型、超參數、任務、輸出路徑等。
- **ablation_study.py**
  - 自動化消融研究腳本，測試不同組合（如只用 importance、只用 quantization 等）對結果的影響。
- **hyperparameter_tuning.py**
  - 超參數自動搜尋（如用 Optuna），找出最佳壓縮/品質平衡點。

***

## 7. `scripts/`
- **run_longbench.sh**
  - Shell script，一鍵執行完整 LongBench 壓縮評估流程。
  - 支援 baseline、ablation、超參數調整等選項。
- **setup_environment.sh**
  - 自動安裝依賴、建立虛擬環境、下載必要資源。
- **download_models.sh**
  - 自動下載 LLaMA2、LongBench 資料集等。

***

## 8. `tests/`
- **test_importance_scoring.py**
  - 單元測試 token_importance.py，確保重要性分數計算正確。
- **test_quantization.py**
  - 單元測試 dynamic_quantization.py，驗證量化精度分配與壓縮效果。

***

# 檔案間的關聯
- `run_compression_experiment.py` 是主入口，會呼叫 unified_compressor.py，後者再串接 token_importance.py、dynamic_quantization.py、selective_propagation.py、compression_layers.py。
- `modified_llama.py` 提供模型 KV cache hook，讓壓縮流程能嵌入 transformer 層。
- `longbench_eval.py` 會用 data_utils.py 處理資料，然後呼叫模型與壓縮器，最後用 metrics.py 計算指標。
- `benchmark_runner.py` 可批次執行多組實驗，收集結果。
- `ablation_study.py`、`hyperparameter_tuning.py` 進行消融與參數搜尋，皆會呼叫主壓縮流程與評估模組。
- 測試檔案確保每個核心模組獨立正確。
- shell script 負責自動化環境建置與批次執行。

# 使用流程建議
1. 先執行 `setup_environment.sh` 安裝依賴。
2. 用 `download_models.sh` 下載模型與資料集。
3. 用 `run_longbench.sh` 或 `run_compression_experiment.py` 執行主實驗。
4. 若要消融或參數搜尋，執行 `ablation_study.py` 或 `hyperparameter_tuning.py`。
5. 用 `benchmark_runner.py` 批次比較不同方法。
6. 用 `tests/` 下的測試檔驗證每個模組。
