# src/evaluation/benchmark_runner.py (更新版)

import torch
import time
import psutil
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging # 加入日誌記錄

# 確保從正確的路徑導入 (根據您的專案結構調整)
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ..models.modified_llama import create_compressed_llama_model
from ..configs.base_config import CompressionConfig # 導入更新後的 Config
from ..utils.memory_utils import MemoryMonitor, get_model_memory_footprint
# from ..utils.eval_utils import PerformanceTimer # PerformanceTimer 移到 metrics.py 了，確認是否還需要

# Setup logging
logger = logging.getLogger(__name__)

class CompressionBenchmark:
    """
    用於串流 KVQuant 量化模型的綜合基準測試套件。
    評估不同量化配置下的性能、記憶體使用情況和量化統計數據。
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        output_dir: str = "./benchmark_results"
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.results = {}

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            logger.warning("Tokenizer does not have a pad token, using eos_token as pad_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Tokenizer loaded.")

    def create_test_configs(self) -> List[Tuple[str, CompressionConfig]]:
        """創建不同的配置場景進行測試 (基於新參數)"""

        # 基礎配置，用於修改
        base_kwargs = {
            "model_name": self.model_name,
            "chunk_size": 256,
            "ema_decay": 0.99,
            "attention_sink_size": 8,
            "outlier_threshold_relative": 5.0, # 使用相對閾值
            "outlier_threshold_abs": None,
        }

        configs = [
            # 基準線模擬 (高精度，無 Outlier 處理) - 注意: 真正的基準是未修改的模型
            ("baseline_int8", CompressionConfig(
                **base_kwargs,
                key_bits_normal=8, key_bits_sink_outlier=8,
                value_bits_normal=8, value_bits_sink_outlier=8,
                outlier_threshold_relative=None, # 關閉 Outlier 檢測
                attention_sink_size=0 # 關閉 Sink 特殊處理
            )),
             # 僅 Sink 處理 (INT8)
            ("sink_int8", CompressionConfig(
                **base_kwargs,
                key_bits_normal=8, key_bits_sink_outlier=8,
                value_bits_normal=8, value_bits_sink_outlier=8,
                outlier_threshold_relative=None # 關閉 Outlier
                # attention_sink_size 保持默認
            )),
            # 平衡 INT4 配置 (Sink/Outlier 使用 INT8)
            ("balanced_int4_sink8_outlier8", CompressionConfig(
                **base_kwargs,
                key_bits_normal=4, key_bits_sink_outlier=8,
                value_bits_normal=4, value_bits_sink_outlier=8,
            )),
            # 激進 INT4 配置 (Sink/Outlier 也是 INT4，僅依賴統計)
            ("aggressive_int4", CompressionConfig(
                **base_kwargs,
                key_bits_normal=4, key_bits_sink_outlier=4,
                value_bits_normal=4, value_bits_sink_outlier=4,
            )),
             # 較大 Chunk Size
            ("chunk512_int4_sink8_outlier8", CompressionConfig(
                **base_kwargs,
                chunk_size=512,
                key_bits_normal=4, key_bits_sink_outlier=8,
                value_bits_normal=4, value_bits_sink_outlier=8,
            )),
            # 測試 Value Group Quantization (假設 head_dim=128, 分8組, 每組16)
             ("value_group_quant", CompressionConfig(
                **base_kwargs,
                key_bits_normal=4, key_bits_sink_outlier=8,
                value_bits_normal=4, value_bits_sink_outlier=8,
                value_quant_groups=8 # 假設 hidden_size=4096, heads=32 => head_dim=128. 128/8=16.
            ))
        ]

        # 動態調整 value_quant_groups (如果模型已加載並知道 head_dim)
        # HACK: 暫時基於模型名稱猜測 head_dim
        head_dim = 128 # 默認 Llama-7b
        if "13b" in self.model_name: head_dim = 128 # Llama-13b 也是 128
        for name, config in configs:
            if config.value_quant_groups > 1:
                 if head_dim % config.value_quant_groups != 0:
                     logger.warning(f"Adjusting value_quant_groups for {name}: {head_dim} not divisible by {config.value_quant_groups}. Using per-channel.")
                     config.value_quant_groups = -1

        return configs


    def generate_test_sequences(self, lengths: List[int], count: int = 5) -> Dict[int, List[str]]:
        """生成不同長度的測試序列 (與之前相同)"""
        logger.info(f"Generating {count} test sequences for lengths: {lengths}...")
        test_sequences = {}
        words = ["test", "sequence", "quantization", "evaluation", "performance",
                 "benchmark", "memory", "efficiency", "streaming", "transformer",
                 "attention", "layer", "context", "token", "llama"]

        for length in lengths:
            sequences = []
            for i in range(count):
                # 生成大致長度的隨機文本
                num_words_approx = length // 5 # 粗略估計
                text = " ".join(np.random.choice(words, size=num_words_approx))

                # 確保達到長度要求
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                retries = 0
                while len(tokens) < length and retries < 5:
                    text += " " + " ".join(np.random.choice(words, size=max(10, (length - len(tokens)) // 5)))
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    retries += 1

                # 如果過長則截斷
                if len(tokens) > length:
                    tokens = tokens[:length]
                    text = self.tokenizer.decode(tokens, skip_special_tokens=True)

                sequences.append(text)
            test_sequences[length] = sequences
            logger.debug(f"Generated sequence for length {length}, approx tokens: {len(self.tokenizer.encode(sequences[0], add_special_tokens=False))}")
        logger.info("Test sequences generated.")
        return test_sequences

    def benchmark_single_config(
        self,
        config_name: str,
        config: CompressionConfig,
        test_sequences: Dict[int, List[str]]
    ) -> Dict[str, Any]:
        """基準測試單一配置"""

        logger.info(f"Benchmarking configuration: {config_name}")

        # 創建模型
        try:
            model = create_compressed_llama_model(self.model_name, config, self.device)
            model.eval()
        except Exception as e:
            logger.error(f"Failed to create model for config {config_name}: {e}", exc_info=True)
            return {'error': f"Model creation failed: {e}"}

        # 獲取模型記憶體佔用
        model_memory = get_model_memory_footprint(model)
        logger.info(f"Model memory footprint: {model_memory}")

        config_results = {
            'config_name': config_name,
            'config': config.__dict__.copy(), # 保存配置副本
            'model_memory': model_memory,
            'sequence_results': {}
        }

        # 測試每個序列長度
        for seq_len, sequences in test_sequences.items():
            logger.info(f"  Testing sequence length: {seq_len}")

            # 檢查模型是否支持此長度
            if seq_len > config.max_position_embeddings:
                 logger.warning(f"Skipping seq_len {seq_len}: exceeds model max_position_embeddings ({config.max_position_embeddings})")
                 continue

            seq_results = {
                'sequence_length': seq_len,
                'num_sequences': len(sequences),
                'measurements': []
            }

            # 測量每個序列
            for i, text in enumerate(tqdm(sequences, desc=f"Seq {seq_len}")):
                try:
                    # 重置狀態
                    if hasattr(model, 'reset_compression_state'):
                        model.reset_compression_state()

                    result = self.measure_single_inference(model, text, config, seq_len)
                    seq_results['measurements'].append(result)

                except torch.cuda.OutOfMemoryError as oom:
                     logger.error(f"    OOM Error processing sequence {i} (length {seq_len}) for config {config_name}. Skipping remaining sequences for this length.")
                     torch.cuda.empty_cache() # 清理顯存
                     # 可以在 seq_results 中標記 OOM
                     seq_results['oom_error'] = True
                     break # 跳過此長度的剩餘序列
                except Exception as e:
                    logger.error(f"    Error processing sequence {i} (length {seq_len}): {e}", exc_info=True)
                    # 可以在 result 中記錄錯誤，而不是跳過
                    seq_results['measurements'].append({'error': str(e)})
                    continue # 繼續下一個序列

            # 聚合此序列長度的結果
            valid_measurements = [m for m in seq_results['measurements'] if 'error' not in m]
            if valid_measurements:
                seq_results['aggregated'] = self.aggregate_measurements(valid_measurements)
            else:
                 seq_results['aggregated'] = {} # 確保有 aggregated 鍵

            # 添加 OOM 標記到聚合結果
            if seq_results.get('oom_error', False):
                 seq_results['aggregated']['oom_occurred'] = True


            config_results['sequence_results'][seq_len] = seq_results

        # 清理模型
        logger.info(f"Finished benchmarking {config_name}. Cleaning up.")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return config_results

    def measure_single_inference(
        self,
        model,
        text: str,
        config: CompressionConfig,
        expected_seq_len: int # 傳入預期長度以截斷
    ) -> Dict[str, Any]:
        """測量單次推理的性能"""

        memory_monitor = MemoryMonitor(interval=0.05) # 更快的監控間隔
        # timer = PerformanceTimer() # 假設 PerformanceTimer 在 metrics.py

        # --- Tokenize 輸入 ---
        # 確保截斷到配置的最大長度或預期長度中的較小者
        max_len = min(config.max_position_embeddings, expected_seq_len)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len # 使用計算出的 max_len
        )
        input_length = inputs['input_ids'].shape[1]

        # 如果 tokenizer 截斷後長度不符，記錄警告
        if input_length < expected_seq_len and input_length < config.max_position_embeddings:
            logger.warning(f"Input sequence truncated by tokenizer to {input_length}, expected {expected_seq_len}.")

        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # --- 開始監控 ---
        memory_monitor.start_monitoring()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        prefill_start_time = time.perf_counter() # 使用 perf_counter

        # --- Prefill (第一次 forward pass) ---
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, return_dict=True) # 使用 return_dict

        if torch.cuda.is_available(): torch.cuda.synchronize()
        prefill_time = time.perf_counter() - prefill_start_time
        # ttft = prefill_time # TTFT 約等於 prefill 時間

        # --- Generation ---
        generation_start_time = time.perf_counter()
        MAX_NEW_TOKENS = 50 # 固定生成 token 數用於比較

        with torch.no_grad():
            # 使用 outputs.past_key_values 作為輸入
            generated = model.generate(
                inputs['input_ids'], # 僅傳遞 input_ids 可能更標準
                attention_mask=inputs['attention_mask'],
                past_key_values=outputs.past_key_values, # 從 prefill 獲取 cache
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        if torch.cuda.is_available(): torch.cuda.synchronize()
        generation_time = time.perf_counter() - generation_start_time

        # --- 停止監控 ---
        memory_stats = memory_monitor.get_peak_memory()
        memory_monitor.stop_monitoring()

        # --- 獲取量化統計數據 ---
        quant_stats = {}
        if hasattr(model, 'get_compression_stats'):
            try:
                quant_stats = model.get_compression_stats() # 現在返回估計值和 outlier count
            except Exception as e:
                logger.warning(f"Failed to get compression stats: {e}")

        # --- 計算指標 ---
        output_length = generated.shape[1] - input_length
        # 確保 output_length 不超過 MAX_NEW_TOKENS
        output_length = min(output_length, MAX_NEW_TOKENS)

        total_time = prefill_time + generation_time
        tokens_per_second = output_length / generation_time if generation_time > 0 else 0

        # 計算理論記憶體節省 (基於配置)
        bits_k = (config.key_bits_normal + config.key_bits_sink_outlier)/2 # 粗略平均
        bits_v = (config.value_bits_normal + config.value_bits_sink_outlier)/2 # 粗略平均
        original_bits = 16 # FP16
        estimated_ratio = (bits_k + bits_v) / (original_bits * 2)
        estimated_savings = 1.0 - estimated_ratio

        return {
            'input_length': input_length,
            'output_length': output_length,
            'prefill_time': prefill_time, # TTFT is essentially prefill time
            'generation_time': generation_time,
            'total_time': total_time,
            'tokens_per_second': tokens_per_second, # Throughput during generation
            'memory_stats': memory_stats,
            'quantization_stats': quant_stats, # 重命名
            'estimated_memory_savings': estimated_savings # 添加理論值
        }

    def aggregate_measurements(self, measurements: List[Dict]) -> Dict[str, Any]:
        """聚合多次運行的測量結果"""

        if not measurements:
            return {}

        aggregated = {}
        numeric_fields = [
            'input_length', 'output_length', 'prefill_time', 'generation_time',
            'total_time', 'tokens_per_second', 'estimated_memory_savings'
        ]

        # 聚合數值指標
        for field in numeric_fields:
            values = [m.get(field, 0) for m in measurements]
            if values:
                aggregated[f'avg_{field}'] = np.mean(values)
                aggregated[f'std_{field}'] = np.std(values)
                aggregated[f'min_{field}'] = np.min(values)
                aggregated[f'max_{field}'] = np.max(values)
                aggregated[f'median_{field}'] = np.median(values)
            else:
                 aggregated[f'avg_{field}'] = 0 # Handle empty list

        # 聚合記憶體統計
        memory_stats_list = [m.get('memory_stats') for m in measurements if m.get('memory_stats')]
        if memory_stats_list:
            mem_fields = ['peak_cpu_memory_mb', 'avg_cpu_memory_mb',
                          'peak_gpu_allocated_mb', 'avg_gpu_allocated_mb',
                          'peak_gpu_reserved_mb', 'avg_gpu_reserved_mb']
            for field in mem_fields:
                values = [ms.get(field, 0) for ms in memory_stats_list if ms] # Check if ms exists
                if values:
                    aggregated[f'avg_{field}'] = np.mean(values)
                    aggregated[f'max_{field}'] = np.max(values) # Often max peak is most relevant

        # 聚合量化統計 (新指標)
        quant_stats_list = [m.get('quantization_stats') for m in measurements if m.get('quantization_stats')]
        if quant_stats_list:
            # 示例：聚合 outlier 數量
            key_outliers = [qs.get('key_outliers_detected', 0) for qs in quant_stats_list if qs]
            val_outliers = [qs.get('value_outliers_detected', 0) for qs in quant_stats_list if qs]
            if key_outliers:
                aggregated['avg_key_outliers'] = np.mean(key_outliers)
            if val_outliers:
                aggregated['avg_value_outliers'] = np.mean(val_outliers)
            # 可以添加更多從 get_compression_stats() 返回的聚合指標

        return aggregated


    def run_full_benchmark(
        self,
        sequence_lengths: List[int] = [1024, 2048, 4096, 8192],
        sequences_per_length: int = 5
    ) -> Dict[str, Any]:
        """運行所有配置的綜合基準測試"""

        logger.info("Starting comprehensive quantization benchmark...")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Sequence lengths: {sequence_lengths}")
        logger.info(f"Sequences per length: {sequences_per_length}")

        test_sequences = self.generate_test_sequences(sequence_lengths, sequences_per_length)
        configs = self.create_test_configs()

        benchmark_results = {
            'model_name': self.model_name,
            'test_setup': {
                'sequence_lengths': sequence_lengths,
                'sequences_per_length': sequences_per_length,
                'device': self.device
            },
            'config_results': {}
        }

        for config_name, config in configs:
            try:
                config_result = self.benchmark_single_config(config_name, config, test_sequences)
                benchmark_results['config_results'][config_name] = config_result
                self.save_results(benchmark_results, f"intermediate_benchmark_results.json") # 保存中間結果
            except Exception as e:
                logger.error(f"Critical error benchmarking {config_name}: {e}", exc_info=True)
                benchmark_results['config_results'][config_name] = {'error': f"Critical failure: {e}"}

        self.save_results(benchmark_results, "comprehensive_benchmark_results.json")
        self.analyze_results(benchmark_results)

        logger.info("Comprehensive benchmark finished.")
        return benchmark_results

    def save_results(self, results: Dict, filename: str):
        """將結果保存到 JSON 文件"""
        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, 'w') as f:
                # 使用自訂轉換函數處理無法序列化的類型 (例如 dataclass)
                json.dump(results, f, indent=2, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
            logger.info(f"Results saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results to {filepath}: {e}")


    def analyze_results(self, results: Dict):
        """生成分析和視覺化圖表"""
        logger.info("Generating analysis...")
        try:
            config_names = list(results.get('config_results', {}).keys())
            sequence_lengths = results.get('test_setup', {}).get('sequence_lengths', [])

            if not config_names or not sequence_lengths:
                 logger.warning("No valid config results or sequence lengths found for analysis.")
                 return

            # 創建比較圖表 (更新指標)
            self.plot_performance_comparison(results, config_names, sequence_lengths)
            self.plot_memory_usage_comparison(results, config_names, sequence_lengths)
            self.plot_quantization_efficiency(results, config_names, sequence_lengths) # 重命名

            # 生成摘要報告 (更新指標)
            self.generate_summary_report(results)
            logger.info("Analysis generated.")
        except Exception as e:
            logger.error(f"Failed to generate analysis: {e}", exc_info=True)

    def plot_performance_comparison(self, results: Dict, config_names: List[str], seq_lengths: List[int]):
        """繪製不同配置的性能比較圖"""
        # (繪圖代碼基本不變，只需確認提取的 key 正確)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6)) # 改為一行兩列: Prefill Time, Tokens/Second
        ax1, ax2 = axes

        # Prefill Time (TTFT)
        for config_name in config_names:
            config_res = results.get('config_results', {}).get(config_name, {})
            if 'error' in config_res: continue
            perf_values = []
            valid_lens = []
            for seq_len in seq_lengths:
                seq_res = config_res.get('sequence_results', {}).get(seq_len, {})
                if seq_res and 'aggregated' in seq_res and not seq_res['aggregated'].get('oom_occurred', False):
                    perf = seq_res['aggregated'].get('avg_prefill_time', 0)
                    perf_values.append(perf)
                    valid_lens.append(seq_len)
            if valid_lens: ax1.plot(valid_lens, perf_values, marker='o', label=config_name)

        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Prefill Time (seconds)')
        ax1.set_title('Prefill Time (Approx TTFT)')
        ax1.legend(fontsize='small')
        ax1.grid(True)
        ax1.set_yscale('log') # Prefill 時間隨長度增長通常超線性

        # Tokens per second (Generation)
        for config_name in config_names:
            config_res = results.get('config_results', {}).get(config_name, {})
            if 'error' in config_res: continue
            perf_values = []
            valid_lens = []
            for seq_len in seq_lengths:
                 seq_res = config_res.get('sequence_results', {}).get(seq_len, {})
                 if seq_res and 'aggregated' in seq_res and not seq_res['aggregated'].get('oom_occurred', False):
                    perf = seq_res['aggregated'].get('avg_tokens_per_second', 0)
                    perf_values.append(perf)
                    valid_lens.append(seq_len)
            if valid_lens: ax2.plot(valid_lens, perf_values, marker='o', label=config_name)

        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Tokens/Second (Generation)')
        ax2.set_title('Generation Throughput')
        ax2.legend(fontsize='small')
        ax2.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'performance_comparison.png')
        try:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance comparison plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to save performance plot: {e}")
        plt.close()


    def plot_memory_usage_comparison(self, results: Dict, config_names: List[str], seq_lengths: List[int]):
        """繪製記憶體使用比較圖"""
        # (繪圖代碼基本不變)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) # Peak GPU Allocated, Peak GPU Reserved

        # Peak GPU Allocated Memory
        for config_name in config_names:
            config_res = results.get('config_results', {}).get(config_name, {})
            if 'error' in config_res: continue
            mem_values = []
            valid_lens = []
            for seq_len in seq_lengths:
                seq_res = config_res.get('sequence_results', {}).get(seq_len, {})
                if seq_res and 'aggregated' in seq_res and not seq_res['aggregated'].get('oom_occurred', False):
                    mem = seq_res['aggregated'].get('max_peak_gpu_allocated_mb', 0)
                    mem_values.append(mem)
                    valid_lens.append(seq_len)
            if valid_lens: ax1.plot(valid_lens, mem_values, marker='o', label=config_name)

        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Peak GPU Allocated (MB)')
        ax1.set_title('Peak GPU Allocated Memory')
        ax1.legend(fontsize='small')
        ax1.grid(True)

        # Peak GPU Reserved Memory
        for config_name in config_names:
            config_res = results.get('config_results', {}).get(config_name, {})
            if 'error' in config_res: continue
            mem_values = []
            valid_lens = []
            for seq_len in seq_lengths:
                seq_res = config_res.get('sequence_results', {}).get(seq_len, {})
                if seq_res and 'aggregated' in seq_res and not seq_res['aggregated'].get('oom_occurred', False):
                    mem = seq_res['aggregated'].get('max_peak_gpu_reserved_mb', 0)
                    mem_values.append(mem)
                    valid_lens.append(seq_len)
            if valid_lens: ax2.plot(valid_lens, mem_values, marker='o', label=config_name)

        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Peak GPU Reserved (MB)')
        ax2.set_title('Peak GPU Reserved Memory')
        ax2.legend(fontsize='small')
        ax2.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'memory_comparison.png')
        try:
             plt.savefig(plot_path, dpi=300, bbox_inches='tight')
             logger.info(f"Memory comparison plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to save memory plot: {e}")
        plt.close()

    def plot_quantization_efficiency(self, results: Dict, config_names: List[str], seq_lengths: List[int]):
        """繪製量化效率指標圖 (例如：估計的記憶體節省)"""
        # (繪圖代碼修改，不再繪製 compression ratio)
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6)) # 只畫 Estimated Memory Savings

        for config_name in config_names:
            config_res = results.get('config_results', {}).get(config_name, {})
            if 'error' in config_res: continue
            savings_values = []
            valid_lens = []
            for seq_len in seq_lengths:
                seq_res = config_res.get('sequence_results', {}).get(seq_len, {})
                if seq_res and 'aggregated' in seq_res and not seq_res['aggregated'].get('oom_occurred', False):
                    # 使用聚合結果中的理論節省值
                    savings = seq_res['aggregated'].get('avg_estimated_memory_savings', 0) * 100 # 轉為百分比
                    savings_values.append(savings)
                    valid_lens.append(seq_len)
            if valid_lens: ax1.plot(valid_lens, savings_values, marker='o', label=config_name)

        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Estimated Memory Savings (%)')
        ax1.set_title('Estimated KV Cache Memory Savings (Based on Bit Config)')
        ax1.legend(fontsize='small')
        ax1.grid(True)
        ax1.set_ylim(bottom=0) # 節省率通常 > 0

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'quantization_efficiency.png')
        try:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Quantization efficiency plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to save quantization plot: {e}")
        plt.close()


    def generate_summary_report(self, results: Dict):
        """生成文本摘要報告 (更新指標)"""
        report_path = os.path.join(self.output_dir, 'benchmark_summary.md')

        with open(report_path, 'w') as f:
            f.write(f"# Quantization Benchmark Summary\n\n")
            f.write(f"**Model:** {results.get('model_name', 'N/A')}\n")
            f.write(f"**Test Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Configuration Performance Overview\n\n")
            f.write("| Configuration | Avg Prefill Time (s) | Avg Gen Throughput (tok/s) | Peak GPU Alloc (MB) | Est. Mem Savings (%) | Avg Key Outliers | Avg Val Outliers |\n")
            f.write("|---|---|---|---|---|---|---|\n")

            for config_name, config_results in results.get('config_results', {}).items():
                if 'error' in config_results:
                    f.write(f"| {config_name} | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |\n")
                    continue

                # 找一個代表性的序列長度 (例如最長的成功運行的)
                rep_len = 0
                rep_agg = None
                valid_lens = sorted([sl for sl, sr in config_results.get('sequence_results', {}).items()
                                     if sr and 'aggregated' in sr and not sr['aggregated'].get('oom_occurred', False)], reverse=True)
                if valid_lens:
                    rep_len = valid_lens[0]
                    rep_agg = config_results['sequence_results'][rep_len]['aggregated']

                if rep_agg:
                    prefill = rep_agg.get('avg_prefill_time', 0)
                    tps = rep_agg.get('avg_tokens_per_second', 0)
                    mem = rep_agg.get('max_peak_gpu_allocated_mb', 0)
                    savings = rep_agg.get('avg_estimated_memory_savings', 0) * 100
                    k_out = rep_agg.get('avg_key_outliers', 0)
                    v_out = rep_agg.get('avg_value_outliers', 0)
                    f.write(f"| {config_name} | {prefill:.3f} | {tps:.1f} | {mem:.0f} | {savings:.1f}% | {k_out:.1f} | {v_out:.1f} |\n")
                else:
                    # 如果所有長度都 OOM 或出錯
                    oom_note = " (OOM/Error)" if any(sr.get('aggregated',{}).get('oom_occurred', False) or 'error' in sr for sr in config_results.get('sequence_results', {}).values()) else ""
                    f.write(f"| {config_name} | N/A{oom_note} | N/A | N/A | N/A | N/A | N/A |\n")

        logger.info(f"Summary report saved to: {report_path}")

# --- 主執行函數 ---
def run_benchmark_suite(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    sequence_lengths: List[int] = [1024, 2048, 4096],
    sequences_per_length: int = 3,
    output_dir: str = "./benchmark_results",
    device: str = "cuda" # 允許傳遞 device
):
    """運行完整的基準測試套件"""
    # 設置頂層日誌
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA not available, switching to CPU. This will be very slow.")
        device = "cpu"

    benchmark = CompressionBenchmark(model_name, device=device, output_dir=output_dir)
    results = benchmark.run_full_benchmark(sequence_lengths, sequences_per_length)

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Streaming KVQuant Benchmark")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--seq_lengths", type=int, nargs='+', default=[1024, 2048, 4096], help="Sequence lengths to test")
    parser.add_argument("--num_seq", type=int, default=3, help="Number of sequences per length")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", help="Directory to save results")
    args = parser.parse_args()

    results = run_benchmark_suite(
        model_name=args.model_name,
        sequence_lengths=args.seq_lengths,
        sequences_per_length=args.num_seq,
        output_dir=args.output_dir,
        device=args.device
    )
    logger.info("Benchmark suite finished.")