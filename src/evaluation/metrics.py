# src/evaluation/metrics.py

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

# 假設你的 config 檔案路徑正確
from configs.base_config import CompressionConfig

class QuantizationMetrics:
    """
    計算並追蹤與串流 KV Cache 量化相關的指標。
    專注於量化位元數、Outlier 統計和估算的記憶體節省。
    """

    def __init__(self, config: CompressionConfig):
        """
        初始化指標計算器。
        Args:
            config: 包含量化設置的 CompressionConfig 物件。
        """
        self.config = config
        self.reset()

    def reset(self):
        """重置所有指標。"""
        self.layer_stats = [] # 儲存每一層的詳細統計數據
        self.total_key_elements = 0
        self.total_value_elements = 0
        self.total_estimated_bits_saved = 0
        self.total_outlier_channels_key = 0
        self.total_outlier_elements_value = 0 # Based on groups or channels
        self.total_sink_elements = 0
        self.total_quantization_time = 0 # 追蹤量化操作本身的時間 (可選)

    def update_layer_stats(
        self,
        layer_idx: int,
        key_quant_info: Dict, # 包含 bits_used, num_outliers, num_sink, shape 等
        value_quant_info: Dict, # 同上
        quantization_time: float = 0.0 # 量化這層 K/V 所花費的時間
    ):
        """
        更新單一層的量化統計數據。
        Args:
            layer_idx: 當前層索引。
            key_quant_info: 包含 Key 量化詳細資訊的字典。
                 - 'shape': (B, H, L, D) 量化後的形狀
                 - 'bits_used_tensor': [B, H, L, D] or [H, D] 每個元素/通道使用的位元數
                 - 'num_outliers': 這層 Key 被標記為 Outlier 的通道數
                 - 'num_sink': 這層 Key 中屬於 Sink Token 的元素數量
            value_quant_info: 包含 Value 量化詳細資訊的字典。
                 - 'shape': (B, H, L, D)
                 - 'bits_used_tensor': [B, H, L, D] or [H, G] or similar
                 - 'num_outliers': 這層 Value 被標記為 Outlier 的通道/分組數或元素數
                 - 'num_sink': 這層 Value 中屬於 Sink Token 的元素數量
            quantization_time: 執行該層量化的時間。
        """
        key_elements = np.prod(key_quant_info['shape'])
        value_elements = np.prod(value_quant_info['shape'])
        total_elements = key_elements + value_elements

        # --- 計算平均使用的位元數 ---
        # 假設 bits_used_tensor 包含每個元素最終使用的位元數
        avg_key_bits = key_quant_info['bits_used_tensor'].float().mean().item()
        avg_value_bits = value_quant_info['bits_used_tensor'].float().mean().item()
        avg_bits = (avg_key_bits * key_elements + avg_value_bits * value_elements) / total_elements if total_elements > 0 else 0

        # --- 計算估算的位元節省 ---
        original_bits = 16 # 假設基準是 FP16
        estimated_bits_saved = (original_bits - avg_bits) * total_elements

        layer_stat = {
            'layer_idx': layer_idx,
            'key_elements': key_elements,
            'value_elements': value_elements,
            'avg_key_bits': avg_key_bits,
            'avg_value_bits': avg_value_bits,
            'avg_bits': avg_bits,
            'key_outliers': key_quant_info.get('num_outliers', 0),
            'value_outliers': value_quant_info.get('num_outliers', 0),
            'key_sink_elements': key_quant_info.get('num_sink', 0),
            'value_sink_elements': value_quant_info.get('num_sink', 0),
            'estimated_bits_saved': estimated_bits_saved,
            'quantization_time': quantization_time,
        }

        self.layer_stats.append(layer_stat)

        # 更新全局累計值
        self.total_key_elements += key_elements
        self.total_value_elements += value_elements
        self.total_estimated_bits_saved += estimated_bits_saved
        self.total_outlier_channels_key += layer_stat['key_outliers'] # 假設 outlier 是 per-channel for key
        # Value outlier 的累計方式取決於其定義 (per group or per element)
        # 這裡假設也是累計標記數量
        self.total_outlier_elements_value += layer_stat['value_outliers']
        self.total_sink_elements += layer_stat['key_sink_elements'] + layer_stat['value_sink_elements']
        self.total_quantization_time += quantization_time

    def get_overall_metrics(self) -> Dict[str, float]:
        """計算所有層的整體量化指標。"""

        if not self.layer_stats:
            return {}

        num_layers = len(self.layer_stats)
        total_elements = self.total_key_elements + self.total_value_elements
        original_total_bits = total_elements * 16 # FP16 baseline

        overall_avg_bits = (original_total_bits - self.total_estimated_bits_saved) / total_elements if total_elements > 0 else 16
        estimated_memory_ratio = overall_avg_bits / 16.0 if original_total_bits > 0 else 1.0
        estimated_memory_savings = 1.0 - estimated_memory_ratio

        # 計算 Outlier 和 Sink 的比例
        total_key_channels = self.config.num_hidden_layers * self.config.num_key_value_heads * (self.config.hidden_size // self.config.num_attention_heads)
        # Value 的總通道/分組數計算需要根據 value_quant_groups 決定
        # 這裡簡化，只計算元素比例
        outlier_key_ratio = self.total_outlier_channels_key / total_key_channels if total_key_channels > 0 else 0
        # outlier_value_ratio = ... # 取決於 value outlier 的定義
        sink_ratio = self.total_sink_elements / total_elements if total_elements > 0 else 0

        return {
            'num_layers_processed': num_layers,
            'total_elements': total_elements,
            'overall_avg_bits': overall_avg_bits,
            'estimated_memory_ratio': estimated_memory_ratio, # 估算的壓縮後大小 / 原始大小 (基於位元數)
            'estimated_memory_savings': estimated_memory_savings, # 估算的記憶體節省比例
            'total_quantization_time': self.total_quantization_time,
            'avg_quantization_time_per_layer': self.total_quantization_time / num_layers if num_layers > 0 else 0,
            'outlier_key_channel_ratio': outlier_key_ratio, # Key Outlier 通道佔總 Key 通道的比例
            # 'outlier_value_ratio': outlier_value_ratio, # Value Outlier 比例
            'sink_token_element_ratio': sink_ratio, # Sink Token 相關元素佔總元素的比例
            # 可以加入更多層級的平均值，例如：
            'avg_layer_avg_key_bits': np.mean([s.get('avg_key_bits', 0) for s in self.layer_stats]),
            'avg_layer_avg_value_bits': np.mean([s.get('avg_value_bits', 0) for s in self.layer_stats]),
        }

    def get_layer_wise_breakdown(self) -> List[Dict]:
        """獲取詳細的逐層統計數據。"""
        return self.layer_stats.copy()

# --- Performance Timer (保持不變) ---
class PerformanceTimer:
    """用於測量性能的計時器工具。"""

    def __init__(self):
        self.start_times = {}
        self.durations = {}

    def start(self, name: str):
        """開始計時一個操作。"""
        self.start_times[name] = time.time()

    def stop(self, name: str) -> float:
        """停止計時並返回持續時間。"""
        if name not in self.start_times:
            return 0.0

        duration = time.time() - self.start_times[name]
        self.durations[name] = duration
        # Keep start time if you might restart it, or delete if one-shot
        # del self.start_times[name]
        return duration

    def get_duration(self, name: str) -> float:
        """獲取記錄的持續時間。"""
        return self.durations.get(name, 0.0)

    def get_all_durations(self) -> Dict[str, float]:
        """獲取所有記錄的持續時間。"""
        return self.durations.copy()

# --- Throughput Calculation (保持不變) ---
def calculate_throughput(
    num_tokens: int,
    total_time: float,
    batch_size: int = 1
) -> Dict[str, float]:
    """計算吞吐量指標。"""

    if total_time <= 0:
        return {'tokens_per_second': 0, 'samples_per_second': 0, 'time_per_token': 0, 'time_per_sample': 0}

    tokens_per_second = num_tokens / total_time
    samples_per_second = batch_size / total_time

    return {
        'tokens_per_second': tokens_per_second,
        'samples_per_second': samples_per_second,
        'time_per_token': total_time / num_tokens if num_tokens > 0 else 0,
        'time_per_sample': total_time / batch_size
    }

# --- Efficiency Calculation (修改為基於量化) ---
def calculate_quantization_efficiency(
    estimated_memory_savings: float, # 使用估算的節省比例
    quality_score: float, # 模型品質分數 (例如 F1, ROUGE, Perplexity loss increase)
    quantization_overhead_time: float = 0.0 # 量化操作引入的額外時間 (可選)
) -> Dict[str, float]:
    """
    計算量化效率指標。
    Args:
        estimated_memory_savings: 根據位元數估算的記憶體節省比例 (0 到 1)。
        quality_score: 模型的任務品質分數 (越高越好，基準為 baseline_score)。
                       或者可以是品質損失 (越低越好)。需要明確定義。
                       假設這裡 quality_score 是相對基準模型的比例 (例如 0.95 表示 95% 品質)。
        quantization_overhead_time: 量化操作引入的額外時間。
    """

    # 品質調整後的節省 (Quality-adjusted savings)
    # 如果 quality_score 是相對值 (0-1)，直接相乘
    quality_adjusted_savings = estimated_memory_savings * quality_score

    # 考慮時間開銷的效率 (Efficiency considering overhead)
    # 將時間開銷轉化為某種懲罰項，這裡僅作記錄
    net_efficiency_score = quality_adjusted_savings # 簡化：暫不將時間納入單一分數

    return {
        'estimated_memory_savings': estimated_memory_savings,
        'quality_score': quality_score, # 或 quality_loss
        'quality_adjusted_savings': quality_adjusted_savings,
        'quantization_overhead_time': quantization_overhead_time,
        'net_efficiency_score': net_efficiency_score # 代表結合了品質和記憶體節省的指標
    }