import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np

class DynamicPrecisionQuantizer:
    """
    Dynamic precision assignment and quantization based on token importance scores
    Implements multi-bit quantization: 8-bit (HIGH), 4-bit (MID), 2-bit (LOW)
    """

    def __init__(self, config):
        self.config = config
        self.theta_h = config.theta_h  # High precision threshold
        self.theta_m = config.theta_m  # Medium precision threshold
        self.high_bits = config.high_precision_bits
        self.medium_bits = config.medium_precision_bits
        self.low_bits = config.low_precision_bits

    def assign_precision_levels(
        self, 
        importance_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Assign precision levels based on importance scores

        Args:
            importance_scores: [batch, seq_len] importance scores

        Returns:
            precision_labels: [batch, seq_len] precision assignments (0=LOW, 1=MID, 2=HIGH)
            precision_stats: dict with count statistics
        """
        batch_size, seq_len = importance_scores.shape

        # Initialize precision labels (0=LOW by default)
        precision_labels = torch.zeros_like(importance_scores, dtype=torch.long)

        # Assign precision levels based on thresholds
        high_mask = importance_scores >= self.theta_h
        medium_mask = (importance_scores >= self.theta_m) & (importance_scores < self.theta_h)

        precision_labels[high_mask] = 2    # HIGH precision
        precision_labels[medium_mask] = 1  # MEDIUM precision
        # LOW precision tokens remain 0

        # Calculate statistics
        total_tokens = batch_size * seq_len
        precision_stats = {
            'high_count': high_mask.sum().item(),
            'medium_count': medium_mask.sum().item(), 
            'low_count': total_tokens - high_mask.sum().item() - medium_mask.sum().item(),
            'high_ratio': high_mask.sum().item() / total_tokens,
            'medium_ratio': medium_mask.sum().item() / total_tokens,
            'low_ratio': (total_tokens - high_mask.sum().item() - medium_mask.sum().item()) / total_tokens
        }
        # print(f"[DEBUG] Importance Scores (min/max/mean): {importance_scores.min():.4f}/{importance_scores.max():.4f}/{importance_scores.mean():.4f}, Precision Stats: {precision_stats}")
        
        return precision_labels, precision_stats

    def get_quantization_params(
        self, 
        tensor: torch.Tensor, 
        num_bits: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute quantization parameters (scale and zero_point) for uniform quantization. This is for by token quantization. For per-channel quantization, use get_quantization_params_per_channel.

        Args:
            tensor: input tensor to quantize
            num_bits: target bit width

        Returns:
            scale: quantization scale factor
            zero_point: quantization zero point
        """
        # Compute min/max along feature dimension
        t_min = tensor.min()
        t_max = tensor.max()

        # Avoid division by zero
        if t_max == t_min:
            scale = torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype)
            zero_point = torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
            return scale, zero_point

        # Compute quantization parameters
        qmin = 0
        qmax = (2 ** num_bits) - 1

        scale = (t_max - t_min) / (qmax - qmin)
        zero_point = qmin - t_min / scale

        return scale, zero_point

    def get_quantization_params_per_channel(
        self,
        tensor: torch.Tensor, # 輸入 Tensor，例如 shape [N, features] 或 [B, S, features]
        num_bits: int,
        feature_dim: int = -1 # 代表 channel/feature 的維度
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """計算 Per-Channel 的量化參數 (scale 和 zero_point)。"""
        # 確保 feature_dim 是正數索引
        actual_feature_dim = feature_dim if feature_dim >= 0 else tensor.dim() + feature_dim
        # 找出需要被 reduce (計算 min/max) 的維度 (除了 feature 維度之外的所有維度)
        dims_to_reduce = [d for d in range(tensor.dim()) if d != actual_feature_dim]

        if not dims_to_reduce: # 如果 tensor 只有 1 維 (代表只有 feature 維度)
            t_min = tensor
            t_max = tensor
        else:
            # 沿著非 feature 維度計算 min/max，結果 tensor 的 shape 會是 [features]
            t_min = tensor.amin(dim=dims_to_reduce)
            t_max = tensor.amax(dim=dims_to_reduce)

        # 計算每個 channel 的量化參數
        qmin = 0
        qmax = (2 ** num_bits) - 1

        # 計算 scale，處理 range 為 0 的情況
        range_val = t_max - t_min
        # 使用 torch.where 避免除以零或極小值
        scale = torch.where(range_val > 1e-8, range_val / (qmax - qmin), torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype))

        # 計算 zero_point，處理 scale 為 0 的情況
        zero_point = qmin - t_min / torch.where(scale > 1e-8, scale, torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype))
        # 四捨五入 zero_point 到整數可能是常見做法，但這裡保持浮點數以簡化
        # zero_point = torch.round(qmin - t_min / scale)

        # 確保 scale 和 zero_point shape 為 [features] (amin/amax 應該已保證)
        return scale, zero_point

    def quantize_tensor(
        self,
        tensor: torch.Tensor, # 輸入 tensor，shape 例如 [features] 或 [B, S, features]
        num_bits: int,
        scale: torch.Tensor,    # 可以是 scalar 或 shape [features]
        zero_point: torch.Tensor # 可以是 scalar 或 shape [features]
    ) -> torch.Tensor:
        """
        對 tensor 應用均勻量化。可處理 scalar 或 per-channel 參數。
        """
        qmin = 0
        qmax = (2 ** num_bits) - 1

        # 確保 scale 和 zero_point 的 shape 可以 broadcast
        # 如果是 per-channel, scale/zp 是 [features], tensor 是 [..., features]
        # broadcasting 會自動處理

        # 量化
        # 使用 clamp 前先 round，避免浮點誤差導致 clamp 範圍外的數值
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)

        # 反量化回原始範圍 (用於模擬量化誤差)
        dequantized = (quantized - zero_point) * scale

        return dequantized

    def apply_mixed_precision_quantization(
        self,
        key_states: torch.Tensor,      # Shape: [batch, seq_len, head_dim]
        value_states: torch.Tensor,    # Shape: [batch, seq_len, head_dim]
        precision_labels: torch.Tensor # Shape: [batch, seq_len]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        根據 precision_labels 應用混合精度量化。
        Keys: Per-Channel 量化
        Values: Per-Token 量化
        """
        batch_size, seq_len, head_dim = key_states.shape
        device = key_states.device

        quantized_keys = torch.zeros_like(key_states)
        quantized_values = torch.zeros_like(value_states)

        # 儲存量化資訊 (選擇性，可能有助於分析)
        quant_info = {
            'scales_k': {},          # 儲存每個精度級別的 per-channel key scales
            'zero_points_k': {},     # 儲存每個精度級別的 per-channel key zero points
            # 'scales_v': {},        # Per-token value scales (可能非常大，暫不儲存)
            # 'zero_points_v': {},   # Per-token value zero points (可能非常大，暫不儲存)
            'bit_assignments': precision_labels.detach().cpu().numpy() # 記錄每個 token 的位元分配
        }

        # 分別處理每個精度級別 (LOW=0, MID=1, HIGH=2)
        for precision_level in [0, 1, 2]:
            # 找出屬於當前精度級別的 token 的 mask
            mask = (precision_labels == precision_level) # Shape [batch, seq_len]
            if not mask.any(): # 如果沒有 token 屬於此級別，則跳過
                continue

            # --- Key 量化 (Per-Channel) ---
            if precision_level == 0: num_bits_k = self.low_bits
            elif precision_level == 1: num_bits_k = self.medium_bits
            else: num_bits_k = self.high_bits # HIGH

            # 選取出所有屬於當前精度級別的 key state
            # keys_subset 的 shape 會是 [N, head_dim]，N 是屬於此級別的 token 總數
            keys_subset = key_states[mask]

            if keys_subset.numel() > 0: # 確保有選出 token
                 # **核心修改**: 計算 Per-Channel 的 scale 和 zero point (只計算一次)
                 # k_scale_ch 和 k_zp_ch 的 shape 都是 [head_dim]
                 k_scale_ch, k_zp_ch = self.get_quantization_params_per_channel(keys_subset, num_bits_k, feature_dim=-1)

                 # (選擇性) 儲存計算出的參數
                 quant_info['scales_k'][precision_level] = k_scale_ch.detach().cpu().numpy()
                 quant_info['zero_points_k'][precision_level] = k_zp_ch.detach().cpu().numpy()

                 # 遍歷屬於此級別的 token 索引，應用 Per-Channel 量化
                 indices = torch.nonzero(mask, as_tuple=False) # 取得 [batch_idx, seq_idx] 列表
                 for b, s in indices:
                     # 使用之前計算好的 k_scale_ch, k_zp_ch (shape [head_dim])
                     quantized_keys[b, s, :] = self.quantize_tensor(
                         key_states[b, s, :], num_bits_k, k_scale_ch, k_zp_ch
                     )
            # --- Key 量化結束 ---

            # --- Value 量化 (Per-Token) ---
            if precision_level == 0: num_bits_v = self.low_bits
            elif precision_level == 1: num_bits_v = self.medium_bits
            else: num_bits_v = self.high_bits # HIGH

            # **保持不變**: 這部分邏輯已經是 Per-Token
            indices_v = torch.nonzero(mask, as_tuple=False) # 再次取得索引以保持清晰
            for b, s in indices_v:
                 # 取出單一 token 的 value state，保持維度 [1, 1, head_dim]
                 v_token = value_states[b:b+1, s:s+1, :]
                 # **核心**: 計算這個 token 自己的 SCALAR scale 和 zero point
                 v_scale_tok, v_zp_tok = self.get_quantization_params(v_token, num_bits_v) # 調用原始的 scalar 版本
                 # 使用 scalar 參數進行量化
                 quantized_values[b, s, :] = self.quantize_tensor(
                     value_states[b, s, :], num_bits_v, v_scale_tok, v_zp_tok
                 )
                 # 注意：如果需要儲存 v_scale_tok, v_zp_tok，quant_info 會變得很大
            # --- Value 量化結束 ---

        return quantized_keys, quantized_values, quant_info

    def estimate_memory_savings(
        self, 
        original_tensor: torch.Tensor,
        precision_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Estimate memory savings from mixed precision quantization

        Args:
            original_tensor: original tensor (assumed fp16)
            precision_labels: precision assignments

        Returns:
            memory_info: dictionary with memory statistics
        """
        total_elements = original_tensor.numel()

        # Count elements by precision
        high_elements = (precision_labels == 2).sum().item() * original_tensor.shape[-1]
        medium_elements = (precision_labels == 1).sum().item() * original_tensor.shape[-1]
        low_elements = (precision_labels == 0).sum().item() * original_tensor.shape[-1]

        # Calculate memory usage (in bytes)
        # Original: 16 bits = 2 bytes per element
        original_memory = total_elements * 2

        compressed_memory = (
            high_elements * (self.high_bits / 8) +
            medium_elements * (self.medium_bits / 8) + 
            low_elements * (self.low_bits / 8)
        )

        compression_ratio = compressed_memory / original_memory
        memory_savings = 1.0 - compression_ratio

        return {
            'original_memory_mb': original_memory / (1024 * 1024),
            'compressed_memory_mb': compressed_memory / (1024 * 1024), 
            'compression_ratio': compression_ratio,
            'memory_savings': memory_savings,
            'high_elements_ratio': high_elements / total_elements,
            'medium_elements_ratio': medium_elements / total_elements,
            'low_elements_ratio': low_elements / total_elements
        }
    