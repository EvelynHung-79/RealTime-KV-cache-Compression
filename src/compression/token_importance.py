import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

class PromptGuidedImportanceScorer:
    """
    Token importance scoring based on prompt-guided attention mechanism
    Implements the three-term importance formula:
    s_i^(l) = α * Â_P,i^(l) * w_l + β * b_pos(i) + γ * r(i)
    """

    def __init__(self, config):
        self.config = config
        self.alpha = config.alpha
        self.beta = config.beta  
        self.gamma = config.gamma
        self.layer_weights = config.layer_weights

    def compute_attention_aggregation(
        self, 
        attention_weights: torch.Tensor,
        prompt_indices: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Compute A_P,i^(l): attention aggregation for prompt tokens

        Args:
            attention_weights: [batch, heads, seq_len, seq_len] 
            prompt_indices: [prompt_len] indices of prompt tokens
            layer_idx: current layer index

        Returns:
            attention_scores: [batch, seq_len] aggregated attention to prompt
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape

        # Extract attention to prompt tokens: [batch, heads, seq_len, prompt_len]
        prompt_attention = attention_weights[:, :, :, prompt_indices]

        # Average across heads and sum over prompt tokens
        # [batch, seq_len]
        attention_agg = prompt_attention.mean(dim=1).sum(dim=-1)

        return attention_agg

    def normalize_attention_scores(
        self, 
        attention_scores: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Normalize attention scores within layer (z-score normalization)
        Computes Â_P,i^(l) = (A_P,i^(l) - μ_l) / σ_l

        Args:
            attention_scores: [batch, seq_len] raw attention aggregation
            layer_idx: current layer index

        Returns:
            normalized_scores: [batch, seq_len] normalized attention scores
        """
        # Z-score normalization per batch
        # mean = attention_scores.mean(dim=-1, keepdim=True)
        # std = attention_scores.std(dim=-1, keepdim=True) + 1e-8
        # normalized_scores = (attention_scores - mean) / std

        # Min-Max normalization per batch to scale scores to [0, 1]
        batch_min = attention_scores.min(dim=-1, keepdim=True)[0]
        batch_max = attention_scores.max(dim=-1, keepdim=True)[0]
        
        denominator = batch_max - batch_min
        zeros = torch.zeros_like(attention_scores)
    
        # Use torch.where to safely divide. If the denominator is close to zero,
        # use the fallback 'zeros' tensor. Otherwise, perform the division.
        normalized_scores = torch.where(
            denominator > 1e-8,
            (attention_scores - batch_min) / denominator,
            zeros
        )

        return normalized_scores

    def compute_position_bias(
        self, 
        seq_len: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute position bias compensation: b_pos(i) = log(i+1) / log(T)

        Args:
            seq_len: sequence length T
            device: target device

        Returns:
            position_bias: [seq_len] position compensation factors
        """

        if seq_len <= 1:
            # If sequence length is 1, position bias is not meaningful. Return 0.
            return torch.zeros(seq_len, device=device).float()

        positions = torch.arange(1, seq_len + 1, device=device).float()
        position_bias = torch.log(positions) / math.log(seq_len)

        return position_bias

    def compute_context_relevance(
        self, 
        seq_len: int, 
        prompt_len: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute context relevance factor: r(i) = min(1.0, N_p/N)

        Args:
            seq_len: total sequence length N  
            prompt_len: prompt length N_p
            device: target device

        Returns:
            context_factor: [seq_len] context relevance factors
        """
        relevance_factor = min(1.0, prompt_len / seq_len)
        context_relevance = torch.full((seq_len,), relevance_factor, device=device)

        return context_relevance

    def compute_importance_scores(
        self,
        attention_weights: torch.Tensor,
        prompt_indices: torch.Tensor, 
        layer_idx: int
    ) -> torch.Tensor:
        """
        Compute final importance scores using three-term formula

        Args:
            attention_weights: [batch, heads, seq_len, seq_len]
            prompt_indices: [prompt_len] prompt token indices  
            layer_idx: current layer index

        Returns:
            importance_scores: [batch, seq_len] final importance scores
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        device = attention_weights.device
        prompt_len = len(prompt_indices)

        # Term 1: Normalized prompt attention with layer weight
        attention_agg = self.compute_attention_aggregation(
            attention_weights, prompt_indices, layer_idx
        )
        normalized_attention = self.normalize_attention_scores(
            attention_agg, layer_idx  
        )
        layer_weight = self.layer_weights[layer_idx]
        term1 = self.alpha * normalized_attention * layer_weight

        # Term 2: Position bias compensation
        position_bias = self.compute_position_bias(seq_len, device)
        term2 = self.beta * position_bias.unsqueeze(0).expand(batch_size, -1)

        # Term 3: Context relevance adjustment
        context_relevance = self.compute_context_relevance(seq_len, prompt_len, device)
        term3 = self.gamma * context_relevance.unsqueeze(0).expand(batch_size, -1)

        # Final importance scores
        importance_scores = term1 + term2 + term3

        return importance_scores

class LayerWiseImportanceTracker:
    """Track and manage importance scores across all layers"""

    def __init__(self, config):
        self.config = config
        self.scorer = PromptGuidedImportanceScorer(config)
        self.layer_scores = {}

    def update_scores(
        self, 
        layer_idx: int,
        attention_weights: torch.Tensor,
        prompt_indices: torch.Tensor
    ) -> torch.Tensor:
        """Update importance scores for a specific layer"""

        scores = self.scorer.compute_importance_scores(
            attention_weights, prompt_indices, layer_idx
        )

        self.layer_scores[layer_idx] = scores.detach().cpu()

        return scores

    def get_cumulative_scores(self, layer_idx: int) -> torch.Tensor:
        """Get cumulative importance scores up to layer_idx"""

        if not self.layer_scores:
            return None

        cumulative = torch.zeros_like(self.layer_scores[0])

        for l in range(min(layer_idx + 1, len(self.layer_scores))):
            if l in self.layer_scores:
                cumulative += self.layer_scores[l]

        return cumulative / (layer_idx + 1)  # Average over layers