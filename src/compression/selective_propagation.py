import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional
import math

class SelectiveTokenPropagator:
    """
    Selective token propagation mechanism inspired by FastKV
    Implements token selection with budget constraints and layer-specific ratios
    """

    def __init__(self, config):
        self.config = config
        self.early_layer_ratio = config.early_layer_ratio
        self.middle_layer_ratio = config.middle_layer_ratio
        self.later_layer_ratio = config.later_layer_ratio

        # Define layer boundaries
        total_layers = config.num_hidden_layers
        self.early_boundary = int(0.3 * total_layers)
        self.middle_boundary = int(0.7 * total_layers)

    def get_layer_propagation_ratio(self, layer_idx: int) -> float:
        """
        Get propagation ratio for specific layer based on layer groups

        Args:
            layer_idx: current layer index

        Returns:
            propagation_ratio: ratio of tokens to propagate to next layer
        """
        if layer_idx < self.early_boundary:
            return self.early_layer_ratio
        elif layer_idx < self.middle_boundary:
            return self.middle_layer_ratio
        else:
            return self.later_layer_ratio

    def compute_token_costs(
        self, 
        precision_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cost for each token based on its precision assignment

        Args:
            precision_labels: [batch, seq_len] precision assignments (0=LOW, 1=MID, 2=HIGH)

        Returns:
            token_costs: [batch, seq_len] cost for each token
        """
        # Define relative costs for different precisions
        cost_map = {
            0: self.config.low_precision_bits / 8,      # LOW precision cost
            1: self.config.medium_precision_bits / 8,   # MID precision cost  
            2: self.config.high_precision_bits / 8      # HIGH precision cost
        }

        token_costs = torch.zeros_like(precision_labels, dtype=torch.float)

        for precision, cost in cost_map.items():
            mask = (precision_labels == precision)
            token_costs[mask] = cost

        return token_costs

    def select_tokens_with_budget(
        self,
        importance_scores: torch.Tensor,
        precision_labels: torch.Tensor,
        budget_ratio: float,
        layer_idx: int
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Select tokens to propagate based on importance scores and budget constraints
        Solves: S^(l) = argmax Σ s_i  s.t. Σ c(prec_i) ≤ B^(l)

        Args:
            importance_scores: [batch, seq_len] token importance scores
            precision_labels: [batch, seq_len] precision assignments
            budget_ratio: fraction of original tokens to keep
            layer_idx: current layer index

        Returns:
            selection_mask: [batch, seq_len] boolean mask for selected tokens
            selection_info: dictionary with selection statistics
        """
        batch_size, seq_len = importance_scores.shape
        device = importance_scores.device

        # Compute token costs
        token_costs = self.compute_token_costs(precision_labels)

        # Calculate budget per batch
        total_budget = seq_len * budget_ratio

        selection_mask = torch.zeros_like(importance_scores, dtype=torch.bool)
        selection_info = {
            'selected_counts': [],
            'budget_utilization': [],
            'avg_importance': [],
            'cost_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }

        # Select tokens for each batch independently
        for b in range(batch_size):
            batch_scores = importance_scores[b]  # [seq_len]
            batch_costs = token_costs[b]         # [seq_len]
            batch_precisions = precision_labels[b]  # [seq_len]

            # Sort by importance score (descending)
            sorted_indices = torch.argsort(batch_scores, descending=True)

            # Greedy selection with budget constraint
            selected_indices = []
            current_cost = 0.0

            for idx in sorted_indices:
                idx_item = idx.item()
                token_cost = batch_costs[idx_item].item()

                if current_cost + token_cost <= total_budget:
                    selected_indices.append(idx_item)
                    current_cost += token_cost
                else:
                    # Try to fit remaining budget with lower cost tokens
                    remaining_budget = total_budget - current_cost
                    if token_cost <= remaining_budget:
                        selected_indices.append(idx_item)
                        current_cost += token_cost

            # Update selection mask
            if selected_indices:
                selection_mask[b, selected_indices] = True

                # Collect statistics
                selected_scores = batch_scores[selected_indices]
                selected_precisions = batch_precisions[selected_indices]

                selection_info['selected_counts'].append(len(selected_indices))
                selection_info['budget_utilization'].append(current_cost / total_budget)
                selection_info['avg_importance'].append(selected_scores.mean().item())

                # Count precision distribution
                for prec in [0, 1, 2]:
                    count = (selected_precisions == prec).sum().item()
                    if prec == 0:
                        selection_info['cost_distribution']['low'] += count
                    elif prec == 1:
                        selection_info['cost_distribution']['medium'] += count
                    else:
                        selection_info['cost_distribution']['high'] += count

        # Aggregate statistics
        if selection_info['selected_counts']:
            selection_info['avg_selected'] = sum(selection_info['selected_counts']) / len(selection_info['selected_counts'])
            selection_info['avg_budget_util'] = sum(selection_info['budget_utilization']) / len(selection_info['budget_utilization'])
            selection_info['overall_avg_importance'] = sum(selection_info['avg_importance']) / len(selection_info['avg_importance'])

        return selection_mask, selection_info

    def apply_token_selection(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        importance_scores: torch.Tensor, 
        precision_labels: torch.Tensor,
        layer_idx: int,
        input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Apply selective token propagation to key/value states

        Args:
            key_states: [batch, seq_len, head_dim] key tensor
            value_states: [batch, seq_len, head_dim] value tensor
            importance_scores: [batch, seq_len] importance scores
            precision_labels: [batch, seq_len] precision labels
            layer_idx: current layer index
            input_ids: [batch, seq_len] input token ids (optional)

        Returns:
            selected_keys: compressed key states
            selected_values: compressed value states  
            selected_scores: importance scores for selected tokens
            selected_labels: precision labels for selected tokens
            propagation_info: selection statistics
        """
        # Get layer-specific propagation ratio
        propagation_ratio = self.get_layer_propagation_ratio(layer_idx)

        # Select tokens with budget constraint
        selection_mask, selection_info = self.select_tokens_with_budget(
            importance_scores, precision_labels, propagation_ratio, layer_idx
        )

        # Apply selection to tensors
        batch_size, seq_len, head_dim = key_states.shape

        # Count selected tokens per batch for indexing
        selected_lengths = selection_mask.sum(dim=1)  # [batch]
        max_selected = selected_lengths.max().item()

        if max_selected == 0:
            # Emergency fallback: select top tokens if budget is too restrictive
            top_k = max(1, int(seq_len * 0.1))  # At least 10% of tokens
            _, top_indices = torch.topk(importance_scores, top_k, dim=1)
            selection_mask = torch.zeros_like(importance_scores, dtype=torch.bool)
            selection_mask.scatter_(1, top_indices, True)
            max_selected = top_k

        # Create padded output tensors
        selected_keys = torch.zeros(batch_size, max_selected, head_dim, 
                                   device=key_states.device, dtype=key_states.dtype)
        selected_values = torch.zeros(batch_size, max_selected, head_dim,
                                     device=value_states.device, dtype=value_states.dtype)
        selected_scores = torch.zeros(batch_size, max_selected, 
                                     device=importance_scores.device, dtype=importance_scores.dtype)
        selected_labels = torch.zeros(batch_size, max_selected,
                                     device=precision_labels.device, dtype=precision_labels.dtype)

        # Fill selected tensors
        for b in range(batch_size):
            mask = selection_mask[b]  # [seq_len]
            selected_count = mask.sum().item()

            if selected_count > 0:
                selected_keys[b, :selected_count] = key_states[b][mask]
                selected_values[b, :selected_count] = value_states[b][mask] 
                selected_scores[b, :selected_count] = importance_scores[b][mask]
                selected_labels[b, :selected_count] = precision_labels[b][mask]

        # Prepare propagation info
        propagation_info = {
            'layer_idx': layer_idx,
            'propagation_ratio': propagation_ratio,
            'original_length': seq_len,
            'max_selected_length': max_selected,
            'selection_mask': selection_mask,
            'selection_stats': selection_info
        }

        return selected_keys, selected_values, selected_scores, selected_labels, propagation_info

    def estimate_compression_ratio(self, layer_idx: int, original_length: int) -> Dict[str, float]:
        """Estimate overall compression ratio up to current layer"""

        cumulative_ratio = 1.0
        for l in range(layer_idx + 1):
            layer_ratio = self.get_layer_propagation_ratio(l)
            cumulative_ratio *= layer_ratio

        return {
            'layer_ratio': self.get_layer_propagation_ratio(layer_idx),
            'cumulative_ratio': cumulative_ratio,
            'estimated_length': int(original_length * cumulative_ratio),
            'compression_factor': 1.0 / cumulative_ratio
        }