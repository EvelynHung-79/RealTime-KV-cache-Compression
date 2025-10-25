from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class CompressionConfig:
    """Real-time Prefill KV Cache Compression Configuration"""

    # Model configuration
    model_name: str = "meta-llama/Llama-2-7b-hf"
    max_position_embeddings: int = 4096
    num_hidden_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32

    # Compression hyperparameters
    alpha: float = 0.4  # Prompt attention weight
    beta: float = 0.3   # Position bias weight  
    gamma: float = 0.3  # Context relevance weight

    # Importance thresholds
    theta_h: float = 0.7  # High precision threshold
    theta_m: float = 0.3  # Medium precision threshold

    # Layer-specific weights (decreasing for later layers)
    layer_weights: Optional[List[float]] = None

    # Propagation ratios for different layer groups
    early_layer_ratio: float = 0.8   # First 30% layers
    middle_layer_ratio: float = 0.6  # Middle 40% layers  
    later_layer_ratio: float = 0.4   # Last 30% layers

    # Quantization bits
    high_precision_bits: int = 16
    medium_precision_bits: int = 8
    low_precision_bits: int = 4

    # Memory and performance
    memory_budget_ratio: float = 0.5  # Target memory reduction
    quality_loss_tolerance: float = 0.05  # Max acceptable quality loss

    # Evaluation settings
    context_lengths: List[int] = None
    batch_sizes: List[int] = None

    def __post_init__(self):
        if self.layer_weights is None:
            # Default layer weights: decreasing from 1.0 to 0.5
            self.layer_weights = [
                1.0 - 0.5 * (i / (self.num_hidden_layers - 1)) 
                for i in range(self.num_hidden_layers)
            ]

        if self.context_lengths is None:
            self.context_lengths = [4096, 8192, 16384, 32768]

        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8]