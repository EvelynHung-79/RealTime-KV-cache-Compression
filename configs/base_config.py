"""
Base configuration for Streaming KVQuant with Sink and Outlier Awareness.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompressionConfig:
    """Configuration for real-time KV cache compression.

    Attributes:
        # Streaming processing parameters
        chunk_size: Size of each processing chunk in tokens (default: 512)
        ema_decay: Exponential moving average decay factor for statistics (default: 0.9)

        # Attention Sink parameters
        attention_sink_size: Number of initial tokens to treat as attention sinks (default: 4)

        # Quantization bit widths
        key_bits_normal: Bit width for normal key quantization (default: 4)
        value_bits_normal: Bit width for normal value quantization (default: 4)
        key_bits_sink_outlier: Bit width for sink/outlier key quantization (default: 8)
        value_bits_sink_outlier: Bit width for sink/outlier value quantization (default: 8)

        # Outlier detection parameters
        outlier_threshold_relative: Relative threshold multiplier for outlier detection (default: 3.0)
        outlier_threshold_abs: Absolute threshold for outlier detection (optional)
        outlier_detection_enabled: Whether to enable outlier detection (default: True)

        # Value quantization strategy
        value_quant_groups: Number of groups for value quantization (0 for per-channel) (default: 0)

        # Key quantization strategy
        key_quant_per_channel: Whether to use per-channel quantization for keys (default: True)

        # RoPE configuration
        quantize_pre_rope: Whether to quantize keys before RoPE (default: True)

        # Memory management
        enable_mixed_precision_cache: Whether to use mixed precision cache storage (default: True)

        # Experimental features
        adaptive_chunk_size: Whether to adaptively adjust chunk size (default: False)
        periodic_stat_reset: Whether to periodically reset statistics (default: False)
        stat_reset_interval: Interval for resetting statistics in chunks (default: 1000)

        # Debugging and logging
        log_statistics: Whether to log compression statistics (default: False)
        verbose: Verbosity level (default: 0)
    """

    # Streaming processing parameters
    chunk_size: int = 512
    ema_decay: float = 0.9

    # Attention Sink parameters
    attention_sink_size: int = 4

    # Quantization bit widths
    key_bits_normal: int = 4
    value_bits_normal: int = 4
    key_bits_sink_outlier: int = 8
    value_bits_sink_outlier: int = 8

    # Outlier detection parameters
    outlier_threshold_relative: float = 3.0
    outlier_threshold_abs: Optional[float] = None
    outlier_detection_enabled: bool = True

    # Value quantization strategy
    value_quant_groups: int = 0  # 0 means per-channel

    # Key quantization strategy
    key_quant_per_channel: bool = True

    # RoPE configuration
    quantize_pre_rope: bool = True

    # Memory management
    enable_mixed_precision_cache: bool = True

    # Experimental features
    adaptive_chunk_size: bool = False
    periodic_stat_reset: bool = False
    stat_reset_interval: int = 1000

    # Debugging and logging
    log_statistics: bool = False
    verbose: int = 0

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate chunk size
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        # Validate EMA decay
        if not 0 < self.ema_decay < 1:
            raise ValueError(f"ema_decay must be in (0, 1), got {self.ema_decay}")

        # Validate attention sink size
        if self.attention_sink_size < 0:
            raise ValueError(f"attention_sink_size must be non-negative, got {self.attention_sink_size}")

        # Validate bit widths
        valid_bits = [1, 2, 3, 4, 8, 16]
        if self.key_bits_normal not in valid_bits:
            raise ValueError(f"key_bits_normal must be in {valid_bits}, got {self.key_bits_normal}")
        if self.value_bits_normal not in valid_bits:
            raise ValueError(f"value_bits_normal must be in {valid_bits}, got {self.value_bits_normal}")
        if self.key_bits_sink_outlier not in valid_bits:
            raise ValueError(f"key_bits_sink_outlier must be in {valid_bits}, got {self.key_bits_sink_outlier}")
        if self.value_bits_sink_outlier not in valid_bits:
            raise ValueError(f"value_bits_sink_outlier must be in {valid_bits}, got {self.value_bits_sink_outlier}")

        # Validate outlier threshold
        if self.outlier_threshold_relative <= 1.0:
            raise ValueError(f"outlier_threshold_relative must be > 1.0, got {self.outlier_threshold_relative}")

        if self.outlier_threshold_abs is not None and self.outlier_threshold_abs <= 0:
            raise ValueError(f"outlier_threshold_abs must be positive, got {self.outlier_threshold_abs}")

        # Validate value quantization groups
        if self.value_quant_groups < 0:
            raise ValueError(f"value_quant_groups must be non-negative, got {self.value_quant_groups}")

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "chunk_size": self.chunk_size,
            "ema_decay": self.ema_decay,
            "attention_sink_size": self.attention_sink_size,
            "key_bits_normal": self.key_bits_normal,
            "value_bits_normal": self.value_bits_normal,
            "key_bits_sink_outlier": self.key_bits_sink_outlier,
            "value_bits_sink_outlier": self.value_bits_sink_outlier,
            "outlier_threshold_relative": self.outlier_threshold_relative,
            "outlier_threshold_abs": self.outlier_threshold_abs,
            "outlier_detection_enabled": self.outlier_detection_enabled,
            "value_quant_groups": self.value_quant_groups,
            "key_quant_per_channel": self.key_quant_per_channel,
            "quantize_pre_rope": self.quantize_pre_rope,
            "enable_mixed_precision_cache": self.enable_mixed_precision_cache,
        }

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)
