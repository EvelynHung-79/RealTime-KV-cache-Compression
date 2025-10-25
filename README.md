# Real-time Prefill KV Cache Compression (Streaming KVQuant Adaptation)

A PyTorch implementation adapting KVQuant principles for real-time KV cache quantization during the prefill stage of LLaMA2 models, focusing on streaming processing and Sink/Outlier awareness.

## Overview

This project implements a real-time quantization approach for the Key-Value (KV) cache during the prefill stage of transformer inference. Unlike traditional offline methods, this approach processes the input sequence in chunks, dynamically calculating quantization parameters based on streaming statistics without prior calibration. It aims to reduce the memory footprint of the KV cache by storing quantized values while retaining all tokens to maintain model quality.

### Key Features

-   **Real-time Streaming Quantization**: Quantization occurs chunk-by-chunk during the prefill pass.
-   **KVQuant Inspired**: Leverages KVQuant's core ideas like Per-channel Key quantization (Pre-RoPE) and differentiated Value quantization.
-   **Online Statistics**: Uses Exponential Moving Average (EMA) of Absmax to dynamically determine quantization scales, eliminating the need for offline calibration.
-   **Sink & Outlier Awareness**: Applies higher precision quantization to initial "Attention Sink" tokens and dynamically detected outlier channels/groups to preserve critical information.
-   **No Token Dropping**: All tokens are processed and retained throughout the layers; compression is achieved solely through quantization.
-   **Chunked Processing**: Handles long sequences by processing them in fixed-size chunks to manage computational overhead.

### Method Overview

The method integrates the following components within each transformer layer during prefill:

1.  **Streaming Statistics Manager**:
    * Maintains running EMA Absmax statistics for Key channels (per-channel) and Value channels/groups across processed chunks.
    * Detects outlier channels/groups based on configurable absolute or relative thresholds using these statistics.
2.  **Real-time Quantizer**:
    * Performs symmetric quantization (e.g., INT8, INT4) based on scales derived from the Streaming Statistics Manager.
    * Applies specific quantization logic:
        * **Keys**: Pre-RoPE, Per-channel quantization. Uses higher precision for Sink tokens and detected Outlier channels.
        * **Values**: Post-RoPE, Group-wise or Simplified Per-channel quantization. Uses higher precision for Sink tokens and detected Outlier groups/channels.
3.  **Modified Attention Module**:
    * Processes input `hidden_states` in chunks (`chunk_size`).
    * Coordinates the calculation of K/V, updates streaming statistics, performs quantization using the Real-time Quantizer, applies RoPE (to high-precision or temporarily dequantized values), and calculates attention using high-precision values.
    * Appends the **quantized** Key and Value chunks to the KV Cache for storage.

## Installation

### Prerequisites

-   Python 3.9+
-   PyTorch 2.0+
-   CUDA 11.8+ (for GPU acceleration)
-   Dependencies listed in `requirements.txt`
-   At least 16GB RAM, 8GB+ GPU memory recommended

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd RealTime-KV-cache-Compression

# Setup environment (creates venv, installs dependencies)
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh # Source is usually not needed here

# Activate environment
source venv/bin/activate

## Usage

### Download Model

Download the desired LLaMA model (e.g., LLaMA2-7B) using the provided script:

```bash
python scripts/download_model.py
```

*(Ensure you have the necessary permissions and Hugging Face Hub login configured if required)*

### Basic Tests

Run unit tests to verify core components:

```bash
# Set PYTHONPATH for imports
export PYTHONPATH=$(pwd)

# Test new quantization components
pytest tests/test_quantization.py -v

# Test overall compression logic (needs update for new structure)
pytest tests/test_compression.py -v

# Run end-to-end functionality test
python tests/test_functionality.py
```

### Running Experiments

Use the main experiment script to run evaluations with the streaming quantization:

```bash
python experiments/run_compression_experiment.py \
    --model_name "models/llama2-7b" \
    --device cuda \
    --max_length 4096 \
    --chunk_size 256 \
    --key_bits_normal 4 \
    --value_bits_normal 4 \
    --attention_sink_size 8 \
    --ema_decay 0.99 \
    --tasks narrativeqa qasper \
    --max_samples 10 \
    --output_dir "./experiments/results" \
    --experiment_name "streaming_quant_test" \
    --baseline # Optionally run baseline comparison
```

*(Adjust parameters in `configs/base_config.py` or via command line)*

#### Ablation Studies & Hyperparameter Tuning

The scripts `experiments/ablation_study.py` and `experiments/hyperparameter_tuning.py` need to be **updated** to reflect the new set of configurable parameters (e.g., `chunk_size`, `ema_decay`, `outlier_thresholds`, `attention_sink_size`, quantization bits).

Example conceptual tuning command (script needs modification):

```bash
# Hypothetical command after updating hyperparameter_tuning.py
python experiments/hyperparameter_tuning.py \
    --method bayesian_optimization \
    --n_trials 30 \
    --search_space chunk_size ema_decay key_bits_normal value_bits_normal \
    --output_dir "./tuning_results"
```

## Configuration

Core hyperparameters are defined in `configs/base_config.py` (`CompressionConfig`). Key parameters for the Streaming KVQuant method include:

| Parameter                 | Description                                                                 | Default |
| :------------------------ | :-------------------------------------------------------------------------- | :------ |
| `chunk_size`              | Size of token chunks processed during prefill                               | 256     |
| `ema_decay`               | Decay factor for EMA Absmax statistics                                      | 0.99    |
| `outlier_threshold_abs`   | Absolute value threshold for marking a channel/group as outlier (optional)  | 6.0     |
| `outlier_threshold_relative`| Relative threshold (vs EMA Absmax) for marking as outlier (optional)        | 5.0     |
| `attention_sink_size`     | Number of initial tokens treated as Attention Sink (higher precision)       | 8       |
| `key_bits_normal`         | Bits for regular Key channel quantization                                   | 4       |
| `key_bits_sink_outlier`   | Bits for Key channel quantization (Sink tokens or Outliers)                 | 8       |
| `value_bits_normal`       | Bits for regular Value group/channel quantization                           | 4       |
| `value_bits_sink_outlier` | Bits for Value group/channel quantization (Sink tokens or Outliers)         | 8       |
| `value_quant_groups`      | Value quantization grouping (-1: per-channel, 1: per-tensor, \>1: group-wise)| -1      |

## Evaluation

### Benchmarks

The implementation supports evaluation on various benchmarks, primarily focused on long-context tasks:

  - **LongBench**: Includes tasks like NarrativeQA, Qasper, MultiFieldQA, HotpotQA, GovReport, MultiNews etc.
  - **Wikitext2**: Standard language modeling perplexity benchmark.
  - **Needle-in-a-Haystack**: Tests information retrieval capability within long contexts.

### Metrics

**Quality Metrics**:

  - Task-specific scores (F1, ROUGE-L, Exact Match Accuracy etc.)
  - Perplexity (for language modeling tasks)

**Performance & Compression Metrics**:

  - Time-To-First-Token (TTFT) - approximated by prefill processing time.
  - Generation Tokens per second.
  - Peak & Average Memory usage (GPU/CPU).
  - **Estimated Memory Savings**: Calculated based on the average number of bits used for quantization compared to the baseline (e.g., FP16).
  - Statistics on Outlier detection rates and Sink token handling.

## Repository Structure

```
real-time-prefill-kv-cache-compression/
├── src/
│   ├── compression/           # Core compression modules
│   │   ├── streaming_quantization.py # NEW: Streaming stats & quantization logic
│   │   └── unified_compressor.py    # Manages stats, reset logic
│   ├── models/               # Modified model architectures
│   │   ├── modified_llama.py       # LLaMA with modified Attention for streaming
│   │   └── compression_layers.py   # Custom layers (if any needed beyond Attention)
│   ├── evaluation/           # Evaluation utilities
│   │   ├── longbench_eval.py       # LongBench evaluator
│   │   ├── metrics.py              # Metrics calculation (Needs update)
│   │   └── benchmark_runner.py     # Benchmark suite (Needs update)
│   └── utils/               # Utility functions (memory, eval, data)
├── configs/                 # Configuration files (base_config updated)
├── experiments/            # Experiment scripts (Needs update for new params)
│   ├── run_compression_experiment.py
│   ├── ablation_study.py
│   └── hyperparameter_tuning.py
├── scripts/               # Shell scripts (setup, download)
└── tests/                 # Unit tests (Needs update/rewrite)
    ├── test_quantization.py # NEW/REWRITE: Test streaming_quantization.py
    └── test_compression.py  # REWRITE: Test integration in attention
    └── test_functionality.py# Update config/checks
```

## Results

### Expected Performance

This streaming quantization method aims for:

  - **Memory Savings**: Significant reduction in KV cache memory footprint (e.g., aiming for 2x-4x reduction depending on target bits) compared to FP16 baseline.
  - **Quality Retention**: High retention of model performance on downstream tasks due to Sink/Outlier handling and retaining all tokens.
  - **Latency**: Prefill latency potentially higher than baseline due to chunking and quantization overhead, but lower than methods requiring complex per-token scoring. Decode latency should be minimally impacted.
  - **Long Context**: Enables processing of longer sequences than possible with uncompressed FP16 KV cache within the same memory constraints.

*(Actual results depend heavily on hyperparameter tuning and specific task evaluation.)*

## Troubleshooting

### Common Issues

1.  **CUDA Out of Memory**:
      * Reduce `chunk_size`.
      * Use lower precision bits (`key_bits_normal`, `value_bits_normal`).
      * Reduce batch size (if applicable during prefill testing).
2.  **Slow Performance**:
      * Ensure CUDA kernels are utilized (PyTorch operations).
      * Profile the quantization and statistics update steps. `ema_decay` close to 1 is cheaper but adapts slower.
      * Increase `chunk_size` (trades memory for potentially fewer overhead calls).
3.  **Quality Degradation**:
      * Increase `attention_sink_size`.
      * Use higher precision bits (`*_bits_normal`, `*_bits_sink_outlier`).
      * Adjust `ema_decay` (faster adaptation might be needed for volatile activations).
      * Tune `outlier_threshold_*` parameters (making them stricter increases precision usage).

### Debug Mode

```bash
# Enable detailed logging (if implemented in code)
export LOG_LEVEL=DEBUG
# Potentially slow down execution for CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
python experiments/run_compression_experiment.py ... --verbose # (Add verbose flag if exists)
```

## Citation

If you use this work in your research, please consider citing the original KVQuant paper and potentially this implementation if adapted.

```bibtex
@article{kvquant-paper-placeholder,
  title={KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization},
  author={KVQuant Authors},
  # ... (Find actual citation details)
  year={2024}
}
```

## Acknowledgements

This work builds upon the principles introduced in:

  - **KVQuant**: For Per-channel Key quantization, Pre-RoPE, Sink/Outlier handling ideas.
  - Research on Attention Sinks in transformers (e.g., StreamingLLM).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m 'Add amazing feature'`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Open a Pull Request

## Support

For questions and support:

  - Open an issue on GitHub.
  - Check the troubleshooting section.
  - Review the configuration documentation (`configs/base_config.py`).

-----

**Note**: This implementation is for research purposes. Performance and stability may vary. Further optimization and testing are recommended for production use.