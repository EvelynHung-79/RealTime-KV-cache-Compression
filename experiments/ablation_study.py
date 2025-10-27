#!/usr/bin/env python3
"""Ablation study for compression components."""

import argparse
import json
import pandas as pd
from pathlib import Path
from src.evaluation.benchmark_runner import BenchmarkRunner
from configs.base_config import CompressionConfig


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_dir", type=str, default="ablation_results/")
    parser.add_argument("--tasks", nargs="+", default=["narrativeqa", "qasper"])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def create_ablation_configs():
    """Create configurations for ablation study."""
    configs = []

    # 1. Baseline: No compression
    configs.append(("Baseline (FP16)", CompressionConfig(
        key_bits_normal=16,
        value_bits_normal=16,
        attention_sink_size=0,
        outlier_detection_enabled=False,
    )))

    # 2. Uniform INT4
    configs.append(("Uniform INT4", CompressionConfig(
        key_bits_normal=4,
        value_bits_normal=4,
        attention_sink_size=0,
        outlier_detection_enabled=False,
    )))

    # 3. INT4 + Sink only
    configs.append(("INT4 + Sink", CompressionConfig(
        key_bits_normal=4,
        value_bits_normal=4,
        key_bits_sink_outlier=8,
        value_bits_sink_outlier=8,
        attention_sink_size=4,
        outlier_detection_enabled=False,
    )))

    # 4. INT4 + Outlier only
    configs.append(("INT4 + Outlier", CompressionConfig(
        key_bits_normal=4,
        value_bits_normal=4,
        key_bits_sink_outlier=8,
        value_bits_sink_outlier=8,
        attention_sink_size=0,
        outlier_detection_enabled=True,
        outlier_threshold_relative=3.0,
    )))

    # 5. Full method: INT4 + Sink + Outlier
    configs.append(("Full (INT4 + Sink + Outlier)", CompressionConfig(
        key_bits_normal=4,
        value_bits_normal=4,
        key_bits_sink_outlier=8,
        value_bits_sink_outlier=8,
        attention_sink_size=4,
        outlier_detection_enabled=True,
        outlier_threshold_relative=3.0,
    )))

    # 6. Different EMA decay
    configs.append(("Full + EMA=0.95", CompressionConfig(
        key_bits_normal=4,
        value_bits_normal=4,
        key_bits_sink_outlier=8,
        value_bits_sink_outlier=8,
        attention_sink_size=4,
        outlier_detection_enabled=True,
        outlier_threshold_relative=3.0,
        ema_decay=0.95,
    )))

    return configs


def main():
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create ablation configurations
    ablation_configs = create_ablation_configs()

    print("="*80)
    print("Ablation Study")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Tasks: {args.tasks}")
    print(f"Number of configurations: {len(ablation_configs)}")
    print("="*80)

    # Run benchmark
    runner = BenchmarkRunner(args.model_name, args.device)
    configs = [config for _, config in ablation_configs]
    config_names = [name for name, _ in ablation_configs]

    results = runner.run_benchmark(configs, args.tasks)

    # Create summary table
    summary_data = []
    for i, (config_name, config) in enumerate(ablation_configs):
        config_key = f"config_{i}"
        config_results = results[config_key]["task_results"]

        row = {"Configuration": config_name}
        for task in args.tasks:
            task_metrics = config_results[task]
            row[f"{task}_rouge1"] = task_metrics.get("rouge1", 0)
            row[f"{task}_f1"] = task_metrics.get("f1", 0)
            row[f"{task}_compression"] = task_metrics.get("mean_compression_ratio_key", 1.0)

        summary_data.append(row)

    # Save results
    df = pd.DataFrame(summary_data)
    csv_file = output_dir / "ablation_summary.csv"
    df.to_csv(csv_file, index=False)

    json_file = output_dir / "ablation_full_results.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("Ablation Summary")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
