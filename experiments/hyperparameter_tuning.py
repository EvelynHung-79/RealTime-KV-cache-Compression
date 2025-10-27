#!/usr/bin/env python3
"""Hyperparameter tuning for compression parameters."""

import argparse
import json
import optuna
import torch
from pathlib import Path
from src.models.modified_llama import CompressedLlamaForCausalLM
from src.evaluation.longbench_eval import LongBenchEvaluator
from configs.base_config import CompressionConfig
from transformers import AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="tuning_results/")
    return parser.parse_args()


def objective(trial, model_name, device="cuda"):
    """Optuna objective function."""

    # Define search space
    config = CompressionConfig(
        chunk_size=trial.suggest_categorical("chunk_size", [256, 512, 1024]),
        ema_decay=trial.suggest_float("ema_decay", 0.85, 0.99),
        attention_sink_size=trial.suggest_int("attention_sink_size", 2, 8),
        key_bits_normal=4,  # Fixed
        value_bits_normal=4,  # Fixed
        key_bits_sink_outlier=8,
        value_bits_sink_outlier=8,
        outlier_threshold_relative=trial.suggest_float("outlier_threshold", 2.0, 4.0),
        outlier_detection_enabled=True,
    )

    # Load model
    model = CompressedLlamaForCausalLM.from_pretrained(
        model_name,
        compression_config=config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Evaluate
    evaluator = LongBenchEvaluator(model, tokenizer, device)
    results = evaluator.evaluate(task="narrativeqa", max_samples=50)

    # Clean up
    del model
    torch.cuda.empty_cache()

    # Objective: maximize F1 while maintaining compression
    f1_score = results.get("f1", 0)
    compression_ratio = results.get("mean_compression_ratio_key", 1.0)

    # Combined objective: F1 * log(compression_ratio)
    import math
    objective_value = f1_score * math.log(compression_ratio + 1)

    return objective_value


def main():
    args = parse_arguments()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Hyperparameter Tuning")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Number of trials: {args.n_trials}")
    print("="*80)

    # Create study
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, args.model_name),
        n_trials=args.n_trials,
    )

    # Print results
    print("\n" + "="*80)
    print("Best Trial")
    print("="*80)
    print(f"Value: {study.best_value:.4f}")
    print("Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    results_file = output_dir / "tuning_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
            "all_trials": [
                {"number": trial.number, "value": trial.value, "params": trial.params}
                for trial in study.trials
            ],
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
