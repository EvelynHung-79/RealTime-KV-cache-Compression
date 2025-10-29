#!/usr/bin/env python3
"""Run compression experiments on LongBench."""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer

# Add root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.models.modified_llama import CompressedLlamaForCausalLM
from src.evaluation.longbench_eval import LongBenchEvaluator
from src.evaluation.metrics import CompressionMetrics, calculate_perplexity
from configs.base_config import CompressionConfig

ALL_LONGBENCH_TASKS = [
    # Multi-doc QA
    "hotpotqa", "2wikimultihopqa", "musique",
    # Single-doc QA
    "narrativeqa", "qasper",
    # Summarization
    "gov_report", "qmsum", "multinews",
    # Few shot
    "triviaqa", "samsum", "trec",
    # Synthetic
    "passage_retrieval_en", "passage_count",
    # Code
    "repobench-p", "lcc"
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run KV cache compression experiments")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Pretrained model name or path")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="longbench",
                       help="Dataset to evaluate on")
    parser.add_argument("--tasks", nargs='+', default=ALL_LONGBENCH_TASKS,
                       help="Specific LongBench task(s) to evaluate (default: all)")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum number of samples to evaluate")

    # Compression arguments
    parser.add_argument("--chunk_size", type=int, default=512,
                       help="Chunk size for streaming processing")
    parser.add_argument("--ema_decay", type=float, default=0.9,
                       help="EMA decay factor")
    parser.add_argument("--attention_sink_size", type=int, default=4,
                       help="Number of attention sink tokens")
    parser.add_argument("--key_bits_normal", type=int, default=4,
                       help="Bit width for normal key quantization")
    parser.add_argument("--value_bits_normal", type=int, default=4,
                       help="Bit width for normal value quantization")
    parser.add_argument("--key_bits_sink_outlier", type=int, default=8,
                       help="Bit width for sink/outlier key quantization")
    parser.add_argument("--value_bits_sink_outlier", type=int, default=8,
                       help="Bit width for sink/outlier value quantization")
    parser.add_argument("--outlier_threshold_relative", type=float, default=3.0,
                       help="Relative threshold for outlier detection")
    parser.add_argument("--disable-outlier-detection", action="store_false",
                   dest="outlier_detection_enabled", # Store result in args.outlier_detection_enabled
                   help="Disable outlier detection (default: enabled)")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/",
                       help="Output directory for results")
    parser.add_argument("--exp_name", type=str, default="compression_exp",
                       help="Experiment name")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create compression config
    compression_config = CompressionConfig(
        chunk_size=args.chunk_size,
        ema_decay=args.ema_decay,
        attention_sink_size=args.attention_sink_size,
        key_bits_normal=args.key_bits_normal,
        value_bits_normal=args.value_bits_normal,
        key_bits_sink_outlier=args.key_bits_sink_outlier,
        value_bits_sink_outlier=args.value_bits_sink_outlier,
        outlier_threshold_relative=args.outlier_threshold_relative,
        outlier_detection_enabled=args.outlier_detection_enabled,
    )

    print("="*80)
    print("Starting Compression Experiment")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Tasks: {args.tasks}")
    # print(f"\nCompression Config:")
    # for key, value in compression_config.to_dict().items():
    #     print(f"  {key}: {value}")
    print("="*80)

    # Load model and tokenizer
    print("\nLoading model...")
    model = CompressedLlamaForCausalLM.from_pretrained(
        args.model_name,
        compression_config=compression_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Run evaluation
    print("\nRunning evaluation...")
    evaluator = LongBenchEvaluator(model, tokenizer, args.device)

    all_results = {}
    for task in args.tasks:
        print(f"\n--- Evaluating Task: {task} ---")
        try:
            task_results = evaluator.evaluate(
                dataset_name=args.dataset,
                task=task,
                max_samples=args.max_samples,
            )
            all_results[task] = task_results
            print(f"--- Task {task} Results ---")
            for key, value in task_results.items():
                 if isinstance(value, float):
                     print(f"  {key}: {value:.4f}")
                 else:
                     print(f"  {key}: {value}")
        except Exception as e:
            print(f"!!! Error evaluating task {task}: {e}")
            all_results[task] = {"error": str(e)}

    # Print results
    print("\n" + "="*80)
    print("Overall Results Summary")
    print("="*80)

    avg_f1 = sum(res.get('f1', 0) for res in all_results.values() if 'f1' in res) / len(all_results)
    avg_rougeL = sum(res.get('rougeL', 0) for res in all_results.values() if 'rougeL' in res) / len(all_results)
    print(f"Average F1 across tasks: {avg_f1:.4f}")
    print(f"Average Rouge-L across tasks: {avg_rougeL:.4f}")

    # Save results
    output_file = output_dir / f"{args.exp_name}_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": compression_config.to_dict(),
            "results_per_task": all_results,
            "args": vars(args),
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
