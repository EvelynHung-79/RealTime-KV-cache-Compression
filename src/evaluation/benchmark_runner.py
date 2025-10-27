"""Benchmark runner for comprehensive evaluation."""

import torch
from typing import Dict, List
from .longbench_eval import LongBenchEvaluator
from .metrics import CompressionMetrics, calculate_perplexity
from configs.base_config import CompressionConfig

class BenchmarkRunner:
    """Run comprehensive benchmarks on compression configurations."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device

    def run_benchmark(
        self,
        compression_configs: List[CompressionConfig],
        tasks: List[str] = ["narrativeqa", "qasper"],
    ) -> Dict:
        """Run benchmark across multiple configurations and tasks.

        Args:
            compression_configs: List of compression configurations to test
            tasks: List of evaluation tasks

        Returns:
            Dictionary of results for each configuration
        """
        results = {}

        for i, config in enumerate(compression_configs):
            print(f"\nEvaluating configuration {i+1}/{len(compression_configs)}")
            print(f"Config: {config.to_dict()}")

            # Load model with this configuration
            from src.models.modified_llama import CompressedLlamaForCausalLM
            from transformers import AutoTokenizer

            model = CompressedLlamaForCausalLM.from_pretrained(
                self.model_name,
                compression_config=config,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Evaluate on each task
            config_results = {}
            evaluator = LongBenchEvaluator(model, tokenizer, self.device)

            for task in tasks:
                print(f"  Task: {task}")
                task_results = evaluator.evaluate(task=task, max_samples=50)
                config_results[task] = task_results

            results[f"config_{i}"] = {
                "config": config.to_dict(),
                "task_results": config_results,
            }

            # Clean up
            del model
            torch.cuda.empty_cache()

        return results

    @staticmethod
    def create_test_configs() -> List[CompressionConfig]:
        """Create a set of test configurations for ablation."""
        configs = []

        # Baseline: No compression (INT16)
        configs.append(CompressionConfig(
            key_bits_normal=16,
            value_bits_normal=16,
            attention_sink_size=0,
            outlier_detection_enabled=False,
        ))

        # INT8 uniform
        configs.append(CompressionConfig(
            key_bits_normal=8,
            value_bits_normal=8,
            attention_sink_size=0,
            outlier_detection_enabled=False,
        ))

        # INT4 uniform
        configs.append(CompressionConfig(
            key_bits_normal=4,
            value_bits_normal=4,
            attention_sink_size=0,
            outlier_detection_enabled=False,
        ))

        # INT4 with Sink protection
        configs.append(CompressionConfig(
            key_bits_normal=4,
            value_bits_normal=4,
            key_bits_sink_outlier=8,
            value_bits_sink_outlier=8,
            attention_sink_size=4,
            outlier_detection_enabled=False,
        ))

        # INT4 with Sink + Outlier detection (Full method)
        configs.append(CompressionConfig(
            key_bits_normal=4,
            value_bits_normal=4,
            key_bits_sink_outlier=8,
            value_bits_sink_outlier=8,
            attention_sink_size=4,
            outlier_detection_enabled=True,
            outlier_threshold_relative=3.0,
        ))

        return configs
