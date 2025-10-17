#!/usr/bin/env python3
"""
Real-time Prefill KV Cache Compression Experiment
Main script for running compression experiments on LLaMA2 with LongBench evaluation

這是一個用於運行LLaMA2模型KV緩存壓縮實驗的主要腳本。
主要功能：
1. 實現實時預填充KV緩存壓縮
2. 使用LongBench進行評估
3. 支持基準測試
"""

import os
import sys
import argparse
import json
import torch
import time
from datetime import datetime
from transformers import AutoTokenizer

from configs.base_config import CompressionConfig
from src.models.modified_llama import create_compressed_llama_model
from src.compression.unified_compressor import RealTimePrefillCompressor
from src.evaluation.longbench_eval import LongBenchEvaluator
from src.utils.memory_utils import MemoryMonitor
from src.utils.eval_utils import setup_logging

def parse_arguments():
    """
    解析命令行參數
    包含以下主要配置：
    1. 模型配置：模型名稱、設備選擇、序列長度等
    2. 壓縮超參數：注意力權重、閾值等
    3. 層傳播比率：控制不同層的壓縮程度
    4. 評估設置：任務選擇、樣本數等
    5. 實驗設置：輸出目錄、實驗名稱等
    """
    parser = argparse.ArgumentParser(description='Real-time Prefill KV Cache Compression Experiment')

    # 模型配置 (Model configuration)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf',
                       help='HuggingFace model name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--max_length', type=int, default=4096,
                       help='Maximum sequence length')

    # Compression hyperparameters
    parser.add_argument('--alpha', type=float, default=0.4,
                       help='Prompt attention weight')
    parser.add_argument('--beta', type=float, default=0.3,
                       help='Position bias weight')
    parser.add_argument('--gamma', type=float, default=0.3,
                       help='Context relevance weight')
    parser.add_argument('--theta_h', type=float, default=0.7,
                       help='High precision threshold')
    parser.add_argument('--theta_m', type=float, default=0.3,
                       help='Medium precision threshold')

    # Layer propagation ratios
    parser.add_argument('--early_ratio', type=float, default=0.8,
                       help='Early layer propagation ratio')
    parser.add_argument('--middle_ratio', type=float, default=0.6,
                       help='Middle layer propagation ratio')
    parser.add_argument('--later_ratio', type=float, default=0.4,
                       help='Later layer propagation ratio')

    # Evaluation settings
    parser.add_argument('--tasks', nargs='+', default=None,
                       help='LongBench tasks to evaluate (default: all)')
    parser.add_argument('--max_samples', type=int, default=50,
                       help='Max samples per task for evaluation')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                       help='Max tokens to generate per sample')

    # Experiment settings
    parser.add_argument('--output_dir', type=str, default='./experiments/results',
                       help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: timestamp)')
    parser.add_argument('--save_model', action='store_true',
                       help='Save compressed model after experiment')

    # Baseline comparison
    parser.add_argument('--baseline', action='store_true',
                       help='Run baseline (no compression) comparison')

    return parser.parse_args()


def create_experiment_config(args):
    """
    從命令行參數創建壓縮配置
    
    配置包含：
    1. 模型基本設置：模型名稱、最大位置嵌入等
    2. 壓縮權重：alpha（提示注意力）、beta（位置偏差）、gamma（上下文相關性）
    3. 精度閾值：theta_h（高精度）、theta_m（中等精度）
    4. 層傳播比率：控制不同深度層的壓縮程度
    """
    config = CompressionConfig(
        model_name=args.model_name,
        max_position_embeddings=args.max_length,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        theta_h=args.theta_h,
        theta_m=args.theta_m,
        early_layer_ratio=args.early_ratio,
        middle_layer_ratio=args.middle_ratio,
        later_layer_ratio=args.later_ratio,
        context_lengths=[args.max_length],
        batch_sizes=[1]
    )
    return config


def run_baseline_experiment(args, tokenizer, output_dir):
    """
    運行基準實驗（無壓縮）
    
    目的：
    1. 建立性能基準線，用於與壓縮模型比較
    2. 使用原始LLaMA模型進行評估
    3. 收集未壓縮模型的性能指標
    4. 保存基準測試結果供後續分析
    """
    print("\n" + "="*50)
    print("RUNNING BASELINE EXPERIMENT (No Compression)")
    print("="*50)

    from transformers import LlamaForCausalLM

    # Load baseline model
    baseline_model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    baseline_model.eval()

    # Create evaluator
    baseline_dir = os.path.join(output_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    # Create dummy config for evaluator
    dummy_config = create_experiment_config(args)
    evaluator = LongBenchEvaluator(baseline_model, tokenizer, dummy_config, baseline_dir)

    # Run evaluation
    start_time = time.time()
    results = evaluator.evaluate_all_tasks(
        tasks=args.tasks,
        max_samples_per_task=args.max_samples
    )
    evaluation_time = time.time() - start_time

    # Save results
    results['experiment_info'] = {
        'type': 'baseline',
        'model_name': args.model_name,
        'evaluation_time': evaluation_time,
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(baseline_dir, "baseline_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_compression_experiment(args, config, tokenizer, output_dir):
    """
    運行主要的壓縮實驗
    
    實現功能：
    1. 內存監控：追踪實驗過程中的內存使用情況
    2. 模型加載：創建並初始化壓縮版的LLaMA模型
    3. 性能評估：使用LongBench基準測試評估壓縮模型
    4. 結果收集：記錄評估指標、內存統計和實驗元數據
    5. 模型保存：可選擇性保存壓縮後的模型
    """
    print("\n" + "="*50)
    print("RUNNING COMPRESSION EXPERIMENT")
    print("="*50)

    # Initialize memory monitor
    memory_monitor = MemoryMonitor()
    memory_monitor.start_monitoring()

    # Create compressed model
    print(f"Loading compressed model: {args.model_name}")
    model = create_compressed_llama_model(args.model_name, config, args.device)

    print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create evaluator
    compression_dir = os.path.join(output_dir, "compressed")
    os.makedirs(compression_dir, exist_ok=True)

    evaluator = LongBenchEvaluator(model, tokenizer, config, compression_dir)

    # Run evaluation
    print(f"Starting evaluation on {len(args.tasks or evaluator.LONGBENCH_TASKS)} tasks...")
    start_time = time.time()

    results = evaluator.evaluate_all_tasks(
        tasks=args.tasks,
        max_samples_per_task=args.max_samples
    )

    evaluation_time = time.time() - start_time
    memory_stats = memory_monitor.get_peak_memory()
    memory_monitor.stop_monitoring()

    # Add experiment metadata
    results['experiment_info'] = {
        'type': 'compressed',
        'model_name': args.model_name,
        'config': config.__dict__,
        'evaluation_time': evaluation_time,
        'memory_stats': memory_stats,
        'timestamp': datetime.now().isoformat(),
        'args': vars(args)
    }

    # Save results
    with open(os.path.join(compression_dir, "compression_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Save model if requested
    if args.save_model:
        model_dir = os.path.join(output_dir, "saved_model")
        print(f"Saving compressed model to {model_dir}")
        model.save_pretrained(model_dir)

    return results


def main():
    """
    主程序流程
    
    執行步驟：
    1. 參數解析和實驗設置
    2. 配置日誌和輸出目錄
    3. 加載分詞器和創建配置
    4. 根據需求執行：
       - 基準實驗（無壓縮）
       - 主要壓縮實驗
    5. 生成實驗總結和結果保存
    """
    args = parse_arguments()

    # 設置實驗 (Setup experiment)
    if args.experiment_name is None:
        args.experiment_name = f"compression_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    setup_logging(os.path.join(output_dir, "experiment.log"))

    print("Real-time Prefill KV Cache Compression Experiment")
    print(f"Experiment: {args.experiment_name}")
    print(f"Model: {args.model_name}")
    print(f"Output directory: {output_dir}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create configuration
    config = create_experiment_config(args)

    # Save configuration
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump({
            'compression_config': config.__dict__,
            'args': vars(args)
        }, f, indent=2)

    results_summary = {}

    try:
        # Run baseline if requested
        if args.baseline:
            baseline_results = run_baseline_experiment(args, tokenizer, output_dir)
            results_summary['baseline'] = baseline_results

        # Run main compression experiment
        compression_results = run_compression_experiment(args, config, tokenizer, output_dir)
        results_summary['compressed'] = compression_results

        # Generate summary report
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)

        if 'compressed' in results_summary:
            comp_results = results_summary['compressed']
            print(f"Overall Quality Score: {comp_results.get('overall_quality_score', 'N/A'):.4f}")

            if 'compression_performance' in comp_results:
                comp_perf = comp_results['compression_performance']
                print(f"Memory Savings: {comp_perf.get('overall_avg_memory_savings', 0)*100:.1f}%")
                print(f"Compression Ratio: {comp_perf.get('overall_avg_compression_ratio', 1.0):.3f}")

        # Save final summary
        with open(os.path.join(output_dir, "experiment_summary.json"), 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()

        # Save error info
        with open(os.path.join(output_dir, "error.log"), 'w') as f:
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())

        return 1

    return 0


if __name__ == "__main__":
    exit(main())
