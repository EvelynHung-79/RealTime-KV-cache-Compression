#!/usr/bin/env python3
"""
Streaming KVQuant Quantization Experiment
Main script for running KV cache quantization experiments on LLaMA models
using streaming statistics with Sink/Outlier awareness, evaluated on LongBench.
"""

import os
import sys
import argparse
import json
import torch
import time
from datetime import datetime
from transformers import AutoTokenizer, LlamaForCausalLM # Import baseline model

# Add root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

# Import updated components
from configs.base_config import CompressionConfig
from src.models.modified_llama import create_compressed_llama_model # This now takes CompressionConfig
# Unified compressor removed as logic moved into model/stats_manager
# from src.compression.unified_compressor import RealTimePrefillCompressor
from src.evaluation.longbench_eval import LongBenchEvaluator
from src.utils.memory_utils import MemoryMonitor
from src.utils.eval_utils import setup_logging

def save_simplified_summary(args, results_summary, output_dir):
    """
    Consolidate Baseline and Quantized results into a single simplified summary file.
    Reports estimated memory savings based on quantization bits.
    """
    simplified_data = {
        'metadata': {
            'experiment_name': args.experiment_name,
            'model_name': args.model_name,
            'device': args.device,
            'timestamp': datetime.now().isoformat()
        },
        'hyperparameters': { # Updated hyperparameters
            'chunk_size': args.chunk_size,
            'ema_decay': args.ema_decay,
            'outlier_threshold_abs': args.outlier_threshold_abs,
            'outlier_threshold_relative': args.outlier_threshold_relative,
            'attention_sink_size': args.attention_sink_size,
            'key_bits_normal': args.key_bits_normal,
            'key_bits_sink_outlier': args.key_bits_sink_outlier,
            'value_bits_normal': args.value_bits_normal,
            'value_bits_sink_outlier': args.value_bits_sink_outlier,
            'value_quant_groups': args.value_quant_groups,
            'max_length': args.max_length,
        },
        'results': {}
    }

    # Helper to extract task breakdown safely
    def extract_task_breakdown(result):
        if result and 'task_breakdown' in result:
            # Structure: {'task_name': {'score': value}, ...}
            return {task: data.get('score', 'N/A')
                    for task, data in result['task_breakdown'].items() if isinstance(data, dict)}
        return {}

    # Process Baseline results
    if 'baseline' in results_summary and results_summary['baseline']:
        baseline_results = results_summary['baseline']
        simplified_data['results']['baseline'] = {
            'overall_quality_score': baseline_results.get('overall_quality_score', 'N/A'),
            'total_tasks': baseline_results.get('num_total_tasks', 0),
            'task_breakdown': extract_task_breakdown(baseline_results)
        }

    # Process Quantized results
    if 'quantized' in results_summary and results_summary['quantized']: # Renamed from 'compressed'
        quantized_results = results_summary['quantized']
        # Retrieve estimated savings from the new get_compression_stats structure
        comp_stats = quantized_results.get('overall_compression_stats', {}) # Fetch stats dict

        simplified_data['results']['quantized'] = {
            'overall_quality_score': quantized_results.get('overall_quality_score', 'N/A'),
            # Report estimated savings, not measured compression ratio
            'estimated_memory_savings': comp_stats.get('estimated_memory_savings', 'N/A'),
            'estimated_memory_ratio': comp_stats.get('estimated_memory_ratio', 'N/A'),
            'estimated_avg_bits_key': comp_stats.get('estimated_avg_bits_key', 'N/A'),
            'estimated_avg_bits_value': comp_stats.get('estimated_avg_bits_value', 'N/A'),
            'total_tasks': quantized_results.get('num_total_tasks', 0),
            'task_breakdown': extract_task_breakdown(quantized_results),
            'outlier_stats': comp_stats.get('outlier_stats', {}) # Add outlier info if available
        }

    # Save simplified results
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_data, f, indent=2, ensure_ascii=False)
        print(f"Simplified summary saved to: {summary_path}")
    except Exception as e:
        print(f"Error saving simplified summary: {e}")

    return simplified_data


def parse_arguments():
    """Parse command line arguments for Streaming KVQuant."""
    parser = argparse.ArgumentParser(description='Streaming KVQuant Quantization Experiment')

    # Model configuration
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf', help='HuggingFace model name')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--max_length', type=int, default=4096, help='Maximum sequence length for model')

    # Streaming and Chunking
    parser.add_argument('--chunk_size', type=int, default=256, help='Token chunk size for streaming prefill')

    # Streaming Statistics (EMA Absmax)
    parser.add_argument('--ema_decay', type=float, default=0.99, help='Decay factor for EMA Absmax statistics')

    # Outlier Detection (Optional, use None to disable)
    parser.add_argument('--outlier_threshold_abs', type=float, default=6.0, help='Absolute threshold for outlier detection (e.g., 6.0, or None)')
    parser.add_argument('--outlier_threshold_relative', type=float, default=5.0, help='Relative threshold (multiple of EMA) for outlier detection (e.g., 5.0, or None)')

    # Attention Sink Handling
    parser.add_argument('--attention_sink_size', type=int, default=8, help='Number of initial tokens treated as Attention Sink')

    # Quantization Settings
    parser.add_argument('--key_bits_normal', type=int, default=4, help='Bits for normal Key channels')
    parser.add_argument('--key_bits_sink_outlier', type=int, default=8, help='Bits for Key channels (Sink/Outlier)')
    parser.add_argument('--value_bits_normal', type=int, default=4, help='Bits for normal Value groups/channels')
    parser.add_argument('--value_bits_sink_outlier', type=int, default=8, help='Bits for Value groups/channels (Sink/Outlier)')
    parser.add_argument('--value_quant_groups', type=int, default=-1, help='Value quantization grouping (-1: per-channel, 1: per-tensor, >1: groups)')

    # Evaluation settings
    parser.add_argument('--tasks', nargs='+', default=None, help='LongBench tasks (default: narrativeqa, qasper, multifieldqa_en)')
    parser.add_argument('--max_samples', type=int, default=10, help='Max samples per task') # Reduced default for faster runs
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Max generated tokens per sample')

    # Experiment settings
    parser.add_argument('--output_dir', type=str, default='./experiments/results', help='Output directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name (default: timestamp)')
    parser.add_argument('--save_model', action='store_true', help='Save quantized model state_dict') # Note: Saving requires custom loading logic

    # Baseline comparison
    parser.add_argument('--baseline', action='store_true', help='Run baseline (FP16, no quantization)')

    # Default tasks if none provided
    parsed_args = parser.parse_args()
    if parsed_args.tasks is None:
        parsed_args.tasks = ['narrativeqa', 'qasper', 'multifieldqa_en'] # Default subset

    # Handle None for thresholds
    if parsed_args.outlier_threshold_abs == 'None':
        parsed_args.outlier_threshold_abs = None
    if parsed_args.outlier_threshold_relative == 'None':
         parsed_args.outlier_threshold_relative = None


    return parsed_args

def create_experiment_config(args) -> CompressionConfig:
    """Create CompressionConfig from parsed arguments."""
    # Note: CompressionConfig now includes quantization parameters directly
    config = CompressionConfig(
        model_name=args.model_name,
        max_position_embeddings=args.max_length,
        # Model structure params (num_layers etc.) might need to be loaded dynamically
        # Or added as arguments if testing different base models
        chunk_size=args.chunk_size,
        ema_decay=args.ema_decay,
        outlier_threshold_abs=args.outlier_threshold_abs,
        outlier_threshold_relative=args.outlier_threshold_relative,
        attention_sink_size=args.attention_sink_size,
        key_bits_normal=args.key_bits_normal,
        key_bits_sink_outlier=args.key_bits_sink_outlier,
        value_bits_normal=args.value_bits_normal,
        value_bits_sink_outlier=args.value_bits_sink_outlier,
        value_quant_groups=args.value_quant_groups,
        context_lengths=[args.max_length], # Keep for potential future use
        batch_sizes=[1] # Keep for potential future use
    )
    # Dynamically load num_layers, hidden_size etc. from actual model config
    from transformers import AutoConfig
    model_conf = AutoConfig.from_pretrained(args.model_name)
    config.num_hidden_layers = getattr(model_conf, 'num_hidden_layers', config.num_hidden_layers)
    config.hidden_size = getattr(model_conf, 'hidden_size', config.hidden_size)
    config.num_attention_heads = getattr(model_conf, 'num_attention_heads', config.num_attention_heads)
    # config.num_key_value_heads = getattr(model_conf, 'num_key_value_heads', config.num_attention_heads) # Handled in CompressionConfig post_init

    # Re-run post_init after updating model params
    config.__post_init__()

    return config

def run_baseline_experiment(args, tokenizer, output_dir):
    """Run baseline experiment (FP16, no quantization)."""
    print("\n" + "="*50)
    print("RUNNING BASELINE EXPERIMENT (FP16)")
    print("="*50)

    # Load baseline model (FP16)
    try:
        baseline_model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16, # Ensure FP16
            device_map="auto" # Use accelerate for auto device placement
        )
        baseline_model.eval()
        print(f"Baseline model {args.model_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading baseline model: {e}")
        return {'error': f"Failed to load baseline model: {e}"}

    # Create evaluator
    baseline_dir = os.path.join(output_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    # Use a dummy CompressionConfig for the evaluator (won't be used by baseline model)
    # But ensure necessary fields like max_position_embeddings are present
    dummy_config = CompressionConfig(model_name=args.model_name, max_position_embeddings=args.max_length)
    # Load model structure params into dummy config for evaluator if needed
    from transformers import AutoConfig
    model_conf = AutoConfig.from_pretrained(args.model_name)
    dummy_config.num_hidden_layers = getattr(model_conf, 'num_hidden_layers', dummy_config.num_hidden_layers)
    dummy_config.__post_init__()


    evaluator = LongBenchEvaluator(baseline_model, tokenizer, dummy_config, baseline_dir)

    # Run evaluation
    start_time = time.time()
    print(f"Starting baseline evaluation on tasks: {args.tasks} (max_samples={args.max_samples})")
    try:
        # Pass max_new_tokens to evaluator
        results = evaluator.evaluate_all_tasks(
            tasks=args.tasks,
            max_samples_per_task=args.max_samples
            # max_new_tokens=args.max_new_tokens # Ensure evaluator uses this
        )
    except Exception as e:
        print(f"Error during baseline evaluation: {e}")
        import traceback
        traceback.print_exc()
        results = {'error': f"Evaluation failed: {e}"}

    evaluation_time = time.time() - start_time
    print(f"Baseline evaluation finished in {evaluation_time:.2f} seconds.")

    # Clean up baseline model from GPU memory
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save results
    if 'error' not in results:
        results['experiment_info'] = {
            'type': 'baseline',
            'model_name': args.model_name,
            'evaluation_time': evaluation_time,
            'timestamp': datetime.now().isoformat()
        }
        # Save comprehensive results for baseline
        baseline_results_path = os.path.join(baseline_dir, "comprehensive_evaluation_results.json")
        try:
            with open(baseline_results_path, 'w', encoding='utf-8') as f:
                 # Need to structure dict correctly for comprehensive results
                 # Assuming evaluate_all_tasks returns the 'overall_results' dict directly
                 json.dump({'overall_results': results, 'task_results': {}}, f, indent=2, ensure_ascii=False) # Adjust if evaluator returns more detail
            print(f"Baseline comprehensive results saved to: {baseline_results_path}")
        except Exception as e:
            print(f"Error saving baseline comprehensive results: {e}")


    return results # Return the overall_results dict


def run_quantization_experiment(args, config, tokenizer, output_dir):
    """Run the main streaming quantization experiment."""
    print("\n" + "="*50)
    print("RUNNING STREAMING QUANTIZATION EXPERIMENT")
    print("="*50)

    # Initialize memory monitor
    memory_monitor = MemoryMonitor()
    memory_monitor.start_monitoring()

    # Create quantized model using the factory function
    print(f"Loading quantized model: {args.model_name} with config:")
    # Print key config parameters
    print(f"  chunk_size={config.chunk_size}, sink={config.attention_sink_size}")
    print(f"  key_bits={config.key_bits_normal}/{config.key_bits_sink_outlier}, value_bits={config.value_bits_normal}/{config.value_bits_sink_outlier}")

    try:
        # Pass the CompressionConfig object directly
        model = create_compressed_llama_model(args.model_name, config, args.device)
        print(f"Quantized model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"Error creating quantized model: {e}")
        memory_monitor.stop_monitoring()
        import traceback
        traceback.print_exc()
        return {'error': f"Failed to create quantized model: {e}"}


    # Create evaluator
    quantized_dir = os.path.join(output_dir, "quantized") # Renamed from "compressed"
    os.makedirs(quantized_dir, exist_ok=True)

    evaluator = LongBenchEvaluator(model, tokenizer, config, quantized_dir)

    # Run evaluation
    print(f"Starting quantized evaluation on tasks: {args.tasks} (max_samples={args.max_samples})")
    start_time = time.time()
    try:
        # Reset stats before evaluation
        if hasattr(model, 'reset_compression_state'):
            model.reset_compression_state()

        # Pass max_new_tokens to evaluator
        results = evaluator.evaluate_all_tasks(
            tasks=args.tasks,
            max_samples_per_task=args.max_samples
            # max_new_tokens=args.max_new_tokens # Ensure evaluator uses this
        )
    except Exception as e:
        print(f"Error during quantized evaluation: {e}")
        import traceback
        traceback.print_exc()
        results = {'error': f"Evaluation failed: {e}"}


    evaluation_time = time.time() - start_time
    memory_stats = memory_monitor.get_peak_memory()
    memory_monitor.stop_monitoring()
    print(f"Quantized evaluation finished in {evaluation_time:.2f} seconds.")


    overall_compression_stats = {}
    if hasattr(model, 'get_compression_stats'):
        overall_compression_stats = model.get_compression_stats() # Get new stats structure

    # Add experiment metadata to the main results dict returned by evaluator
    if 'error' not in results:
        results['experiment_info'] = {
            'type': 'quantized',
            'model_name': args.model_name,
            'config': config.__dict__.copy(), # Save the config used
            'evaluation_time': evaluation_time,
            'memory_stats': memory_stats,
            'timestamp': datetime.now().isoformat(),
            'args': vars(args).copy()
        }
        # Add the overall compression stats here
        results['overall_compression_stats'] = overall_compression_stats

        # Save comprehensive results for quantized run
        quantized_results_path = os.path.join(quantized_dir, "comprehensive_evaluation_results.json")
        try:
             with open(quantized_results_path, 'w', encoding='utf-8') as f:
                 # Assuming evaluate_all_tasks returns the 'overall_results' dict directly
                 # Merge overall_compression_stats into the saved dict
                 save_data = {'overall_results': results.copy(), 'task_results': {}} # Adjust structure as needed
                 if 'overall_compression_stats' in save_data['overall_results']:
                     # Merge stats into compression_performance field if expected by summary script
                     perf_key = 'compression_performance'
                     if perf_key not in save_data['overall_results']:
                         save_data['overall_results'][perf_key] = {}
                     save_data['overall_results'][perf_key].update(save_data['overall_results']['overall_compression_stats'])

                 json.dump(save_data, f, indent=2, default=lambda x: str(x) if isinstance(x, torch.Tensor) else x.__dict__ if hasattr(x, '__dict__') else str(x))

             print(f"Quantized comprehensive results saved to: {quantized_results_path}")
        except Exception as e:
            print(f"Error saving quantized comprehensive results: {e}")


    # Save model state_dict if requested (note: requires custom loading)
    if args.save_model:
        model_dir = os.path.join(output_dir, "saved_quantized_model")
        os.makedirs(model_dir, exist_ok=True)
        print(f"Saving quantized model state_dict to {model_dir}")
        # Only save state_dict, loading requires re-instantiating the custom model first
        torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
        # Save the compression config alongside
        with open(os.path.join(model_dir, "compression_config.json"), 'w') as f:
            json.dump(config.__dict__, f, indent=2)


    # Clean up quantized model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results # Return the overall_results dict


def main():
    """Main execution flow."""
    args = parse_arguments()

    # Setup experiment directory and name
    if args.experiment_name is None:
        args.experiment_name = f"stream_quant_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(output_dir, "experiment.log")
    setup_logging(log_file)

    print("Streaming KVQuant Quantization Experiment")
    print(f"Experiment: {args.experiment_name}")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Output directory: {output_dir}")
    print(f"Log file: {log_file}")

    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return 1

    # Create configuration
    try:
        config = create_experiment_config(args)
    except Exception as e:
         print(f"Error creating experiment config: {e}")
         return 1

    # Save configuration
    config_path = os.path.join(output_dir, "config.json")
    try:
        with open(config_path, 'w') as f:
            # Use default=str for potentially non-serializable items if needed
            json.dump({
                'compression_config': config.__dict__,
                'args': vars(args)
            }, f, indent=2, default=str)
        print(f"Configuration saved to: {config_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")


    results_summary = {}

    try:
        # Run baseline if requested
        if args.baseline:
            baseline_overall_results = run_baseline_experiment(args, tokenizer, output_dir)
            results_summary['baseline'] = baseline_overall_results

        # Run main quantization experiment
        quantized_overall_results = run_quantization_experiment(args, config, tokenizer, output_dir)
        results_summary['quantized'] = quantized_overall_results

        # Save simplified summary comparing baseline and quantized
        simplified_summary = save_simplified_summary(args, results_summary, output_dir)

        # Generate console summary report
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)

        if 'quantized' in simplified_summary['results'] and simplified_summary['results']['quantized']:
            quant_res = simplified_summary['results']['quantized']
            print(f"Overall Quality Score (Quantized): {quant_res.get('overall_quality_score', 'N/A')}")
            print(f"  Estimated Memory Savings: {quant_res.get('estimated_memory_savings', 'N/A')}")
            print(f"  Estimated Memory Ratio: {quant_res.get('estimated_memory_ratio', 'N/A')}")
            # Add outlier reporting if available
            if 'outlier_stats' in quant_res and quant_res['outlier_stats']:
                 print(f"  Outlier Stats: {quant_res['outlier_stats']}")

        if 'baseline' in simplified_summary['results'] and simplified_summary['results']['baseline']:
             base_res = simplified_summary['results']['baseline']
             print(f"Overall Quality Score (Baseline): {base_res.get('overall_quality_score', 'N/A')}")


        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()

        # Save error info
        error_log_path = os.path.join(output_dir, "error.log")
        with open(error_log_path, 'w') as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())
        print(f"Error details saved to: {error_log_path}")

        return 1 # Indicate failure

    return 0 # Indicate success


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)