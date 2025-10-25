import torch
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from itertools import product
from transformers import AutoTokenizer
import argparse
import logging

# --- 路徑設置 (確保能找到 src 和 configs) ---
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
# ---

from src.models.modified_llama import create_compressed_llama_model
from configs.base_config import CompressionConfig # 引用更新後的 Config
from src.evaluation.longbench_eval import LongBenchEvaluator
from src.utils.memory_utils import MemoryMonitor
from src.utils.eval_utils import setup_logging # 加入日誌

# --- Setup Logging ---
# Note: It might be better to configure logging once in the main entry point
# setup_logging(level=logging.INFO)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AblationStudy:
    """
    Ablation study for Streaming KVQuant Quantization (Sink/Outlier Aware).
    Focuses on parameters like chunk size, EMA decay, sink size, bits, and outlier handling.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cuda",
        output_dir: str = "./ablation_results"
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.results = {} # Store results per study type

        os.makedirs(output_dir, exist_ok=True)
        setup_logging(os.path.join(output_dir, "ablation_study.log")) # Setup logging to file
        logging.info(f"Initialized AblationStudy for model: {model_name} on device: {device}")

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                logging.warning("Tokenizer does not have a pad token, setting to eos_token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info("Tokenizer loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load tokenizer for {model_name}: {e}")
            raise

    def define_ablation_dimensions(self) -> Dict[str, List]:
        """Define the dimensions and values for the new ablation study parameters."""
        logging.info("Defining ablation dimensions...")
        return {
            # Streaming and Chunking
            'chunk_size': [128, 256, 512],

            # Streaming Statistics (EMA Absmax)
            'ema_decay': [0.9, 0.99, 0.999],

            # Outlier Detection (Test disabling or varying thresholds)
            # Use tuples: (abs_thresh, rel_thresh). None disables that threshold.
            'outlier_thresholds': [(None, None), (6.0, None), (None, 5.0), (6.0, 5.0)],

            # Attention Sink Handling
            'attention_sink_size': [0, 4, 8, 16], # 0 disables sink handling

            # Quantization Bits (Test different precision levels)
            # Use tuples: (key_norm, key_sink/outlier, val_norm, val_sink/outlier)
            'quantization_bits': [
                (8, 8, 8, 8), # Baseline: INT8 only
                (4, 8, 4, 8), # Target: INT4 normal, INT8 special
                (4, 4, 4, 4), # INT4 only
                (2, 4, 2, 4), # Experimental: INT2 normal, INT4 special (might need specific kernels)
            ],

            # Value Quantization Grouping
            'value_quant_groups': [-1, 64, 128] # -1 = per-channel, 64/128 = group size (adjust if not divisible by head_dim)
        }

    def create_baseline_config(self) -> CompressionConfig:
        """Create a reasonable baseline configuration for the new architecture."""
        logging.info("Creating baseline configuration...")
        # Adjust defaults as needed based on initial experiments
        return CompressionConfig(
            model_name=self.model_name,
            chunk_size=256,
            ema_decay=0.99,
            outlier_threshold_abs=6.0,
            outlier_threshold_relative=5.0,
            attention_sink_size=8,
            key_bits_normal=4,
            key_bits_sink_outlier=8,
            value_bits_normal=4,
            value_bits_sink_outlier=8,
            value_quant_groups=-1 # Default to per-channel for value initially
        )

    def evaluate_single_config(
        self,
        config: CompressionConfig,
        config_name: str,
        quick_eval: bool = True
    ) -> Dict[str, Any]:
        """Evaluate a single configuration (remains largely the same)."""
        logging.info(f"Evaluating config: {config_name}")

        model = None # Ensure model is defined for cleanup
        try:
            # Create model
            model = create_compressed_llama_model(self.model_name, config, self.device)
            model.eval()

            # Create evaluator
            eval_dir = os.path.join(self.output_dir, f"eval_{config_name}")
            os.makedirs(eval_dir, exist_ok=True)

            evaluator = LongBenchEvaluator(model, self.tokenizer, config, eval_dir)

            # Use subset of tasks for quick evaluation
            if quick_eval:
                # Use fewer tasks/samples for faster ablation
                test_tasks = ['narrativeqa'] # Reduced further for speed
                max_samples = 5 # Reduced further for speed
                max_new_tokens = 30 # Reduced further for speed
                logging.info(f"Performing quick evaluation: tasks={test_tasks}, max_samples={max_samples}")
            else:
                test_tasks = ['narrativeqa', 'qasper'] # Example full eval tasks
                max_samples = 20
                max_new_tokens = 50
                logging.info(f"Performing full evaluation: tasks={test_tasks}, max_samples={max_samples}")


            # Monitor memory during evaluation
            memory_monitor = MemoryMonitor()
            memory_monitor.start_monitoring()

            # Run evaluation
            eval_results = evaluator.evaluate_all_tasks(
                tasks=test_tasks,
                max_samples_per_task=max_samples,
                max_new_tokens = max_new_tokens # Pass max_new_tokens
            )

            memory_stats = memory_monitor.get_peak_memory()
            memory_monitor.stop_monitoring()

            # Get compression statistics (adapt based on what get_compression_stats returns now)
            compression_stats = {}
            if hasattr(model, 'get_compression_stats'):
                compression_stats = model.get_compression_stats() # This needs to be updated

            # Compile results
            result = {
                'config_name': config_name,
                'config_params': config.__dict__.copy(), # Store actual params used
                'evaluation_results': eval_results,
                'memory_stats': memory_stats,
                'compression_stats': compression_stats # Store whatever stats are available
            }
            logging.info(f"Evaluation successful for {config_name}. Quality score: {eval_results.get('overall_quality_score', 'N/A')}")
            return result

        except Exception as e:
            logging.error(f"Error evaluating config {config_name}: {e}", exc_info=True) # Log traceback
            return {'config_name': config_name, 'error': str(e)}
        finally:
            # Clean up GPU memory
            if model is not None:
                del model
            torch.cuda.empty_cache()
            logging.debug(f"Cleaned up resources for {config_name}")


    def run_parameter_ablation(self, param_name: str, study_name: str):
        """Generic function to run ablation on a single parameter."""
        logging.info(f"\n{'='*50}\nRUNNING ABLATION STUDY: {study_name.upper()}\n{'='*50}")

        base_config = self.create_baseline_config()
        dimensions = self.define_ablation_dimensions()

        if param_name not in dimensions:
            logging.error(f"Parameter '{param_name}' not found in ablation dimensions.")
            return {}

        results = {}
        param_values = dimensions[param_name]

        # Always evaluate the baseline config first
        baseline_name = "baseline_config"
        logging.info(f"Evaluating baseline: {baseline_name}")
        try:
             results[baseline_name] = self.evaluate_single_config(base_config, baseline_name, quick_eval=True)
        except Exception as e:
             logging.error(f"Error evaluating baseline {baseline_name}: {e}", exc_info=True)
             results[baseline_name] = {'error': str(e)}


        for i, value in enumerate(param_values):
            # Create a unique, descriptive name
            if isinstance(value, tuple):
                config_name = f"{param_name}_" + "_".join(map(str, value))
            else:
                config_name = f"{param_name}_{value}"
            # Sanitize name if needed (e.g., replace None)
            config_name = config_name.replace("None","Disabled")

            # Check if this value IS the baseline value, skip re-evaluation if so
            if hasattr(base_config, param_name) and getattr(base_config, param_name) == value:
                 logging.info(f"Skipping re-evaluation for baseline value: {config_name}")
                 # Optionally link to baseline results or just note it
                 results[config_name] = {"status": "matches_baseline", "baseline_ref": baseline_name}
                 continue

            logging.info(f"\nTesting {config_name} ({i+1}/{len(param_values)})")

            # Create configuration by modifying the baseline
            config = self.create_baseline_config() # Start fresh from baseline
            try:
                # Handle special cases like tuples
                if param_name == 'outlier_thresholds':
                    setattr(config, 'outlier_threshold_abs', value[0])
                    setattr(config, 'outlier_threshold_relative', value[1])
                elif param_name == 'quantization_bits':
                    setattr(config, 'key_bits_normal', value[0])
                    setattr(config, 'key_bits_sink_outlier', value[1])
                    setattr(config, 'value_bits_normal', value[2])
                    setattr(config, 'value_bits_sink_outlier', value[3])
                else:
                    setattr(config, param_name, value)

                # Re-run __post_init__ if necessary (e.g., for value_quant_groups validation)
                if hasattr(config, '__post_init__'):
                    config.__post_init__()

                result = self.evaluate_single_config(config, config_name, quick_eval=True)
                results[config_name] = result

            except Exception as e:
                logging.error(f"Error testing {config_name}: {e}", exc_info=True)
                results[config_name] = {'error': str(e)}

        # Save results for this specific study
        output_filename = os.path.join(self.output_dir, f"{study_name}_ablation.json")
        try:
            with open(output_filename, 'w') as f:
                json.dump(results, f, indent=2, default=lambda o: '<not serializable>') # Handle non-serializable parts
            logging.info(f"Saved {study_name} results to {output_filename}")
        except Exception as e:
            logging.error(f"Failed to save results for {study_name}: {e}")

        # Analyze results
        self.analyze_results(results, study_name)

        return results


    def analyze_results(self, results: Dict[str, Any], study_name: str):
        """Analyze ablation results and save to CSV."""
        logging.info(f"\nAnalyzing {study_name} results...")
        analysis_data = []

        baseline_score = float('-inf')
        baseline_memory = float('-inf') # Use theoretical memory saving estimate

        # Find baseline result
        baseline_result = results.get("baseline_config")
        if baseline_result and 'error' not in baseline_result:
             baseline_score = baseline_result.get('evaluation_results', {}).get('overall_quality_score', 0)
             # Update how memory saving is retrieved based on new stats
             baseline_memory = baseline_result.get('compression_stats', {}).get('estimated_memory_savings', 0) # Example key
             logging.info(f"Baseline - Score: {baseline_score:.4f}, Est. Memory Savings: {baseline_memory*100:.1f}%")
        else:
             logging.warning("Could not find valid baseline result for comparison.")


        for config_name, result in results.items():
            if 'error' in result or result.get("status") == "matches_baseline":
                logging.warning(f"Skipping analysis for {config_name} due to error or being baseline.")
                continue

            eval_results = result.get('evaluation_results', {})
            quality_score = eval_results.get('overall_quality_score', 0)

            # Update how memory saving is retrieved
            comp_stats = result.get('compression_stats', {})
            memory_savings = comp_stats.get('estimated_memory_savings', 0) # Example key
            # Add other relevant stats if available
            key_outliers = comp_stats.get('key_outliers_detected', 'N/A')
            val_outliers = comp_stats.get('value_outliers_detected', 'N/A')


            # Calculate differences from baseline
            quality_diff = quality_score - baseline_score if baseline_score > float('-inf') else 0
            memory_diff = memory_savings - baseline_memory if baseline_memory > float('-inf') else 0

            # Store config parameters alongside results
            params = result.get('config_params', {})

            analysis_entry = {
                'config_name': config_name,
                'quality_score': quality_score,
                'est_memory_savings': memory_savings,
                'quality_vs_baseline': quality_diff,
                'memory_vs_baseline': memory_diff,
                'key_outliers': key_outliers,
                'val_outliers': val_outliers,
                **params # Add all config params used
            }
            analysis_data.append(analysis_entry)

        if analysis_data:
            analysis_df = pd.DataFrame(analysis_data)
            # Sort by quality score or a combined metric
            analysis_df = analysis_df.sort_values(by='quality_score', ascending=False)

            output_csv = os.path.join(self.output_dir, f"{study_name}_analysis.csv")
            try:
                analysis_df.to_csv(output_csv, index=False)
                logging.info(f"Saved analysis for {study_name} to {output_csv}")
                print(f"\n--- Top 3 Configs for {study_name} (by Quality) ---")
                print(analysis_df.head(3).to_string(index=False))
                print("-"*(len(study_name) + 20))

            except Exception as e:
                logging.error(f"Failed to save analysis CSV for {study_name}: {e}")
        else:
            logging.warning(f"No valid results to analyze for {study_name}.")


    def run_all_ablations(self) -> Dict[str, Any]:
        """Run all defined ablation studies sequentially."""
        logging.info("\n" + "="*60 + "\nSTARTING ALL ABLATION STUDIES\n" + "="*60)
        all_results = {}
        dimensions = self.define_ablation_dimensions()

        # Define which studies to run
        studies_to_run = {
            "chunk_size": "Chunk Size",
            "ema_decay": "EMA Decay",
            "outlier_thresholds": "Outlier Thresholds",
            "attention_sink_size": "Attention Sink Size",
            "quantization_bits": "Quantization Bits",
            "value_quant_groups": "Value Quant Grouping"
        }

        for param_key, study_title in studies_to_run.items():
            if param_key in dimensions:
                 study_results = self.run_parameter_ablation(param_key, study_title.lower().replace(" ", "_"))
                 all_results[study_title] = study_results
            else:
                 logging.warning(f"Skipping ablation for '{param_key}' as it's not defined in dimensions.")


        # Save comprehensive results (optional, can be large)
        # comprehensive_output = os.path.join(self.output_dir, "comprehensive_ablation_results.json")
        # try:
        #     with open(comprehensive_output, 'w') as f:
        #         json.dump(all_results, f, indent=2, default=lambda o: '<not serializable>')
        #     logging.info(f"Saved comprehensive results to {comprehensive_output}")
        # except Exception as e:
        #     logging.error(f"Failed to save comprehensive results: {e}")

        logging.info("\n" + "="*60 + "\nALL ABLATION STUDIES COMPLETED\n" + "="*60)
        return all_results

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Run ablation study for Streaming KVQuant compression")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="HuggingFace model name/path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="./ablation_results", help="Directory to save results")
    parser.add_argument("--study_type", type=str, choices=[
        'chunk_size', 'ema_decay', 'outlier_thresholds', 'attention_sink_size',
        'quantization_bits', 'value_quant_groups', 'all'
    ], default='all', help="Which ablation study to run ('all' runs sequentially)")

    args = parser.parse_args()

    # Create ablation study runner
    ablation = AblationStudy(
        model_name=args.model_name,
        device=args.device,
        output_dir=args.output_dir
    )

    # Run specified study
    if args.study_type == 'all':
        ablation.run_all_ablations()
    else:
        # Map study_type argument to parameter key and study name
        study_map = {
            "chunk_size": ("chunk_size", "Chunk Size"),
            "ema_decay": ("ema_decay", "EMA Decay"),
            "outlier_thresholds": ("outlier_thresholds", "Outlier Thresholds"),
            "attention_sink_size": ("attention_sink_size", "Attention Sink Size"),
            "quantization_bits": ("quantization_bits", "Quantization Bits"),
            "value_quant_groups": ("value_quant_groups", "Value Quant Grouping")
        }
        if args.study_type in study_map:
             param_key, study_title = study_map[args.study_type]
             ablation.run_parameter_ablation(param_key, study_title.lower().replace(" ", "_"))
        else:
             logging.error(f"Unknown study type: {args.study_type}")


    logging.info(f"\nAblation study completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()