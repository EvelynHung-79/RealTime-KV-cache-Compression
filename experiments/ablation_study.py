import torch
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Any
from itertools import product
import pandas as pd
from transformers import AutoTokenizer

from ..models.modified_llama import create_compressed_llama_model  
from ..configs.base_config import CompressionConfig
from ..evaluation.longbench_eval import LongBenchEvaluator
from ..utils.memory_utils import MemoryMonitor

class AblationStudy:
    """
    Comprehensive ablation study for Real-time Prefill KV Cache Compression
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
        self.results = {}

        os.makedirs(output_dir, exist_ok=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def define_ablation_dimensions(self) -> Dict[str, List]:
        """Define the dimensions and values for ablation study"""

        return {
            # Importance scoring weights (α, β, γ)
            'alpha': [0.2, 0.4, 0.6, 0.8],  # Prompt attention weight
            'beta': [0.1, 0.2, 0.3, 0.4],   # Position bias weight
            'gamma': [0.1, 0.2, 0.3, 0.4],  # Context relevance weight

            # Precision thresholds
            'theta_h': [0.6, 0.7, 0.8],     # High precision threshold
            'theta_m': [0.2, 0.3, 0.4],     # Medium precision threshold

            # Layer propagation ratios
            'early_layer_ratio': [0.7, 0.8, 0.9],
            'middle_layer_ratio': [0.5, 0.6, 0.7],
            'later_layer_ratio': [0.3, 0.4, 0.5],

            # Quantization bits
            'high_precision_bits': [8, 16],
            'medium_precision_bits': [4, 8],
            'low_precision_bits': [2, 4]
        }

    def create_baseline_config(self) -> CompressionConfig:
        """Create baseline configuration"""
        return CompressionConfig(
            model_name=self.model_name,
            alpha=0.4, beta=0.3, gamma=0.3,
            theta_h=0.7, theta_m=0.3,
            early_layer_ratio=0.8,
            middle_layer_ratio=0.6,
            later_layer_ratio=0.4
        )

    def run_importance_weights_ablation(self) -> Dict[str, Any]:
        """Ablation study on importance scoring weights (α, β, γ)"""

        print("\n" + "="*50)
        print("IMPORTANCE WEIGHTS ABLATION STUDY")
        print("="*50)

        base_config = self.create_baseline_config()
        dimensions = self.define_ablation_dimensions()

        # Test different combinations of α, β, γ (ensuring they sum to 1.0)
        weight_combinations = []

        for alpha in dimensions['alpha']:
            for beta in dimensions['beta']:
                gamma = 1.0 - alpha - beta
                if 0.1 <= gamma <= 0.4:  # Ensure gamma is in reasonable range
                    weight_combinations.append((alpha, beta, gamma))

        results = {}

        for i, (alpha, beta, gamma) in enumerate(weight_combinations):
            config_name = f"weights_a{alpha:.1f}_b{beta:.1f}_g{gamma:.1f}"
            print(f"\nTesting {config_name} ({i+1}/{len(weight_combinations)})")

            # Create configuration
            config = CompressionConfig(
                model_name=self.model_name,
                alpha=alpha, beta=beta, gamma=gamma,
                theta_h=base_config.theta_h,
                theta_m=base_config.theta_m,
                early_layer_ratio=base_config.early_layer_ratio,
                middle_layer_ratio=base_config.middle_layer_ratio,
                later_layer_ratio=base_config.later_layer_ratio
            )

            try:
                result = self.evaluate_single_config(config, config_name, quick_eval=True)
                results[config_name] = result

            except Exception as e:
                print(f"Error testing {config_name}: {e}")
                results[config_name] = {'error': str(e)}

        # Save results
        with open(os.path.join(self.output_dir, "importance_weights_ablation.json"), 'w') as f:
            json.dump(results, f, indent=2)

        # Analyze results
        self.analyze_importance_weights_results(results)

        return results

    def run_precision_thresholds_ablation(self) -> Dict[str, Any]:
        """Ablation study on precision assignment thresholds"""

        print("\n" + "="*50)
        print("PRECISION THRESHOLDS ABLATION STUDY")
        print("="*50)

        base_config = self.create_baseline_config()
        dimensions = self.define_ablation_dimensions()

        results = {}

        for theta_h in dimensions['theta_h']:
            for theta_m in dimensions['theta_m']:
                if theta_m < theta_h:  # Ensure logical ordering
                    config_name = f"thresholds_h{theta_h:.1f}_m{theta_m:.1f}"
                    print(f"\nTesting {config_name}")

                    config = CompressionConfig(
                        model_name=self.model_name,
                        alpha=base_config.alpha,
                        beta=base_config.beta,
                        gamma=base_config.gamma,
                        theta_h=theta_h,
                        theta_m=theta_m,
                        early_layer_ratio=base_config.early_layer_ratio,
                        middle_layer_ratio=base_config.middle_layer_ratio,
                        later_layer_ratio=base_config.later_layer_ratio
                    )

                    try:
                        result = self.evaluate_single_config(config, config_name, quick_eval=True)
                        results[config_name] = result

                    except Exception as e:
                        print(f"Error testing {config_name}: {e}")
                        results[config_name] = {'error': str(e)}

        # Save results
        with open(os.path.join(self.output_dir, "precision_thresholds_ablation.json"), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_propagation_ratios_ablation(self) -> Dict[str, Any]:
        """Ablation study on layer propagation ratios"""

        print("\n" + "="*50)
        print("PROPAGATION RATIOS ABLATION STUDY")
        print("="*50)

        base_config = self.create_baseline_config()
        dimensions = self.define_ablation_dimensions()

        results = {}

        for early_ratio in dimensions['early_layer_ratio']:
            for middle_ratio in dimensions['middle_layer_ratio']:
                for later_ratio in dimensions['later_layer_ratio']:
                    # Ensure logical ordering: early >= middle >= later
                    if early_ratio >= middle_ratio >= later_ratio:
                        config_name = f"ratios_e{early_ratio:.1f}_m{middle_ratio:.1f}_l{later_ratio:.1f}"
                        print(f"\nTesting {config_name}")

                        config = CompressionConfig(
                            model_name=self.model_name,
                            alpha=base_config.alpha,
                            beta=base_config.beta,
                            gamma=base_config.gamma,
                            theta_h=base_config.theta_h,
                            theta_m=base_config.theta_m,
                            early_layer_ratio=early_ratio,
                            middle_layer_ratio=middle_ratio,
                            later_layer_ratio=later_ratio
                        )

                        try:
                            result = self.evaluate_single_config(config, config_name, quick_eval=True)
                            results[config_name] = result

                        except Exception as e:
                            print(f"Error testing {config_name}: {e}")
                            results[config_name] = {'error': str(e)}

        # Save results
        with open(os.path.join(self.output_dir, "propagation_ratios_ablation.json"), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_quantization_bits_ablation(self) -> Dict[str, Any]:
        """Ablation study on quantization bit configurations"""

        print("\n" + "="*50)
        print("QUANTIZATION BITS ABLATION STUDY")
        print("="*50)

        base_config = self.create_baseline_config()
        dimensions = self.define_ablation_dimensions()

        results = {}

        for high_bits in dimensions['high_precision_bits']:
            for medium_bits in dimensions['medium_precision_bits']:
                for low_bits in dimensions['low_precision_bits']:
                    # Ensure logical ordering: high >= medium >= low
                    if high_bits >= medium_bits >= low_bits:
                        config_name = f"bits_h{high_bits}_m{medium_bits}_l{low_bits}"
                        print(f"\nTesting {config_name}")

                        config = CompressionConfig(
                            model_name=self.model_name,
                            alpha=base_config.alpha,
                            beta=base_config.beta,
                            gamma=base_config.gamma,
                            theta_h=base_config.theta_h,
                            theta_m=base_config.theta_m,
                            early_layer_ratio=base_config.early_layer_ratio,
                            middle_layer_ratio=base_config.middle_layer_ratio,
                            later_layer_ratio=base_config.later_layer_ratio,
                            high_precision_bits=high_bits,
                            medium_precision_bits=medium_bits,
                            low_precision_bits=low_bits
                        )

                        try:
                            result = self.evaluate_single_config(config, config_name, quick_eval=True)
                            results[config_name] = result

                        except Exception as e:
                            print(f"Error testing {config_name}: {e}")
                            results[config_name] = {'error': str(e)}

        # Save results
        with open(os.path.join(self.output_dir, "quantization_bits_ablation.json"), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_component_ablation(self) -> Dict[str, Any]:
        """Ablation study disabling individual components"""

        print("\n" + "="*50)
        print("COMPONENT ABLATION STUDY")
        print("="*50)

        base_config = self.create_baseline_config()

        component_configs = {
            "full_system": base_config,

            "no_prompt_attention": CompressionConfig(
                model_name=self.model_name,
                alpha=0.0, beta=0.5, gamma=0.5,  # Disable prompt attention
                theta_h=base_config.theta_h, theta_m=base_config.theta_m,
                early_layer_ratio=base_config.early_layer_ratio,
                middle_layer_ratio=base_config.middle_layer_ratio,
                later_layer_ratio=base_config.later_layer_ratio
            ),

            "no_position_bias": CompressionConfig(
                model_name=self.model_name,
                alpha=0.7, beta=0.0, gamma=0.3,  # Disable position bias
                theta_h=base_config.theta_h, theta_m=base_config.theta_m,
                early_layer_ratio=base_config.early_layer_ratio,
                middle_layer_ratio=base_config.middle_layer_ratio,
                later_layer_ratio=base_config.later_layer_ratio
            ),

            "no_context_relevance": CompressionConfig(
                model_name=self.model_name,
                alpha=0.6, beta=0.4, gamma=0.0,  # Disable context relevance
                theta_h=base_config.theta_h, theta_m=base_config.theta_m,
                early_layer_ratio=base_config.early_layer_ratio,
                middle_layer_ratio=base_config.middle_layer_ratio,
                later_layer_ratio=base_config.later_layer_ratio
            ),

            "uniform_precision": CompressionConfig(
                model_name=self.model_name,
                alpha=base_config.alpha, beta=base_config.beta, gamma=base_config.gamma,
                theta_h=1.0, theta_m=1.0,  # All tokens get same precision
                early_layer_ratio=base_config.early_layer_ratio,
                middle_layer_ratio=base_config.middle_layer_ratio,
                later_layer_ratio=base_config.later_layer_ratio
            ),

            "uniform_propagation": CompressionConfig(
                model_name=self.model_name,
                alpha=base_config.alpha, beta=base_config.beta, gamma=base_config.gamma,
                theta_h=base_config.theta_h, theta_m=base_config.theta_m,
                early_layer_ratio=0.6, middle_layer_ratio=0.6, later_layer_ratio=0.6  # Same ratio for all layers
            ),

            "no_quantization": CompressionConfig(
                model_name=self.model_name,
                alpha=base_config.alpha, beta=base_config.beta, gamma=base_config.gamma,
                theta_h=base_config.theta_h, theta_m=base_config.theta_m,
                early_layer_ratio=base_config.early_layer_ratio,
                middle_layer_ratio=base_config.middle_layer_ratio,
                later_layer_ratio=base_config.later_layer_ratio,
                high_precision_bits=16, medium_precision_bits=16, low_precision_bits=16  # No quantization
            )
        }

        results = {}

        for config_name, config in component_configs.items():
            print(f"\nTesting {config_name}")

            try:
                result = self.evaluate_single_config(config, config_name, quick_eval=True)
                results[config_name] = result

            except Exception as e:
                print(f"Error testing {config_name}: {e}")
                results[config_name] = {'error': str(e)}

        # Save results
        with open(os.path.join(self.output_dir, "component_ablation.json"), 'w') as f:
            json.dump(results, f, indent=2)

        # Analyze component contributions
        self.analyze_component_ablation_results(results)

        return results

    def evaluate_single_config(
        self,
        config: CompressionConfig,
        config_name: str,
        quick_eval: bool = True
    ) -> Dict[str, Any]:
        """Evaluate a single configuration"""

        # Create model
        model = create_compressed_llama_model(self.model_name, config, self.device)
        model.eval()

        # Create evaluator
        eval_dir = os.path.join(self.output_dir, f"eval_{config_name}")
        os.makedirs(eval_dir, exist_ok=True)

        evaluator = LongBenchEvaluator(model, self.tokenizer, config, eval_dir)

        # Use subset of tasks for quick evaluation
        if quick_eval:
            test_tasks = ['narrativeqa', 'qasper']
            max_samples = 10
        else:
            test_tasks = None  # All tasks
            max_samples = 50

        # Monitor memory during evaluation
        memory_monitor = MemoryMonitor()
        memory_monitor.start_monitoring()

        # Run evaluation
        results = evaluator.evaluate_all_tasks(
            tasks=test_tasks,
            max_samples_per_task=max_samples
        )

        memory_stats = memory_monitor.get_peak_memory()
        memory_monitor.stop_monitoring()

        # Get compression statistics
        compression_stats = {}
        if hasattr(model, 'get_compression_stats'):
            compression_stats = model.get_compression_stats()

        # Clean up
        del model
        torch.cuda.empty_cache()

        # Compile results
        result = {
            'config_name': config_name,
            'config': config.__dict__.copy(),
            'evaluation_results': results,
            'memory_stats': memory_stats,
            'compression_stats': compression_stats
        }

        return result

    def analyze_importance_weights_results(self, results: Dict[str, Any]):
        """Analyze importance weights ablation results"""

        print("\nAnalyzing importance weights results...")

        # Extract performance metrics
        analysis_data = []

        for config_name, result in results.items():
            if 'error' in result:
                continue

            # Parse config name to extract weights
            parts = config_name.split('_')
            alpha = float(parts[1][1:])  # Skip 'a' prefix
            beta = float(parts[2][1:])   # Skip 'b' prefix
            gamma = float(parts[3][1:])  # Skip 'g' prefix

            eval_results = result.get('evaluation_results', {})
            quality_score = eval_results.get('overall_quality_score', 0)

            compression_perf = eval_results.get('compression_performance', {})
            memory_savings = compression_perf.get('overall_avg_memory_savings', 0)

            analysis_data.append({
                'config_name': config_name,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'quality_score': quality_score,
                'memory_savings': memory_savings,
                'combined_score': quality_score * (1 + memory_savings)  # Quality-weighted efficiency
            })

        # Find best configurations
        if analysis_data:
            # Best quality
            best_quality = max(analysis_data, key=lambda x: x['quality_score'])
            print(f"Best Quality: {best_quality['config_name']} - Score: {best_quality['quality_score']:.4f}")

            # Best memory savings
            best_memory = max(analysis_data, key=lambda x: x['memory_savings'])
            print(f"Best Memory: {best_memory['config_name']} - Savings: {best_memory['memory_savings']*100:.1f}%")

            # Best combined
            best_combined = max(analysis_data, key=lambda x: x['combined_score'])
            print(f"Best Combined: {best_combined['config_name']} - Score: {best_combined['combined_score']:.4f}")

            # Save analysis
            analysis_df = pd.DataFrame(analysis_data)
            analysis_df.to_csv(os.path.join(self.output_dir, "importance_weights_analysis.csv"), index=False)

    def analyze_component_ablation_results(self, results: Dict[str, Any]):
        """Analyze component ablation results"""

        print("\nAnalyzing component ablation results...")

        # Get baseline (full system) performance
        baseline_result = results.get('full_system')
        if not baseline_result or 'error' in baseline_result:
            print("No valid baseline result found")
            return

        baseline_quality = baseline_result['evaluation_results'].get('overall_quality_score', 0)
        baseline_memory = baseline_result['evaluation_results'].get('compression_performance', {}).get('overall_avg_memory_savings', 0)

        print(f"\nBaseline Performance:")
        print(f"  Quality Score: {baseline_quality:.4f}")
        print(f"  Memory Savings: {baseline_memory*100:.1f}%")

        print("\nComponent Contributions (vs baseline):")

        for config_name, result in results.items():
            if config_name == 'full_system' or 'error' in result:
                continue

            quality = result['evaluation_results'].get('overall_quality_score', 0)
            memory = result['evaluation_results'].get('compression_performance', {}).get('overall_avg_memory_savings', 0)

            quality_diff = baseline_quality - quality
            memory_diff = baseline_memory - memory

            print(f"  {config_name}:")
            print(f"    Quality Drop: {quality_diff:+.4f}")
            print(f"    Memory Drop: {memory_diff*100:+.1f}%")

    def run_comprehensive_ablation(self) -> Dict[str, Any]:
        """Run comprehensive ablation study"""

        print("\n" + "="*60)
        print("COMPREHENSIVE ABLATION STUDY")
        print("="*60)

        all_results = {}

        # Run individual ablation studies
        try:
            print("\n1. Component Ablation...")
            all_results['component_ablation'] = self.run_component_ablation()
        except Exception as e:
            print(f"Component ablation failed: {e}")

        try:
            print("\n2. Importance Weights Ablation...")
            all_results['importance_weights_ablation'] = self.run_importance_weights_ablation()
        except Exception as e:
            print(f"Importance weights ablation failed: {e}")

        try:
            print("\n3. Precision Thresholds Ablation...")
            all_results['precision_thresholds_ablation'] = self.run_precision_thresholds_ablation()
        except Exception as e:
            print(f"Precision thresholds ablation failed: {e}")

        try:
            print("\n4. Propagation Ratios Ablation...")
            all_results['propagation_ratios_ablation'] = self.run_propagation_ratios_ablation()
        except Exception as e:
            print(f"Propagation ratios ablation failed: {e}")

        try:
            print("\n5. Quantization Bits Ablation...")
            all_results['quantization_bits_ablation'] = self.run_quantization_bits_ablation()
        except Exception as e:
            print(f"Quantization bits ablation failed: {e}")

        # Save comprehensive results
        with open(os.path.join(self.output_dir, "comprehensive_ablation_results.json"), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        # Generate summary
        self.generate_ablation_summary(all_results)

        return all_results

    def generate_ablation_summary(self, all_results: Dict[str, Any]):
        """Generate comprehensive ablation summary"""

        summary_path = os.path.join(self.output_dir, "ablation_summary.md")

        with open(summary_path, 'w') as f:
            f.write("# Comprehensive Ablation Study Summary\n\n")
            f.write(f"**Model:** {self.model_name}\n\n")

            for study_name, results in all_results.items():
                f.write(f"## {study_name.replace('_', ' ').title()}\n\n")

                if isinstance(results, dict) and len(results) > 0:
                    successful_configs = [k for k, v in results.items() if 'error' not in v]
                    f.write(f"- **Configurations Tested:** {len(results)}\n")
                    f.write(f"- **Successful Runs:** {len(successful_configs)}\n")

                    if successful_configs:
                        # Find best performing configuration in this study
                        best_config = None
                        best_score = 0

                        for config_name in successful_configs:
                            result = results[config_name]
                            score = result.get('evaluation_results', {}).get('overall_quality_score', 0)
                            if score > best_score:
                                best_score = score
                                best_config = config_name

                        if best_config:
                            f.write(f"- **Best Configuration:** {best_config}\n")
                            f.write(f"- **Best Score:** {best_score:.4f}\n")

                f.write("\n")

        print(f"Ablation summary saved to: {summary_path}")

def main():
    """Main function to run ablation study"""

    import argparse

    parser = argparse.ArgumentParser(description="Run ablation study for KV cache compression")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./ablation_results")
    parser.add_argument("--study_type", type=str, choices=[
        'component', 'importance_weights', 'precision_thresholds',
        'propagation_ratios', 'quantization_bits', 'comprehensive'
    ], default='comprehensive')

    args = parser.parse_args()

    # Create ablation study
    ablation = AblationStudy(
        model_name=args.model_name,
        device=args.device,
        output_dir=args.output_dir
    )

    # Run specified study
    if args.study_type == 'component':
        results = ablation.run_component_ablation()
    elif args.study_type == 'importance_weights':
        results = ablation.run_importance_weights_ablation()
    elif args.study_type == 'precision_thresholds':
        results = ablation.run_precision_thresholds_ablation()
    elif args.study_type == 'propagation_ratios':
        results = ablation.run_propagation_ratios_ablation()
    elif args.study_type == 'quantization_bits':
        results = ablation.run_quantization_bits_ablation()
    else:  # comprehensive
        results = ablation.run_comprehensive_ablation()

    print(f"\nAblation study completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()