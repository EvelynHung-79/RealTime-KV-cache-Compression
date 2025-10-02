import torch
import time
import psutil
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.modified_llama import create_compressed_llama_model
from ..configs.base_config import CompressionConfig
from ..utils.memory_utils import MemoryMonitor, get_model_memory_footprint
from ..utils.eval_utils import PerformanceTimer

class CompressionBenchmark:
    """
    Comprehensive benchmark suite for compression performance evaluation
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        output_dir: str = "./benchmark_results"
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

    def create_test_configs(self) -> List[Tuple[str, CompressionConfig]]:
        """Create different configuration scenarios for testing"""

        base_config = CompressionConfig(model_name=self.model_name)

        configs = [
            ("baseline", CompressionConfig(
                model_name=self.model_name,
                alpha=0.0, beta=0.0, gamma=0.0,  # No compression
                theta_h=1.0, theta_m=1.0
            )),
            ("conservative", CompressionConfig(
                model_name=self.model_name,
                alpha=0.3, beta=0.2, gamma=0.5,
                theta_h=0.8, theta_m=0.5,
                early_layer_ratio=0.9, middle_layer_ratio=0.8, later_layer_ratio=0.6
            )),
            ("balanced", CompressionConfig(
                model_name=self.model_name,
                alpha=0.4, beta=0.3, gamma=0.3,
                theta_h=0.7, theta_m=0.3,
                early_layer_ratio=0.8, middle_layer_ratio=0.6, later_layer_ratio=0.4
            )),
            ("aggressive", CompressionConfig(
                model_name=self.model_name,
                alpha=0.5, beta=0.3, gamma=0.2,
                theta_h=0.6, theta_m=0.2,
                early_layer_ratio=0.7, middle_layer_ratio=0.5, later_layer_ratio=0.3
            )),
            ("prompt_focused", CompressionConfig(
                model_name=self.model_name,
                alpha=0.7, beta=0.2, gamma=0.1,
                theta_h=0.7, theta_m=0.3
            )),
            ("position_focused", CompressionConfig(
                model_name=self.model_name,
                alpha=0.2, beta=0.6, gamma=0.2,
                theta_h=0.7, theta_m=0.3
            ))
        ]

        return configs

    def generate_test_sequences(self, lengths: List[int], count: int = 5) -> Dict[int, List[str]]:
        """Generate test sequences of different lengths"""

        test_sequences = {}

        for length in lengths:
            sequences = []
            for i in range(count):
                # Generate random text of approximately the target length
                words = ["test", "sequence", "compression", "evaluation", "performance", 
                        "benchmark", "memory", "efficiency", "optimization", "transformer",
                        "attention", "layer", "context", "token", "processing"]

                text = " ".join(np.random.choice(words, size=length // 5))

                # Ensure it meets the length requirement
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                while len(tokens) < length:
                    text += " " + " ".join(np.random.choice(words, size=10))
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)

                # Truncate if too long
                if len(tokens) > length:
                    tokens = tokens[:length]
                    text = self.tokenizer.decode(tokens, skip_special_tokens=True)

                sequences.append(text)

            test_sequences[length] = sequences

        return test_sequences

    def benchmark_single_config(
        self,
        config_name: str,
        config: CompressionConfig,
        test_sequences: Dict[int, List[str]]
    ) -> Dict[str, Any]:
        """Benchmark a single configuration"""

        print(f"\nBenchmarking configuration: {config_name}")

        # Create model
        model = create_compressed_llama_model(self.model_name, config, self.device)
        model.eval()

        # Get model memory footprint
        model_memory = get_model_memory_footprint(model)

        config_results = {
            'config_name': config_name,
            'config': config.__dict__.copy(),
            'model_memory': model_memory,
            'sequence_results': {}
        }

        # Test each sequence length
        for seq_len, sequences in test_sequences.items():
            print(f"  Testing sequence length: {seq_len}")

            seq_results = {
                'sequence_length': seq_len,
                'num_sequences': len(sequences),
                'measurements': []
            }

            for i, text in enumerate(tqdm(sequences, desc=f"Seq {seq_len}")):
                try:
                    result = self.measure_single_inference(model, text, config)
                    seq_results['measurements'].append(result)
                except Exception as e:
                    print(f"    Error processing sequence {i}: {e}")
                    continue

            # Aggregate results for this sequence length
            if seq_results['measurements']:
                seq_results['aggregated'] = self.aggregate_measurements(seq_results['measurements'])

            config_results['sequence_results'][seq_len] = seq_results

        # Clean up model
        del model
        torch.cuda.empty_cache()

        return config_results

    def measure_single_inference(
        self,
        model,
        text: str,
        config: CompressionConfig
    ) -> Dict[str, Any]:
        """Measure performance for a single inference"""

        # Initialize memory monitor
        memory_monitor = MemoryMonitor(interval=0.1)
        timer = PerformanceTimer()

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_position_embeddings
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        input_length = inputs['input_ids'].shape[1]

        # Start monitoring
        memory_monitor.start_monitoring()

        # Reset compression state
        if hasattr(model, 'reset_compression_state'):
            model.reset_compression_state()

        # Measure TTFT (Time to First Token)
        timer.start('prefill')

        with torch.no_grad():
            # First forward pass (prefill)
            outputs = model(**inputs, use_cache=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        ttft = timer.stop('prefill')

        # Measure generation
        timer.start('generation')

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        generation_time = timer.stop('generation')

        # Stop monitoring
        memory_stats = memory_monitor.get_peak_memory()
        memory_monitor.stop_monitoring()

        # Get compression statistics
        compression_stats = {}
        if hasattr(model, 'get_compression_stats'):
            compression_stats = model.get_compression_stats()

        # Calculate metrics
        output_length = generated.shape[1] - input_length
        total_time = ttft + generation_time

        return {
            'input_length': input_length,
            'output_length': output_length,
            'ttft': ttft,
            'generation_time': generation_time,
            'total_time': total_time,
            'tokens_per_second': output_length / generation_time if generation_time > 0 else 0,
            'memory_stats': memory_stats,
            'compression_stats': compression_stats
        }

    def aggregate_measurements(self, measurements: List[Dict]) -> Dict[str, Any]:
        """Aggregate measurements across multiple runs"""

        if not measurements:
            return {}

        # Numeric fields to aggregate
        numeric_fields = [
            'input_length', 'output_length', 'ttft', 'generation_time', 
            'total_time', 'tokens_per_second'
        ]

        aggregated = {}

        for field in numeric_fields:
            values = [m.get(field, 0) for m in measurements]
            aggregated[f'avg_{field}'] = np.mean(values)
            aggregated[f'std_{field}'] = np.std(values)
            aggregated[f'min_{field}'] = np.min(values)
            aggregated[f'max_{field}'] = np.max(values)
            aggregated[f'median_{field}'] = np.median(values)

        # Aggregate memory stats
        memory_stats = [m.get('memory_stats', {}) for m in measurements if m.get('memory_stats')]
        if memory_stats:
            memory_fields = ['peak_cpu_memory_mb', 'peak_gpu_allocated_mb', 'peak_gpu_reserved_mb']
            for field in memory_fields:
                values = [ms.get(field, 0) for ms in memory_stats]
                if values:
                    aggregated[f'avg_{field}'] = np.mean(values)
                    aggregated[f'max_{field}'] = np.max(values)

        # Aggregate compression stats
        compression_stats = [m.get('compression_stats', {}) for m in measurements if m.get('compression_stats')]
        if compression_stats:
            comp_fields = ['avg_compression_ratio', 'avg_memory_savings', 'total_processing_time']
            for field in comp_fields:
                values = [cs.get(field, 0) for cs in compression_stats]
                if values:
                    aggregated[f'avg_{field}'] = np.mean(values)
                    aggregated[f'std_{field}'] = np.std(values)

        return aggregated

    def run_full_benchmark(
        self,
        sequence_lengths: List[int] = [1024, 2048, 4096, 8192],
        sequences_per_length: int = 5
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark across all configurations"""

        print("Starting comprehensive compression benchmark...")
        print(f"Model: {self.model_name}")
        print(f"Sequence lengths: {sequence_lengths}")
        print(f"Sequences per length: {sequences_per_length}")

        # Generate test sequences
        print("\nGenerating test sequences...")
        test_sequences = self.generate_test_sequences(sequence_lengths, sequences_per_length)

        # Get test configurations
        configs = self.create_test_configs()

        benchmark_results = {
            'model_name': self.model_name,
            'test_setup': {
                'sequence_lengths': sequence_lengths,
                'sequences_per_length': sequences_per_length,
                'device': self.device
            },
            'config_results': {}
        }

        # Run benchmarks for each configuration
        for config_name, config in configs:
            try:
                config_result = self.benchmark_single_config(config_name, config, test_sequences)
                benchmark_results['config_results'][config_name] = config_result

                # Save intermediate results
                self.save_results(benchmark_results, f"intermediate_{config_name}_results.json")

            except Exception as e:
                print(f"Error benchmarking {config_name}: {e}")
                benchmark_results['config_results'][config_name] = {'error': str(e)}

        # Save final results
        self.save_results(benchmark_results, "comprehensive_benchmark_results.json")

        # Generate analysis
        self.analyze_results(benchmark_results)

        return benchmark_results

    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {filepath}")

    def analyze_results(self, results: Dict):
        """Generate analysis and visualizations"""

        print("\nGenerating analysis...")

        # Extract data for comparison
        config_names = list(results['config_results'].keys())
        sequence_lengths = results['test_setup']['sequence_lengths']

        # Create comparison plots
        self.plot_performance_comparison(results, config_names, sequence_lengths)
        self.plot_memory_usage_comparison(results, config_names, sequence_lengths)
        self.plot_compression_efficiency(results, config_names, sequence_lengths)

        # Generate summary report
        self.generate_summary_report(results)

    def plot_performance_comparison(self, results: Dict, config_names: List[str], seq_lengths: List[int]):
        """Plot performance comparison across configurations"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # TTFT comparison
        for config_name in config_names:
            if 'error' in results['config_results'].get(config_name, {}):
                continue

            ttft_values = []
            for seq_len in seq_lengths:
                seq_results = results['config_results'][config_name]['sequence_results'].get(seq_len, {})
                ttft = seq_results.get('aggregated', {}).get('avg_ttft', 0)
                ttft_values.append(ttft)

            ax1.plot(seq_lengths, ttft_values, marker='o', label=config_name)

        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('TTFT (seconds)')
        ax1.set_title('Time to First Token')
        ax1.legend()
        ax1.grid(True)

        # Tokens per second comparison
        for config_name in config_names:
            if 'error' in results['config_results'].get(config_name, {}):
                continue

            tps_values = []
            for seq_len in seq_lengths:
                seq_results = results['config_results'][config_name]['sequence_results'].get(seq_len, {})
                tps = seq_results.get('aggregated', {}).get('avg_tokens_per_second', 0)
                tps_values.append(tps)

            ax2.plot(seq_lengths, tps_values, marker='o', label=config_name)

        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Tokens/Second')
        ax2.set_title('Generation Throughput')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_memory_usage_comparison(self, results: Dict, config_names: List[str], seq_lengths: List[int]):
        """Plot memory usage comparison"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Peak GPU memory
        for config_name in config_names:
            if 'error' in results['config_results'].get(config_name, {}):
                continue

            memory_values = []
            for seq_len in seq_lengths:
                seq_results = results['config_results'][config_name]['sequence_results'].get(seq_len, {})
                memory = seq_results.get('aggregated', {}).get('max_peak_gpu_allocated_mb', 0)
                memory_values.append(memory)

            ax1.plot(seq_lengths, memory_values, marker='o', label=config_name)

        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Peak GPU Memory (MB)')
        ax1.set_title('Peak GPU Memory Usage')
        ax1.legend()
        ax1.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'memory_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_compression_efficiency(self, results: Dict, config_names: List[str], seq_lengths: List[int]):
        """Plot compression efficiency metrics"""

        # Extract compression ratios and memory savings
        compression_data = []

        for config_name in config_names:
            if 'error' in results['config_results'].get(config_name, {}):
                continue

            for seq_len in seq_lengths:
                seq_results = results['config_results'][config_name]['sequence_results'].get(seq_len, {})
                aggregated = seq_results.get('aggregated', {})

                compression_ratio = aggregated.get('avg_avg_compression_ratio', 1.0)
                memory_savings = aggregated.get('avg_avg_memory_savings', 0.0)

                compression_data.append({
                    'config': config_name,
                    'seq_len': seq_len,
                    'compression_ratio': compression_ratio,
                    'memory_savings': memory_savings * 100  # Convert to percentage
                })

        if compression_data:
            # Create heatmaps
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Prepare data for heatmaps
            configs = sorted(set(d['config'] for d in compression_data))
            seq_lens = sorted(set(d['seq_len'] for d in compression_data))

            # Compression ratio heatmap
            ratio_matrix = np.zeros((len(configs), len(seq_lens)))
            for i, config in enumerate(configs):
                for j, seq_len in enumerate(seq_lens):
                    ratio = next((d['compression_ratio'] for d in compression_data 
                                if d['config'] == config and d['seq_len'] == seq_len), 1.0)
                    ratio_matrix[i, j] = ratio

            sns.heatmap(ratio_matrix, xticklabels=seq_lens, yticklabels=configs, 
                       annot=True, fmt='.3f', ax=ax1, cmap='RdYlBu_r')
            ax1.set_title('Compression Ratio')
            ax1.set_xlabel('Sequence Length')

            # Memory savings heatmap
            savings_matrix = np.zeros((len(configs), len(seq_lens)))
            for i, config in enumerate(configs):
                for j, seq_len in enumerate(seq_lens):
                    savings = next((d['memory_savings'] for d in compression_data 
                                  if d['config'] == config and d['seq_len'] == seq_len), 0.0)
                    savings_matrix[i, j] = savings

            sns.heatmap(savings_matrix, xticklabels=seq_lens, yticklabels=configs,
                       annot=True, fmt='.1f', ax=ax2, cmap='RdYlGn')
            ax2.set_title('Memory Savings (%)')
            ax2.set_xlabel('Sequence Length')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'compression_efficiency.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def generate_summary_report(self, results: Dict):
        """Generate text summary report"""

        report_path = os.path.join(self.output_dir, 'benchmark_summary.md')

        with open(report_path, 'w') as f:
            f.write(f"# Compression Benchmark Summary\n\n")
            f.write(f"**Model:** {results['model_name']}\n")
            f.write(f"**Test Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Configuration Performance\n\n")

            for config_name, config_results in results['config_results'].items():
                if 'error' in config_results:
                    f.write(f"### {config_name}\n")
                    f.write(f"**Status:** Error - {config_results['error']}\n\n")
                    continue

                f.write(f"### {config_name}\n")

                # Find best performing sequence length
                best_seq_len = None
                best_tps = 0

                for seq_len, seq_results in config_results['sequence_results'].items():
                    aggregated = seq_results.get('aggregated', {})
                    tps = aggregated.get('avg_tokens_per_second', 0)
                    if tps > best_tps:
                        best_tps = tps
                        best_seq_len = seq_len

                if best_seq_len:
                    best_results = config_results['sequence_results'][best_seq_len]['aggregated']

                    f.write(f"- **Best Performance (Seq Len {best_seq_len}):**\n")
                    f.write(f"  - TTFT: {best_results.get('avg_ttft', 0):.3f}s\n")
                    f.write(f"  - Throughput: {best_results.get('avg_tokens_per_second', 0):.1f} tokens/s\n")
                    f.write(f"  - Memory Savings: {best_results.get('avg_avg_memory_savings', 0)*100:.1f}%\n")
                    f.write(f"  - Compression Ratio: {best_results.get('avg_avg_compression_ratio', 1.0):.3f}\n")

                f.write("\n")

        print(f"Summary report saved to: {report_path}")

def run_benchmark_suite(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    sequence_lengths: List[int] = [1024, 2048, 4096],
    sequences_per_length: int = 3,
    output_dir: str = "./benchmark_results"
):
    """Run the complete benchmark suite"""

    benchmark = CompressionBenchmark(model_name, output_dir=output_dir)
    results = benchmark.run_full_benchmark(sequence_lengths, sequences_per_length)

    return results

if __name__ == "__main__":
    # Example usage
    results = run_benchmark_suite(
        model_name="meta-llama/Llama-2-7b-hf",
        sequence_lengths=[1024, 2048, 4096],
        sequences_per_length=2,  # Small for testing
        output_dir="./benchmark_test"
    )