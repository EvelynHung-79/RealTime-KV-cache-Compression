import torch
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import pandas as pd

from ..models.modified_llama import create_compressed_llama_model
from ..configs.base_config import CompressionConfig
from ..evaluation.longbench_eval import LongBenchEvaluator
from ..utils.memory_utils import MemoryMonitor

class HyperparameterTuner:
    """
    Hyperparameter optimization for Real-time Prefill KV Cache Compression
    Supports multiple optimization strategies: Grid Search, Random Search, Bayesian Optimization
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cuda",
        output_dir: str = "./tuning_results"
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.evaluation_history = []

        os.makedirs(output_dir, exist_ok=True)

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def define_search_space(self) -> Dict[str, Tuple]:
        """Define hyperparameter search space"""

        return {
            # Importance scoring weights
            'alpha': (0.1, 0.8),      # Prompt attention weight
            'beta': (0.05, 0.5),      # Position bias weight  
            'gamma': (0.05, 0.5),     # Context relevance weight

            # Precision thresholds
            'theta_h': (0.5, 0.9),    # High precision threshold
            'theta_m': (0.1, 0.5),    # Medium precision threshold

            # Layer propagation ratios
            'early_layer_ratio': (0.6, 1.0),
            'middle_layer_ratio': (0.4, 0.8),
            'later_layer_ratio': (0.2, 0.6),

            # Memory budget constraints
            'memory_budget_ratio': (0.3, 0.8),
            'quality_loss_tolerance': (0.01, 0.1)
        }

    def create_config_from_params(self, params: Dict[str, float]) -> CompressionConfig:
        """Create configuration from hyperparameters"""

        # Normalize importance weights to sum to 1
        alpha = params['alpha']
        beta = params['beta']
        gamma = params['gamma']

        total = alpha + beta + gamma
        alpha = alpha / total
        beta = beta / total
        gamma = gamma / total

        return CompressionConfig(
            model_name=self.model_name,
            alpha=alpha,
            beta=beta, 
            gamma=gamma,
            theta_h=params['theta_h'],
            theta_m=params['theta_m'],
            early_layer_ratio=params['early_layer_ratio'],
            middle_layer_ratio=params['middle_layer_ratio'],
            later_layer_ratio=params['later_layer_ratio'],
            memory_budget_ratio=params['memory_budget_ratio'],
            quality_loss_tolerance=params['quality_loss_tolerance']
        )

    def objective_function(
        self,
        config: CompressionConfig,
        weight_quality: float = 0.6,
        weight_efficiency: float = 0.3,
        weight_speed: float = 0.1
    ) -> float:
        """
        Multi-objective function combining quality, memory efficiency, and speed
        """

        try:
            # Create and evaluate model
            model = create_compressed_llama_model(self.model_name, config, self.device)
            model.eval()

            # Create evaluator with quick evaluation
            eval_dir = os.path.join(self.output_dir, "temp_eval")
            os.makedirs(eval_dir, exist_ok=True)

            evaluator = LongBenchEvaluator(model, self.tokenizer, config, eval_dir)

            # Monitor memory and time
            memory_monitor = MemoryMonitor()
            memory_monitor.start_monitoring()

            import time
            start_time = time.time()

            # Quick evaluation on subset of tasks
            results = evaluator.evaluate_all_tasks(
                tasks=['narrativeqa', 'qasper'],  # Limited tasks for speed
                max_samples_per_task=5           # Limited samples
            )

            evaluation_time = time.time() - start_time
            memory_stats = memory_monitor.get_peak_memory()
            memory_monitor.stop_monitoring()

            # Extract metrics
            quality_score = results.get('overall_quality_score', 0)
            compression_perf = results.get('compression_performance', {})
            memory_savings = compression_perf.get('overall_avg_memory_savings', 0)

            # Speed metric (inverse of evaluation time)
            speed_score = 1.0 / (evaluation_time + 1.0)  # +1 to avoid division by zero

            # Combined objective (higher is better)
            objective_score = (
                weight_quality * quality_score +
                weight_efficiency * memory_savings +
                weight_speed * speed_score
            )

            # Store evaluation history
            eval_record = {
                'config': config.__dict__.copy(),
                'quality_score': quality_score,
                'memory_savings': memory_savings,
                'speed_score': speed_score,
                'objective_score': objective_score,
                'evaluation_time': evaluation_time,
                'memory_stats': memory_stats
            }

            self.evaluation_history.append(eval_record)

            # Clean up
            del model
            torch.cuda.empty_cache()

            return objective_score

        except Exception as e:
            print(f"Error in objective function: {e}")
            return 0.0  # Return worst score on error

    def grid_search(
        self,
        n_points_per_dim: int = 3,
        max_evaluations: Optional[int] = None
    ) -> Dict[str, Any]:
        """Grid search optimization"""

        print("\n" + "="*50)
        print("GRID SEARCH HYPERPARAMETER OPTIMIZATION")
        print("="*50)

        search_space = self.define_search_space()

        # Create grid points for each dimension
        grid_points = {}
        for param_name, (low, high) in search_space.items():
            grid_points[param_name] = np.linspace(low, high, n_points_per_dim)

        # Generate all combinations
        import itertools

        param_names = list(grid_points.keys())
        param_values = list(grid_points.values())

        all_combinations = list(itertools.product(*param_values))

        if max_evaluations and len(all_combinations) > max_evaluations:
            # Randomly sample if too many combinations
            indices = np.random.choice(len(all_combinations), max_evaluations, replace=False)
            all_combinations = [all_combinations[i] for i in indices]

        print(f"Testing {len(all_combinations)} parameter combinations...")

        best_score = -np.inf
        best_params = None
        best_config = None

        for i, param_values in enumerate(all_combinations):
            params = dict(zip(param_names, param_values))

            print(f"\nEvaluation {i+1}/{len(all_combinations)}")
            print(f"Parameters: {params}")

            # Create configuration
            config = self.create_config_from_params(params)

            # Evaluate
            score = self.objective_function(config)

            print(f"Objective Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_config = config

                print(f"*** New best score: {best_score:.4f} ***")

        results = {
            'method': 'grid_search',
            'best_score': best_score,
            'best_params': best_params,
            'best_config': best_config.__dict__.copy() if best_config else None,
            'n_evaluations': len(all_combinations),
            'evaluation_history': self.evaluation_history.copy()
        }

        # Save results
        with open(os.path.join(self.output_dir, "grid_search_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def random_search(
        self,
        n_trials: int = 50,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """Random search optimization"""

        print("\n" + "="*50)
        print("RANDOM SEARCH HYPERPARAMETER OPTIMIZATION")
        print("="*50)

        np.random.seed(random_seed)
        search_space = self.define_search_space()

        best_score = -np.inf
        best_params = None
        best_config = None

        for i in range(n_trials):
            # Sample random parameters
            params = {}
            for param_name, (low, high) in search_space.items():
                params[param_name] = np.random.uniform(low, high)

            print(f"\nTrial {i+1}/{n_trials}")
            print(f"Parameters: {params}")

            # Create configuration
            config = self.create_config_from_params(params)

            # Evaluate
            score = self.objective_function(config)

            print(f"Objective Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_config = config

                print(f"*** New best score: {best_score:.4f} ***")

        results = {
            'method': 'random_search',
            'best_score': best_score,
            'best_params': best_params,
            'best_config': best_config.__dict__.copy() if best_config else None,
            'n_trials': n_trials,
            'evaluation_history': self.evaluation_history.copy()
        }

        # Save results
        with open(os.path.join(self.output_dir, "random_search_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def bayesian_optimization(
        self,
        n_trials: int = 30,
        n_initial_points: int = 5
    ) -> Dict[str, Any]:
        """Bayesian optimization using Optuna"""

        print("\n" + "="*50)
        print("BAYESIAN OPTIMIZATION HYPERPARAMETER TUNING")
        print("="*50)

        search_space = self.define_search_space()

        def optuna_objective(trial):
            # Sample parameters
            params = {}
            for param_name, (low, high) in search_space.items():
                params[param_name] = trial.suggest_float(param_name, low, high)

            print(f"\nTrial {trial.number + 1}")
            print(f"Parameters: {params}")

            # Create configuration and evaluate
            config = self.create_config_from_params(params)
            score = self.objective_function(config)

            print(f"Objective Score: {score:.4f}")

            return score

        # Create Optuna study
        study = optuna.create_study(direction='maximize')

        try:
            study.optimize(optuna_objective, n_trials=n_trials)

            best_params = study.best_params
            best_score = study.best_value
            best_config = self.create_config_from_params(best_params)

            print(f"\n*** Optimization completed ***")
            print(f"Best score: {best_score:.4f}")
            print(f"Best parameters: {best_params}")

            results = {
                'method': 'bayesian_optimization',
                'best_score': best_score,
                'best_params': best_params,
                'best_config': best_config.__dict__.copy(),
                'n_trials': n_trials,
                'optuna_study': {
                    'best_trial': study.best_trial.number,
                    'best_value': study.best_value,
                    'best_params': study.best_params
                },
                'evaluation_history': self.evaluation_history.copy()
            }

        except Exception as e:
            print(f"Bayesian optimization failed: {e}")

            results = {
                'method': 'bayesian_optimization',
                'error': str(e),
                'evaluation_history': self.evaluation_history.copy()
            }

        # Save results
        with open(os.path.join(self.output_dir, "bayesian_optimization_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def evolutionary_search(
        self,
        population_size: int = 20,
        n_generations: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ) -> Dict[str, Any]:
        """Evolutionary/Genetic Algorithm optimization"""

        print("\n" + "="*50)
        print("EVOLUTIONARY HYPERPARAMETER OPTIMIZATION")
        print("="*50)

        search_space = self.define_search_space()
        param_names = list(search_space.keys())
        param_bounds = list(search_space.values())

        # Initialize population
        population = []
        for _ in range(population_size):
            individual = []
            for low, high in param_bounds:
                individual.append(np.random.uniform(low, high))
            population.append(individual)

        best_score = -np.inf
        best_individual = None
        best_config = None

        for generation in range(n_generations):
            print(f"\nGeneration {generation + 1}/{n_generations}")

            # Evaluate population
            fitness_scores = []
            for i, individual in enumerate(population):
                params = dict(zip(param_names, individual))
                config = self.create_config_from_params(params)
                score = self.objective_function(config)
                fitness_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_individual = individual.copy()
                    best_config = config
                    print(f"  New best score: {best_score:.4f}")

            print(f"  Generation best: {max(fitness_scores):.4f}")
            print(f"  Generation avg: {np.mean(fitness_scores):.4f}")

            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
                tournament_scores = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_scores)]
                new_population.append(population[winner_idx].copy())

            # Crossover and mutation
            for i in range(0, population_size - 1, 2):
                if np.random.random() < crossover_rate:
                    # Single-point crossover
                    crossover_point = np.random.randint(1, len(param_names))
                    child1 = new_population[i][:crossover_point] + new_population[i+1][crossover_point:]
                    child2 = new_population[i+1][:crossover_point] + new_population[i][crossover_point:]
                    new_population[i] = child1
                    new_population[i+1] = child2

                # Mutation
                for individual in [new_population[i], new_population[i+1]]:
                    for j in range(len(individual)):
                        if np.random.random() < mutation_rate:
                            low, high = param_bounds[j]
                            individual[j] = np.random.uniform(low, high)

            population = new_population

        best_params = dict(zip(param_names, best_individual)) if best_individual else None

        results = {
            'method': 'evolutionary_search',
            'best_score': best_score,
            'best_params': best_params,
            'best_config': best_config.__dict__.copy() if best_config else None,
            'population_size': population_size,
            'n_generations': n_generations,
            'evaluation_history': self.evaluation_history.copy()
        }

        # Save results
        with open(os.path.join(self.output_dir, "evolutionary_search_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def compare_methods(
        self,
        methods: List[str] = ['random_search', 'bayesian_optimization'],
        n_trials_per_method: int = 20
    ) -> Dict[str, Any]:
        """Compare different optimization methods"""

        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION METHOD COMPARISON")
        print("="*60)

        comparison_results = {}

        for method in methods:
            print(f"\nRunning {method}...")
            self.evaluation_history = []  # Reset history for each method

            try:
                if method == 'grid_search':
                    results = self.grid_search(n_points_per_dim=3, max_evaluations=n_trials_per_method)
                elif method == 'random_search':
                    results = self.random_search(n_trials=n_trials_per_method)
                elif method == 'bayesian_optimization':
                    results = self.bayesian_optimization(n_trials=n_trials_per_method)
                elif method == 'evolutionary_search':
                    results = self.evolutionary_search(
                        population_size=min(10, n_trials_per_method // 2),
                        n_generations=max(2, n_trials_per_method // 10)
                    )
                else:
                    print(f"Unknown method: {method}")
                    continue

                comparison_results[method] = results

            except Exception as e:
                print(f"Method {method} failed: {e}")
                comparison_results[method] = {'error': str(e)}

        # Analyze comparison results
        self.analyze_method_comparison(comparison_results)

        # Save comparison results
        with open(os.path.join(self.output_dir, "method_comparison_results.json"), 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)

        return comparison_results

    def analyze_method_comparison(self, comparison_results: Dict[str, Any]):
        """Analyze and summarize method comparison results"""

        print("\n" + "="*50)
        print("METHOD COMPARISON ANALYSIS")
        print("="*50)

        summary_data = []

        for method, results in comparison_results.items():
            if 'error' in results:
                print(f"{method}: FAILED - {results['error']}")
                continue

            best_score = results.get('best_score', 0)
            n_evaluations = results.get('n_trials', results.get('n_evaluations', 0))

            print(f"\n{method.upper()}:")
            print(f"  Best Score: {best_score:.4f}")
            print(f"  Evaluations: {n_evaluations}")
            print(f"  Efficiency: {best_score/n_evaluations:.6f} (score per evaluation)")

            if 'best_params' in results and results['best_params']:
                print(f"  Best Parameters:")
                for param, value in results['best_params'].items():
                    print(f"    {param}: {value:.4f}")

            summary_data.append({
                'method': method,
                'best_score': best_score,
                'n_evaluations': n_evaluations,
                'efficiency': best_score / n_evaluations if n_evaluations > 0 else 0
            })

        # Find best method
        if summary_data:
            best_method = max(summary_data, key=lambda x: x['best_score'])
            most_efficient = max(summary_data, key=lambda x: x['efficiency'])

            print(f"\n*** SUMMARY ***")
            print(f"Best Score: {best_method['method']} ({best_method['best_score']:.4f})")
            print(f"Most Efficient: {most_efficient['method']} ({most_efficient['efficiency']:.6f})")

        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.output_dir, "method_comparison_summary.csv"), index=False)

def main():
    """Main function for hyperparameter tuning"""

    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter tuning for KV cache compression")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./tuning_results")
    parser.add_argument("--method", type=str, choices=[
        'grid_search', 'random_search', 'bayesian_optimization',
        'evolutionary_search', 'compare_all'
    ], default='random_search')
    parser.add_argument("--n_trials", type=int, default=20)

    args = parser.parse_args()

    # Create tuner
    tuner = HyperparameterTuner(
        model_name=args.model_name,
        device=args.device,
        output_dir=args.output_dir
    )

    # Run optimization
    if args.method == 'grid_search':
        results = tuner.grid_search(n_points_per_dim=3, max_evaluations=args.n_trials)
    elif args.method == 'random_search':
        results = tuner.random_search(n_trials=args.n_trials)
    elif args.method == 'bayesian_optimization':
        results = tuner.bayesian_optimization(n_trials=args.n_trials)
    elif args.method == 'evolutionary_search':
        results = tuner.evolutionary_search(
            population_size=min(20, args.n_trials),
            n_generations=max(5, args.n_trials // 4)
        )
    elif args.method == 'compare_all':
        results = tuner.compare_methods(
            methods=['random_search', 'bayesian_optimization'],
            n_trials_per_method=args.n_trials
        )

    print(f"\nHyperparameter tuning completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()