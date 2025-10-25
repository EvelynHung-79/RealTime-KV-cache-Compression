import torch
import json
import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import optuna
# from sklearn.gaussian_process import GaussianProcessRegressor # Bayesian Optimization改用Optuna內建的
# from sklearn.gaussian_process.kernels import Matern # Bayesian Optimization改用Optuna內建的
import pandas as pd
from transformers import AutoTokenizer

# --- Add root directory to sys.path ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
# --- Imports from your project ---
# 確保引用路徑正確
from src.models.modified_llama import create_compressed_llama_model
from configs.base_config import CompressionConfig
from src.evaluation.longbench_eval import LongBenchEvaluator
from src.utils.memory_utils import MemoryMonitor

class HyperparameterTuner:
    """
    Hyperparameter optimization for Streaming KVQuant Quantization.
    Supports multiple optimization strategies: Grid Search, Random Search, Bayesian Optimization, Evolutionary Search.
    Updated for the new architecture (no selective propagation, streaming stats).
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
        self.evaluation_history = [] # Tracks results for each trial

        os.makedirs(output_dir, exist_ok=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"HyperparameterTuner initialized for model: {model_name} on device: {device}")
        print(f"Results will be saved to: {output_dir}")

    def define_search_space(self) -> Dict[str, Any]:
        """Define hyperparameter search space for the new architecture"""

        # Using Optuna suggestion types directly for Bayesian Optimization compatibility
        return {
            # Streaming and Chunking
            'chunk_size': ('suggest_categorical', [128, 256, 512]), # Categorical choice

            # Streaming Statistics (EMA Absmax)
            'ema_decay': ('suggest_float', [0.95, 0.999]), # Float range

            # Outlier Detection (Optional tuning - can be fixed)
            # 'outlier_threshold_abs': ('suggest_float', [4.0, 8.0]),
            # 'outlier_threshold_relative': ('suggest_float', [3.0, 7.0]),

            # Attention Sink Handling
            'attention_sink_size': ('suggest_int', [4, 16]), # Integer range

            # Quantization Settings (Key) - Focus on normal bits, keep sink/outlier high
            'key_bits_normal': ('suggest_int', [4, 8]), # Integer range (e.g., 4 or 8)
            # 'key_bits_sink_outlier': ('suggest_int', [8, 8]), # Usually fixed at 8

            # Quantization Settings (Value) - Focus on normal bits
            'value_bits_normal': ('suggest_int', [2, 6]), # Integer range (e.g., 2, 4, maybe 6)
            # 'value_bits_sink_outlier': ('suggest_int', [8, 8]), # Usually fixed at 8

            # Value Quantization Grouping (Example: -1 for per-channel, 64 for group size)
            # This might be better fixed or explored separately due to its structural impact
            # 'value_quant_groups': ('suggest_categorical', [-1, 64]), # Per-channel or groups of 64
        }

    def create_config_from_params(self, params: Dict[str, Any]) -> CompressionConfig:
        """Create CompressionConfig from sampled hyperparameters"""

        # Create a base config and override with tuned parameters
        # Need to load actual model config to get num_layers etc. if base_config doesn't have them
        # For simplicity, assume base_config provides reasonable defaults
        config = CompressionConfig(model_name=self.model_name)

        config.chunk_size = params['chunk_size']
        config.ema_decay = params['ema_decay']
        # config.outlier_threshold_abs = params.get('outlier_threshold_abs', 6.0) # Use default if not tuned
        # config.outlier_threshold_relative = params.get('outlier_threshold_relative', 5.0) # Use default if not tuned
        config.attention_sink_size = params['attention_sink_size']
        config.key_bits_normal = params['key_bits_normal']
        # config.key_bits_sink_outlier = params.get('key_bits_sink_outlier', 8) # Use default
        config.value_bits_normal = params['value_bits_normal']
        # config.value_bits_sink_outlier = params.get('value_bits_sink_outlier', 8) # Use default
        # config.value_quant_groups = params.get('value_quant_groups', -1) # Use default

        # --- IMPORTANT ---
        # Ensure parameters not in the search space use sensible defaults from CompressionConfig
        # For example, outlier thresholds, sink/outlier bits, value_quant_groups might be fixed
        config.outlier_threshold_abs = 6.0 # Example fixed value
        config.outlier_threshold_relative = 5.0 # Example fixed value
        config.key_bits_sink_outlier = 8 # Example fixed value
        config.value_bits_sink_outlier = 8 # Example fixed value
        config.value_quant_groups = -1 # Example fixed value (per-channel)

        # Remove obsolete parameters explicitly if they exist in the base class definition
        obsolete_params = ['alpha', 'beta', 'gamma', 'theta_h', 'theta_m',
                           'early_layer_ratio', 'middle_layer_ratio', 'later_layer_ratio',
                           'high_precision_bits', 'medium_precision_bits', 'low_precision_bits',
                           'memory_budget_ratio', 'quality_loss_tolerance']
        for p in obsolete_params:
            if hasattr(config, p):
                delattr(config, p) # Or set to None/default if required by downstream code

        return config

    def objective_function(
        self,
        config: CompressionConfig,
        weight_quality: float = 0.7,      # Increased weight on quality
        weight_efficiency: float = 0.2,   # Weight on estimated memory savings
        weight_speed: float = 0.1         # Weight on generation speed (throughput)
    ) -> float:
        """
        Multi-objective function combining quality, estimated memory efficiency, and speed.
        Higher is better.
        """
        trial_results = { # Store detailed metrics for this trial
            'config': config.__dict__.copy(),
            'quality_score': 0,
            'estimated_memory_savings': 0,
            'speed_score': 0,
            'objective_score': 0.0, # Default to worst score
            'evaluation_time': 0,
            'memory_stats': {},
            'error': None
        }

        try:
            # Create and evaluate model using the new config
            print(f"  Creating model with config: {config.__dict__}")
            model = create_compressed_llama_model(self.model_name, config, self.device)
            model.eval()

            # Create evaluator with quick evaluation settings
            eval_dir = os.path.join(self.output_dir, f"temp_eval_{np.random.randint(10000)}")
            os.makedirs(eval_dir, exist_ok=True)
            evaluator = LongBenchEvaluator(model, self.tokenizer, config, eval_dir)

            # Monitor memory and time
            memory_monitor = MemoryMonitor()
            memory_monitor.start_monitoring()
            timer_start = time.time()

            # Quick evaluation on a subset of tasks/samples
            print("  Starting quick evaluation...")
            eval_results = evaluator.evaluate_all_tasks(
                tasks=['narrativeqa', 'qasper'],  # Limited tasks for speed
                max_samples_per_task=5           # Limited samples
            )
            print("  Evaluation finished.")

            evaluation_time = time.time() - timer_start
            memory_stats = memory_monitor.get_peak_memory()
            memory_monitor.stop_monitoring()

            # Extract metrics
            quality_score = eval_results.get('overall_quality_score', 0)

            # --- Get Estimated Memory Savings ---
            compression_stats = model.get_compression_stats() # Get stats from the model/compressor
            estimated_memory_savings = compression_stats.get('estimated_memory_savings', 0.0)

            # --- Speed Metric ---
            # Use average tokens per second from evaluation results if available
            # This requires LongBenchEvaluator to aggregate and return it
            # Placeholder: Use inverse of evaluation time
            avg_tps = 0
            if 'task_results' in eval_results:
                 all_tps = []
                 for task_res in eval_results['task_results'].values():
                      if 'compression_metrics' in task_res:
                           all_tps.append(task_res['compression_metrics'].get('avg_tokens_per_second', 0))
                 if all_tps:
                      avg_tps = np.mean(all_tps)

            speed_score = avg_tps / 100.0 # Normalize TPS (e.g., assume 100 TPS is good)
            # Fallback if TPS not available
            if speed_score == 0:
                 speed_score = 1.0 / (evaluation_time + 1.0) # Add 1 to avoid division by zero

            # --- Combined Objective ---
            objective_score = (
                weight_quality * quality_score +
                weight_efficiency * estimated_memory_savings +
                weight_speed * speed_score
            )

            # Store results for history
            trial_results.update({
                'quality_score': quality_score,
                'estimated_memory_savings': estimated_memory_savings,
                'speed_score': speed_score,
                'objective_score': objective_score,
                'evaluation_time': evaluation_time,
                'memory_stats': memory_stats,
                'eval_results_summary': eval_results # Store summary
            })
            print(f"  Trial Metrics: Quality={quality_score:.4f}, Est.Savings={estimated_memory_savings:.4f}, Speed={speed_score:.4f} -> Objective={objective_score:.4f}")


        except Exception as e:
            print(f"Error in objective function: {e}")
            import traceback
            traceback.print_exc()
            trial_results['error'] = str(e)
            objective_score = 0.0 # Return worst score on error

        finally:
            # Clean up GPU memory
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
            # Append results regardless of success/failure
            self.evaluation_history.append(trial_results)

        return objective_score

    # --- Search Methods (Largely unchanged, but use new search space/config) ---

    def grid_search(
        self,
        # Grid search is less practical with the new, larger parameter space.
        # Consider removing or simplifying it significantly.
        # Example: Tune only bits, fixing others.
        n_points_per_dim: int = 2,
        max_evaluations: Optional[int] = 16 # Keep low for grid search
    ) -> Dict[str, Any]:
        """Grid search optimization (Simplified for new space)"""
        print("\n" + "="*50)
        print("GRID SEARCH HYPERPARAMETER OPTIMIZATION (Limited Scope)")
        print("="*50)

        # --- Select a SUBSET of parameters for grid search ---
        search_space_subset = {
            'key_bits_normal': [4, 8],
            'value_bits_normal': [4, 6],
            'chunk_size': [256, 512],
            'attention_sink_size': [8], # Fixed
            'ema_decay': [0.99] # Fixed
        }
        print(f"Grid searching over parameters: {list(search_space_subset.keys())}")

        # Create grid points
        grid_points = search_space_subset

        # Generate combinations
        import itertools
        param_names = list(grid_points.keys())
        param_values_list = list(grid_points.values())
        all_combinations_tuples = list(itertools.product(*param_values_list))
        all_combinations = [dict(zip(param_names, values)) for values in all_combinations_tuples]


        if max_evaluations and len(all_combinations) > max_evaluations:
            indices = np.random.choice(len(all_combinations), max_evaluations, replace=False)
            all_combinations = [all_combinations[i] for i in indices]

        print(f"Testing {len(all_combinations)} parameter combinations...")

        best_score = -np.inf
        best_params = None
        best_config_dict = None

        for i, params_subset in enumerate(all_combinations):
            print(f"\nEvaluation {i+1}/{len(all_combinations)}")
            print(f"Parameters (Subset): {params_subset}")

            # --- Create full config using defaults for non-tuned params ---
            full_params = { # Start with defaults or fixed values
                'chunk_size': params_subset.get('chunk_size', 256),
                'ema_decay': params_subset.get('ema_decay', 0.99),
                'attention_sink_size': params_subset.get('attention_sink_size', 8),
                'key_bits_normal': params_subset.get('key_bits_normal', 4),
                'value_bits_normal': params_subset.get('value_bits_normal', 4),
                # Add other fixed params as needed by create_config_from_params
            }
            config = self.create_config_from_params(full_params)

            score = self.objective_function(config)
            print(f"Objective Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_params = full_params.copy() # Store the full params used
                best_config_dict = config.__dict__.copy()
                print(f"*** New best score: {best_score:.4f} ***")

        results = {
            'method': 'grid_search',
            'best_score': best_score,
            'best_params': best_params,
            'best_config': best_config_dict,
            'n_evaluations': len(all_combinations),
            'evaluation_history': self.evaluation_history.copy()
        }

        with open(os.path.join(self.output_dir, "grid_search_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str) # Use default=str for non-serializable

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
        best_config_dict = None

        for i in range(n_trials):
            params = {}
            # Sample parameters based on defined types
            for param_name, definition in search_space.items():
                suggest_type = definition[0]
                suggest_args = definition[1]
                if suggest_type == 'suggest_float':
                    params[param_name] = np.random.uniform(suggest_args[0], suggest_args[1])
                elif suggest_type == 'suggest_int':
                    params[param_name] = np.random.randint(suggest_args[0], suggest_args[1] + 1)
                elif suggest_type == 'suggest_categorical':
                    params[param_name] = np.random.choice(suggest_args)
                else:
                    raise ValueError(f"Unsupported suggestion type: {suggest_type}")

            print(f"\nTrial {i+1}/{n_trials}")
            print(f"Sampled Parameters: {params}")

            config = self.create_config_from_params(params)
            score = self.objective_function(config)
            print(f"Objective Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_config_dict = config.__dict__.copy()
                print(f"*** New best score: {best_score:.4f} ***")

        results = {
            'method': 'random_search',
            'best_score': best_score,
            'best_params': best_params,
            'best_config': best_config_dict,
            'n_trials': n_trials,
            'evaluation_history': self.evaluation_history.copy()
        }

        with open(os.path.join(self.output_dir, "random_search_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def bayesian_optimization(
        self,
        n_trials: int = 30,
        # n_initial_points: int = 5 # Optuna handles startup strategy
    ) -> Dict[str, Any]:
        """Bayesian optimization using Optuna"""
        print("\n" + "="*50)
        print("BAYESIAN OPTIMIZATION HYPERPARAMETER TUNING (Optuna)")
        print("="*50)

        search_space = self.define_search_space()

        def optuna_objective_wrapper(trial):
            # Sample parameters using Optuna's trial object
            params = {}
            for param_name, definition in search_space.items():
                suggest_type = definition[0]
                suggest_args = definition[1]
                if suggest_type == 'suggest_float':
                    params[param_name] = trial.suggest_float(param_name, suggest_args[0], suggest_args[1])
                elif suggest_type == 'suggest_int':
                    params[param_name] = trial.suggest_int(param_name, suggest_args[0], suggest_args[1])
                elif suggest_type == 'suggest_categorical':
                    params[param_name] = trial.suggest_categorical(param_name, suggest_args)
                else:
                    raise ValueError(f"Unsupported suggestion type: {suggest_type}")

            print(f"\nTrial {trial.number + 1}/{n_trials}")
            print(f"Optuna Sampled Parameters: {params}")

            # Create configuration and evaluate using the main objective function
            config = self.create_config_from_params(params)
            score = self.objective_function(config) # Calls the main objective func

            print(f"Objective Score: {score:.4f}")
            return score # Optuna maximizes this score

        # Create Optuna study
        study = optuna.create_study(direction='maximize')
        best_config_dict = None # Initialize

        try:
            study.optimize(optuna_objective_wrapper, n_trials=n_trials)

            best_params = study.best_params
            best_score = study.best_value
            # Recreate the best config based on best_params
            best_config = self.create_config_from_params(best_params)
            best_config_dict = best_config.__dict__.copy()


            print(f"\n*** Optimization completed ***")
            print(f"Best score: {best_score:.4f}")
            print(f"Best parameters: {best_params}")

            results = {
                'method': 'bayesian_optimization',
                'best_score': best_score,
                'best_params': best_params,
                'best_config': best_config_dict,
                'n_trials': n_trials,
                'optuna_study': {
                    'best_trial_number': study.best_trial.number,
                    'best_value': study.best_value,
                    'best_params': study.best_params
                },
                'evaluation_history': self.evaluation_history.copy()
            }

        except Exception as e:
            print(f"Bayesian optimization failed: {e}")
            import traceback
            traceback.print_exc()
            results = {
                'method': 'bayesian_optimization',
                'error': str(e),
                'n_trials': n_trials,
                'evaluation_history': self.evaluation_history.copy() # Still save history
            }

        # Save results
        with open(os.path.join(self.output_dir, "bayesian_optimization_results.json"), 'w') as f:
            # Handle non-serializable Optuna objects if any (study object itself isn't saved)
            json.dump(results, f, indent=2, default=str)

        return results

    def evolutionary_search(
        self,
        population_size: int = 10, # Smaller population might be needed
        n_generations: int = 5,   # Fewer generations for quicker runs
        mutation_rate: float = 0.2, # Higher mutation might explore better
        crossover_rate: float = 0.7
    ) -> Dict[str, Any]:
        """Evolutionary/Genetic Algorithm optimization (Simplified)"""
        print("\n" + "="*50)
        print("EVOLUTIONARY HYPERPARAMETER OPTIMIZATION")
        print("="*50)

        search_space = self.define_search_space()
        param_names = list(search_space.keys())

        # Function to generate a random individual respecting bounds/categories
        def generate_individual():
            individual = {}
            for param_name, definition in search_space.items():
                suggest_type = definition[0]
                suggest_args = definition[1]
                if suggest_type == 'suggest_float':
                    individual[param_name] = np.random.uniform(suggest_args[0], suggest_args[1])
                elif suggest_type == 'suggest_int':
                    individual[param_name] = np.random.randint(suggest_args[0], suggest_args[1] + 1)
                elif suggest_type == 'suggest_categorical':
                    individual[param_name] = np.random.choice(suggest_args)
            return individual

        # Initialize population
        population = [generate_individual() for _ in range(population_size)]

        best_score_overall = -np.inf
        best_individual_overall = None
        best_config_dict = None

        for generation in range(n_generations):
            print(f"\nGeneration {generation + 1}/{n_generations}")

            # Evaluate population
            fitness_scores = []
            for i, individual in enumerate(population):
                print(f"  Evaluating individual {i+1}/{population_size}: {individual}")
                config = self.create_config_from_params(individual)
                score = self.objective_function(config)
                fitness_scores.append(score)
                print(f"  Score: {score:.4f}")

                if score > best_score_overall:
                    best_score_overall = score
                    best_individual_overall = individual.copy()
                    best_config_dict = config.__dict__.copy()
                    print(f"    *** New best overall score: {best_score_overall:.4f} ***")

            # --- Simple Genetic Operations ---
            if not fitness_scores: # Handle empty fitness scores
                print("Warning: No fitness scores calculated for this generation.")
                break # Stop if evaluation fails repeatedly

            fitness_scores_np = np.array(fitness_scores)
            avg_fitness = np.mean(fitness_scores_np) if fitness_scores_np.size > 0 else 0
            best_gen_fitness = np.max(fitness_scores_np) if fitness_scores_np.size > 0 else -np.inf

            print(f"  Generation best score: {best_gen_fitness:.4f}")
            print(f"  Generation average score: {avg_fitness:.4f}")


            # Selection (Tournament selection)
            new_population = []
            for _ in range(population_size):
                tournament_size = 3
                if population_size < tournament_size: # Handle small population
                    tournament_size = population_size
                if tournament_size <= 0: continue # Skip if population depleted

                contender_indices = np.random.choice(len(population), tournament_size, replace=False)
                winner_idx = contender_indices[np.argmax(fitness_scores_np[contender_indices])]
                new_population.append(population[winner_idx].copy())

            population = new_population # Parents for next gen based on selection

            # Crossover (Simplified: swap parameters) & Mutation
            offspring_population = []
            for i in range(0, population_size, 2):
                 parent1 = population[i]
                 # Ensure there is a second parent
                 parent2 = population[i+1] if i + 1 < population_size else population[np.random.randint(population_size)] # Use random if odd pop size

                 child1, child2 = parent1.copy(), parent2.copy()

                 if np.random.rand() < crossover_rate:
                     # One-point crossover on parameter dictionary keys
                     keys_to_swap = np.random.choice(param_names, size=len(param_names)//2, replace=False)
                     for key in keys_to_swap:
                         child1[key], child2[key] = child2[key], child1[key]

                 # Mutation
                 for child in [child1, child2]:
                     for param_name in param_names:
                          if np.random.rand() < mutation_rate:
                              # Re-sample the single parameter
                              definition = search_space[param_name]
                              suggest_type = definition[0]
                              suggest_args = definition[1]
                              if suggest_type == 'suggest_float':
                                  child[param_name] = np.random.uniform(suggest_args[0], suggest_args[1])
                              elif suggest_type == 'suggest_int':
                                  child[param_name] = np.random.randint(suggest_args[0], suggest_args[1] + 1)
                              elif suggest_type == 'suggest_categorical':
                                  child[param_name] = np.random.choice(suggest_args)
                     offspring_population.append(child)

            # Ensure population size is maintained
            population = offspring_population[:population_size]
            # If population shrunk due to odd number, add random individuals
            while len(population) < population_size:
                 population.append(generate_individual())


        results = {
            'method': 'evolutionary_search',
            'best_score': best_score_overall,
            'best_params': best_individual_overall,
            'best_config': best_config_dict,
            'population_size': population_size,
            'n_generations': n_generations,
            'evaluation_history': self.evaluation_history.copy()
        }

        with open(os.path.join(self.output_dir, "evolutionary_search_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    # --- Comparison Method (Unchanged structure, calls updated methods) ---
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
            self.evaluation_history = [] # Reset history for each method's run

            try:
                if method == 'grid_search':
                    # Grid search might be too slow/limited, use small settings
                    results = self.grid_search(n_points_per_dim=2, max_evaluations=min(n_trials_per_method, 16))
                elif method == 'random_search':
                    results = self.random_search(n_trials=n_trials_per_method)
                elif method == 'bayesian_optimization':
                    results = self.bayesian_optimization(n_trials=n_trials_per_method)
                elif method == 'evolutionary_search':
                    # Adjust population/generations based on total trials
                    pop = max(5, n_trials_per_method // 4)
                    gen = max(2, n_trials_per_method // pop)
                    results = self.evolutionary_search(population_size=pop, n_generations=gen)
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

    # --- Analysis Method (Unchanged structure) ---
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
            # Get correct count based on method
            if 'n_evaluations' in results: # Grid search
                n_evaluations = results['n_evaluations']
            elif 'n_trials' in results: # Random/Bayesian
                 n_evaluations = results['n_trials']
            elif 'population_size' in results: # Evolutionary
                 n_evaluations = results['population_size'] * results['n_generations']
            else:
                 n_evaluations = len(results.get('evaluation_history', []))


            print(f"\n{method.upper()}:")
            print(f"  Best Objective Score: {best_score:.4f}")
            print(f"  Number of Evaluations: {n_evaluations}")

            if 'best_params' in results and results['best_params']:
                print(f"  Best Parameters Found:")
                for param, value in results['best_params'].items():
                     if isinstance(value, float):
                         print(f"    {param}: {value:.4f}")
                     else:
                          print(f"    {param}: {value}")
            else:
                print("  Best parameters not available.")


            efficiency = best_score / n_evaluations if n_evaluations > 0 else 0
            print(f"  Efficiency (Score per Eval): {efficiency:.6f}")


            summary_data.append({
                'method': method,
                'best_score': best_score,
                'n_evaluations': n_evaluations,
                'efficiency': efficiency
            })

        # Find best method based on score
        if summary_data:
            best_method_by_score = max(summary_data, key=lambda x: x['best_score'])
            most_efficient_method = max(summary_data, key=lambda x: x['efficiency'])

            print(f"\n*** OVERALL SUMMARY ***")
            print(f"Method with Best Score: {best_method_by_score['method']} ({best_method_by_score['best_score']:.4f})")
            print(f"Most Efficient Method (Score/Eval): {most_efficient_method['method']} ({most_efficient_method['efficiency']:.6f})")

            # Save summary table
            try:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(os.path.join(self.output_dir, "method_comparison_summary.csv"), index=False)
                print(f"Comparison summary saved to {os.path.join(self.output_dir, 'method_comparison_summary.csv')}")
            except Exception as e:
                print(f"Could not save summary CSV: {e}")


def main():
    """Main function for hyperparameter tuning"""
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Streaming KVQuant Compression")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./tuning_results")
    parser.add_argument("--method", type=str, choices=[
        'grid_search', 'random_search', 'bayesian_optimization',
        'evolutionary_search', 'compare_all'
    ], default='bayesian_optimization') # Default to Bayesian
    parser.add_argument("--n_trials", type=int, default=25) # Default number of trials/evaluations

    args = parser.parse_args()

    # Create tuner instance
    tuner = HyperparameterTuner(
        model_name=args.model_name,
        device=args.device,
        output_dir=args.output_dir
    )

    # Run the selected optimization method
    print(f"Starting tuning with method: {args.method}, Number of trials/evaluations: {args.n_trials}")

    if args.method == 'grid_search':
        # Note: Grid search is limited in scope in the updated code
        results = tuner.grid_search(max_evaluations=args.n_trials)
    elif args.method == 'random_search':
        results = tuner.random_search(n_trials=args.n_trials)
    elif args.method == 'bayesian_optimization':
        results = tuner.bayesian_optimization(n_trials=args.n_trials)
    elif args.method == 'evolutionary_search':
        # Adjust population/generations based on n_trials
        pop = max(5, args.n_trials // 5)
        gen = max(3, args.n_trials // pop)
        results = tuner.evolutionary_search(population_size=pop, n_generations=gen)
    elif args.method == 'compare_all':
        # Compare fewer methods by default or make it configurable
        results = tuner.compare_methods(
            methods=['random_search', 'bayesian_optimization'], # Example subset
            n_trials_per_method=args.n_trials
        )

    print(f"\nHyperparameter tuning ({args.method}) completed!")
    print(f"Results saved to: {args.output_dir}")

    # Optionally print the best result found
    if 'best_score' in results:
         print(f"Best objective score found: {results['best_score']:.4f}")
         if 'best_params' in results:
              print("Best parameters:")
              print(json.dumps(results['best_params'], indent=2))

if __name__ == "__main__":
    main()