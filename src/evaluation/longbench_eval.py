import torch
import json
import os
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import time

from .metrics import CompressionMetrics
from ..utils.eval_utils import calculate_rouge, calculate_f1, calculate_accuracy

class LongBenchEvaluator:
    """
    LongBench evaluation suite for compressed LLM models
    Supports 13 long-context tasks with compression-aware metrics
    """

    LONGBENCH_TASKS = [
        # Single-Document QA
        'narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh',
        # Multi-Document QA  
        'hotpotqa', '2wikimqa', 'musique',
        # Summarization
        'gov_report', 'qmsum', 'multi_news', 'vcsum',
        # Few-shot Learning
        'trec', 'triviaqa',
        # Synthetic Tasks
        'samsum', 'lsht', 'passage_count', 'passage_retrieval_en', 'passage_retrieval_zh'
    ]

    def __init__(self, model, tokenizer, config, output_dir: str = "./longbench_results"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = output_dir
        self.metrics_calculator = CompressionMetrics()

        os.makedirs(output_dir, exist_ok=True)

    def load_longbench_dataset(self, task_name: str) -> Dataset:
        """Load specific LongBench task dataset"""
        try:
            # Try loading from HuggingFace datasets
            dataset = load_dataset("THUDM/LongBench", task_name, split="test")
            return dataset
        except Exception as e:
            print(f"Failed to load {task_name} from HuggingFace: {e}")
            # Fallback to local loading if available
            return self._load_local_longbench(task_name)

    def _load_local_longbench(self, task_name: str) -> Optional[Dataset]:
        """Load LongBench dataset from local files"""
        local_path = f"./data/longbench/{task_name}.jsonl"

        if not os.path.exists(local_path):
            print(f"Local dataset not found: {local_path}")
            return None

        data = []
        with open(local_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        return Dataset.from_list(data)

    def format_prompt(self, sample: Dict, task_name: str) -> str:
        """Format input prompt based on task type"""

        if task_name in ['narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh']:
            # Single-document QA format
            context = sample.get('context', '')
            question = sample.get('input', '')
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        elif task_name in ['hotpotqa', '2wikimqa', 'musique']:
            # Multi-document QA format
            context = sample.get('context', '')
            question = sample.get('input', '')
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        elif task_name in ['gov_report', 'qmsum', 'multi_news', 'vcsum']:
            # Summarization format
            text = sample.get('context', '')
            prompt = f"Please summarize the following text:\n\n{text}\n\nSummary:"

        elif task_name in ['trec', 'triviaqa']:
            # Few-shot learning format
            context = sample.get('context', '')
            question = sample.get('input', '')
            prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"

        else:
            # Default format
            context = sample.get('context', '')
            input_text = sample.get('input', '')
            if context:
                prompt = f"Context: {context}\n\nInput: {input_text}\n\nOutput:"
            else:
                prompt = f"Input: {input_text}\n\nOutput:"

        return prompt

    def generate_response(
        self, 
        prompt: str, 
        max_new_tokens: int = 100,
        temperature: float = 0.1
    ) -> Tuple[str, Dict]:
        """Generate response with compression statistics"""

        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_position_embeddings
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Reset compression state for new sequence
        if hasattr(self.model, 'reset_compression_state'):
            self.model.reset_compression_state()

        start_time = time.time()

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generation_time = time.time() - start_time

        # Decode response
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Get compression statistics
        compression_stats = {}
        if hasattr(self.model, 'get_compression_stats'):
            compression_stats = self.model.get_compression_stats()

        # Calculate performance metrics
        performance_stats = {
            'generation_time': generation_time,
            'input_length': input_length,
            'output_length': len(generated_tokens),
            'total_tokens': input_length + len(generated_tokens),
            'tokens_per_second': len(generated_tokens) / generation_time if generation_time > 0 else 0,
            'ttft': compression_stats.get('total_processing_time', 0),  # Approximate TTFT
        }

        return response, {**compression_stats, **performance_stats}

    def evaluate_task(
        self, 
        task_name: str, 
        max_samples: Optional[int] = None,
        max_new_tokens: int = 100
    ) -> Dict:
        """Evaluate model on specific LongBench task"""

        print(f"Evaluating task: {task_name}")

        # Load dataset
        dataset = self.load_longbench_dataset(task_name)
        if dataset is None:
            return {"error": f"Could not load dataset for {task_name}"}

        # Limit samples if specified
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        results = []
        all_predictions = []
        all_references = []
        all_compression_stats = []

        for i, sample in enumerate(tqdm(dataset, desc=f"Processing {task_name}")):
            try:
                # Format prompt
                prompt = self.format_prompt(sample, task_name)

                # Generate response
                prediction, stats = self.generate_response(
                    prompt, max_new_tokens=max_new_tokens
                )

                # Get reference answer
                reference = sample.get('answers', [''])[0] if isinstance(sample.get('answers'), list) else sample.get('answer', '')

                # Calculate task-specific metrics
                task_metrics = self._calculate_task_metrics(prediction, reference, task_name)

                result = {
                    'sample_id': i,
                    'prediction': prediction,
                    'reference': reference,
                    'task_metrics': task_metrics,
                    'compression_stats': stats
                }

                results.append(result)
                all_predictions.append(prediction)
                all_references.append(reference)
                all_compression_stats.append(stats)

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        # Aggregate results
        task_results = self._aggregate_task_results(
            all_predictions, all_references, all_compression_stats, task_name
        )

        # Save detailed results
        results_file = os.path.join(self.output_dir, f"{task_name}_detailed_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Task {task_name} completed. Results saved to {results_file}")

        return task_results

    def _calculate_task_metrics(self, prediction: str, reference: str, task_name: str) -> Dict:
        """Calculate task-specific evaluation metrics"""

        metrics = {}

        if task_name in ['narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh', 
                        'hotpotqa', '2wikimqa', 'musique']:
            # QA tasks: use F1 and exact match
            metrics['f1'] = calculate_f1(prediction, reference)
            metrics['exact_match'] = 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0

        elif task_name in ['gov_report', 'qmsum', 'multi_news', 'vcsum']:
            # Summarization tasks: use ROUGE scores
            rouge_scores = calculate_rouge(prediction, reference)
            metrics.update(rouge_scores)

        elif task_name in ['trec', 'triviaqa']:
            # Classification/QA tasks: use accuracy
            metrics['accuracy'] = calculate_accuracy(prediction, reference)
            metrics['f1'] = calculate_f1(prediction, reference)

        else:
            # Default: use multiple metrics
            metrics['f1'] = calculate_f1(prediction, reference)
            metrics['rouge_l'] = calculate_rouge(prediction, reference).get('rouge_l', 0)

        return metrics

    def _aggregate_task_results(
        self, 
        predictions: List[str], 
        references: List[str],
        compression_stats: List[Dict],
        task_name: str
    ) -> Dict:
        """Aggregate results for entire task"""

        # Aggregate quality metrics
        quality_metrics = {}

        if task_name in ['narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh',
                        'hotpotqa', '2wikimqa', 'musique']:
            f1_scores = [calculate_f1(p, r) for p, r in zip(predictions, references)]
            exact_matches = [1.0 if p.strip().lower() == r.strip().lower() else 0.0 
                           for p, r in zip(predictions, references)]
            quality_metrics = {
                'avg_f1': np.mean(f1_scores),
                'avg_exact_match': np.mean(exact_matches)
            }

        elif task_name in ['gov_report', 'qmsum', 'multi_news', 'vcsum']:
            rouge_scores = [calculate_rouge(p, r) for p, r in zip(predictions, references)]
            quality_metrics = {
                'avg_rouge_1': np.mean([s.get('rouge_1', 0) for s in rouge_scores]),
                'avg_rouge_2': np.mean([s.get('rouge_2', 0) for s in rouge_scores]), 
                'avg_rouge_l': np.mean([s.get('rouge_l', 0) for s in rouge_scores])
            }

        # Aggregate compression metrics
        if compression_stats:
            compression_metrics = {
                'avg_compression_ratio': np.mean([s.get('avg_compression_ratio', 1.0) for s in compression_stats]),
                'avg_memory_savings': np.mean([s.get('avg_memory_savings', 0.0) for s in compression_stats]),
                'avg_processing_time': np.mean([s.get('total_processing_time', 0.0) for s in compression_stats]),
                'avg_tokens_per_second': np.mean([s.get('tokens_per_second', 0.0) for s in compression_stats]),
                'avg_precision_distribution': {
                    'high': np.mean([s.get('precision_distribution', {}).get('high_ratio', 0) for s in compression_stats]),
                    'medium': np.mean([s.get('precision_distribution', {}).get('medium_ratio', 0) for s in compression_stats]),
                    'low': np.mean([s.get('precision_distribution', {}).get('low_ratio', 0) for s in compression_stats])
                }
            }
        else:
            compression_metrics = {}

        return {
            'task_name': task_name,
            'num_samples': len(predictions),
            'quality_metrics': quality_metrics,
            'compression_metrics': compression_metrics,
            'overall_score': quality_metrics.get('avg_f1', quality_metrics.get('avg_rouge_l', 0))
        }

    def evaluate_all_tasks(
        self, 
        tasks: Optional[List[str]] = None,
        max_samples_per_task: Optional[int] = None
    ) -> Dict:
        """Evaluate model on all or selected LongBench tasks"""

        tasks = tasks or self.LONGBENCH_TASKS
        all_results = {}

        print(f"Starting evaluation on {len(tasks)} tasks...")

        for task in tasks:
            try:
                result = self.evaluate_task(
                    task, 
                    max_samples=max_samples_per_task
                )
                all_results[task] = result
            except Exception as e:
                print(f"Failed to evaluate {task}: {e}")
                all_results[task] = {"error": str(e)}

        # Aggregate overall results
        overall_results = self._aggregate_overall_results(all_results)

        # Save comprehensive results
        results_file = os.path.join(self.output_dir, "comprehensive_evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'overall_results': overall_results,
                'task_results': all_results
            }, f, indent=2, ensure_ascii=False)

        print(f"\nEvaluation completed! Results saved to {results_file}")

        return overall_results

    def _aggregate_overall_results(self, task_results: Dict) -> Dict:
        """Aggregate results across all tasks"""

        successful_tasks = [r for r in task_results.values() if 'error' not in r]

        if not successful_tasks:
            return {"error": "No tasks completed successfully"}

        # Average quality metrics
        avg_overall_score = np.mean([r['overall_score'] for r in successful_tasks])

        # Average compression metrics
        compression_metrics = {}
        if all('compression_metrics' in r for r in successful_tasks):
            compression_metrics = {
                'overall_avg_compression_ratio': np.mean([
                    r['compression_metrics'].get('avg_compression_ratio', 1.0) 
                    for r in successful_tasks
                ]),
                'overall_avg_memory_savings': np.mean([
                    r['compression_metrics'].get('avg_memory_savings', 0.0)
                    for r in successful_tasks
                ]),
                'overall_avg_processing_time': np.mean([
                    r['compression_metrics'].get('avg_processing_time', 0.0)
                    for r in successful_tasks
                ])
            }

        return {
            'num_successful_tasks': len(successful_tasks),
            'num_total_tasks': len(task_results),
            'overall_quality_score': avg_overall_score,
            'compression_performance': compression_metrics,
            'task_breakdown': {
                task: {'score': result.get('overall_score', 0)} 
                for task, result in task_results.items() 
                if 'error' not in result
            }
        }