"""LongBench evaluation for long-context understanding."""

import torch
from typing import Dict, List
from tqdm import tqdm
from ..utils.eval_utils import compute_rouge, compute_f1
from ..utils.data_utils import load_dataset

class LongBenchEvaluator:
    """Evaluator for LongBench dataset."""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(
        self,
        dataset_name: str = "THUDM/LongBench",
        task: str = "narrativeqa",
        max_samples: int = 100,
    ) -> Dict:
        """Evaluate model on LongBench task.

        Args:
            dataset_name: LongBench dataset name
            task: Specific task to evaluate
            max_samples: Maximum number of samples

        Returns:
            Dictionary of evaluation metrics
        """
        # Load dataset
        dataset = load_dataset(dataset_name, split="test", max_samples=max_samples)

        predictions = []
        references = []
        compression_stats_list = []

        self.model.eval()
        with torch.no_grad():
            for example in tqdm(dataset, desc=f"Evaluating {task}"):
                # Reset compression state
                if hasattr(self.model, 'reset_compression_state'):
                    self.model.reset_compression_state()

                # Prepare input
                context = example.get('context', '')
                question = example.get('question', '')
                reference = example.get('answer', '')

                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

                # Generate
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                )

                # Decode
                prediction = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                predictions.append(prediction)
                references.append(reference)

                # Collect compression stats
                if hasattr(self.model, 'get_compression_stats'):
                    stats = self.model.get_compression_stats()
                    compression_stats_list.append(stats)

        # Compute metrics
        rouge_scores = compute_rouge(predictions, references)
        f1_score = compute_f1(predictions, references)

        # Aggregate compression stats
        if compression_stats_list:
            avg_compression_stats = self._aggregate_compression_stats(compression_stats_list)
        else:
            avg_compression_stats = {}

        results = {
            **rouge_scores,
            'f1': f1_score,
            **avg_compression_stats,
        }

        return results

    def _aggregate_compression_stats(self, stats_list: List[Dict]) -> Dict:
        """Aggregate compression statistics across samples."""
        if not stats_list:
            return {}

        aggregated = {}
        keys = ['avg_key_bits', 'avg_value_bits', 'outlier_ratio', 
                'compression_ratio_key', 'compression_ratio_value']

        for key in keys:
            values = [s.get(key, 0) for s in stats_list if key in s]
            if values:
                aggregated[f'mean_{key}'] = sum(values) / len(values)

        return aggregated
