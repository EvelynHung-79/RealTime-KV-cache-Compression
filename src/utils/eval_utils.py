import logging
import os
import re
from typing import List, Dict, Any
from rouge_score import rouge_scorer
import numpy as np

def setup_logging(log_file: str = None, level: int = logging.INFO):
    """Setup logging configuration"""
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if log_file:
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=level, format=format_str)

def normalize_text(text: str) -> str:
    """Normalize text for evaluation"""
    # Convert to lowercase
    text = text.lower().strip()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove punctuation for some metrics
    text = re.sub(r'[^\w\s]', '', text)

    return text

def calculate_rouge(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores"""
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)

        return {
            'rouge_1': scores['rouge1'].fmeasure,
            'rouge_2': scores['rouge2'].fmeasure,
            'rouge_l': scores['rougeL'].fmeasure
        }
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}

def calculate_f1(prediction: str, reference: str) -> float:
    """Calculate F1 score for QA tasks"""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Calculate precision and recall
    common_tokens = set(pred_tokens) & set(ref_tokens)

    if not common_tokens:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1

def calculate_accuracy(prediction: str, reference: str) -> float:
    """Calculate exact match accuracy"""
    pred_normalized = normalize_text(prediction)
    ref_normalized = normalize_text(reference)

    return 1.0 if pred_normalized == ref_normalized else 0.0

def calculate_bleu(prediction: str, reference: str) -> float:
    """Calculate BLEU score (simplified implementation)"""
    try:
        from sacrebleu import sentence_bleu
        score = sentence_bleu(prediction, [reference])
        return score.score / 100.0  # Normalize to 0-1
    except ImportError:
        # Fallback to simple n-gram overlap
        pred_tokens = normalize_text(prediction).split()
        ref_tokens = normalize_text(reference).split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        # Simple unigram BLEU approximation
        common = set(pred_tokens) & set(ref_tokens)
        return len(common) / len(pred_tokens) if pred_tokens else 0.0

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across multiple samples"""
    if not metrics_list:
        return {}

    # Get all metric names
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    # Calculate mean for each metric
    aggregated = {}
    for key in all_keys:
        values = [m.get(key, 0.0) for m in metrics_list]
        aggregated[f'avg_{key}'] = np.mean(values)
        aggregated[f'std_{key}'] = np.std(values)
        aggregated[f'min_{key}'] = np.min(values)
        aggregated[f'max_{key}'] = np.max(values)

    return aggregated