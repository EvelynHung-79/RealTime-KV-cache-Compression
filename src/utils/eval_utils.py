"""Evaluation utilities."""

import numpy as np
from rouge_score import rouge_scorer
from typing import List, Dict

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores.

    Args:
        predictions: List of predicted texts
        references: List of reference texts

    Returns:
        Dictionary of ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for pred, ref in zip(predictions, references):
        score = scorer.score(pred, ref)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)

    return {
        'rouge1': np.mean(scores['rouge1']),
        'rouge2': np.mean(scores['rouge2']),
        'rougeL': np.mean(scores['rougeL']),
    }

def compute_f1(predictions: List[str], references: List[str]) -> float:
    """Compute token-level F1 score."""
    f1_scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1_scores.append(0.0)
            continue

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)

    return np.mean(f1_scores)
