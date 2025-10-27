"""Data loading and preprocessing utilities."""

import torch
from datasets import load_dataset as hf_load_dataset
from typing import Dict, List

def load_dataset(dataset_name: str, split: str = "test", max_samples: int = None):
    """Load dataset from HuggingFace.

    Args:
        dataset_name: Name of dataset
        split: Dataset split
        max_samples: Maximum number of samples to load

    Returns:
        Dataset object
    """
    dataset = hf_load_dataset(dataset_name, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset

def prepare_batch(
    texts: List[str],
    tokenizer,
    max_length: int = 2048,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Prepare batch of texts for model input.

    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        device: Device to place tensors on

    Returns:
        Dictionary with input_ids and attention_mask
    """
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }
