import os
import json
import torch
from typing import Dict, List, Tuple, Optional, Any
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer
import numpy as np

class LongBenchDataLoader:
    """Data loader for LongBench dataset"""

    TASK_CONFIGS = {
        'narrativeqa': {
            'type': 'qa',
            'context_key': 'context',
            'question_key': 'input',
            'answer_key': 'answers',
            'max_length': 16384
        },
        'qasper': {
            'type': 'qa', 
            'context_key': 'context',
            'question_key': 'input',
            'answer_key': 'answers',
            'max_length': 16384
        },
        'multifieldqa_en': {
            'type': 'qa',
            'context_key': 'context',
            'question_key': 'input', 
            'answer_key': 'answers',
            'max_length': 8192
        },
        'multifieldqa_zh': {
            'type': 'qa',
            'context_key': 'context',
            'question_key': 'input',
            'answer_key': 'answers',
            'max_length': 8192
        },
        'hotpotqa': {
            'type': 'qa',
            'context_key': 'context',
            'question_key': 'input',
            'answer_key': 'answers',
            'max_length': 16384
        },
        '2wikimqa': {
            'type': 'qa',
            'context_key': 'context', 
            'question_key': 'input',
            'answer_key': 'answers',
            'max_length': 16384
        },
        'musique': {
            'type': 'qa',
            'context_key': 'context',
            'question_key': 'input',
            'answer_key': 'answers',
            'max_length': 16384
        },
        'gov_report': {
            'type': 'summarization',
            'context_key': 'context',
            'question_key': None,
            'answer_key': 'answers',
            'max_length': 32768
        },
        'qmsum': {
            'type': 'summarization',
            'context_key': 'context',
            'question_key': None,
            'answer_key': 'answers',
            'max_length': 16384
        },
        'multi_news': {
            'type': 'summarization',
            'context_key': 'context',
            'question_key': None,
            'answer_key': 'answers',
            'max_length': 16384
        },
        'vcsum': {
            'type': 'summarization',
            'context_key': 'context',
            'question_key': None,
            'answer_key': 'answers',
            'max_length': 16384
        },
        'trec': {
            'type': 'classification',
            'context_key': 'context',
            'question_key': 'input',
            'answer_key': 'answers',
            'max_length': 4096
        },
        'triviaqa': {
            'type': 'qa',
            'context_key': 'context',
            'question_key': 'input',
            'answer_key': 'answers',
            'max_length': 16384
        }
    }

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def load_task_dataset(self, task_name: str) -> Optional[Dataset]:
        """Load dataset for specific task"""
        if task_name not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task_name}")

        try:
            dataset = load_dataset(
                "THUDM/LongBench", 
                task_name, 
                split="test",
                cache_dir=self.cache_dir
            )
            return dataset
        except Exception as e:
            print(f"Failed to load {task_name}: {e}")
            return None

    def preprocess_sample(self, sample: Dict, task_name: str, tokenizer: PreTrainedTokenizer) -> Dict:
        """Preprocess a single sample"""
        config = self.TASK_CONFIGS[task_name]

        # Extract context and question
        context = sample.get(config['context_key'], '')
        question = sample.get(config['question_key'], '') if config['question_key'] else ''

        # Format prompt based on task type
        if config['type'] == 'qa':
            if question:
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            else:
                prompt = f"Context: {context}\n\nAnswer:"
        elif config['type'] == 'summarization':
            prompt = f"Please summarize the following text:\n\n{context}\n\nSummary:"
        elif config['type'] == 'classification':
            prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"{context}\n{question}" if question else context

        # Tokenize and truncate if necessary
        max_length = min(config['max_length'], tokenizer.model_max_length)

        # Estimate prompt length
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

        if len(prompt_tokens) > max_length - 100:  # Leave room for generation
            # Truncate context while preserving question
            if config['type'] == 'qa' and question:
                question_tokens = tokenizer.encode(f"Question: {question}\n\nAnswer:", add_special_tokens=False)
                available_context_length = max_length - len(question_tokens) - 50

                context_tokens = tokenizer.encode(context, add_special_tokens=False)
                if len(context_tokens) > available_context_length:
                    # Keep both beginning and end of context
                    keep_start = available_context_length // 2
                    keep_end = available_context_length - keep_start
                    truncated_tokens = context_tokens[:keep_start] + context_tokens[-keep_end:]
                    context = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            else:
                # Simple truncation
                available_length = max_length - 50
                truncated_tokens = prompt_tokens[:available_length]
                prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        # Extract reference answer
        answers = sample.get(config['answer_key'], [])
        if isinstance(answers, list) and len(answers) > 0:
            reference = answers[0]
        elif isinstance(answers, str):
            reference = answers
        else:
            reference = ""

        return {
            'prompt': prompt,
            'reference': reference,
            'task_type': config['type'],
            'original_length': len(tokenizer.encode(prompt, add_special_tokens=False))
        }

class DataCollator:
    """Collate function for batching samples"""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.model_max_length

    def __call__(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate samples into batch"""
        prompts = [sample['prompt'] for sample in samples]
        references = [sample['reference'] for sample in samples]

        # Tokenize prompts
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'references': references,
            'prompts': prompts
        }

def create_synthetic_long_context(
    tokenizer: PreTrainedTokenizer,
    base_length: int = 4096,
    prompt_length: int = 128,
    seed: int = 42
) -> Tuple[str, torch.Tensor]:
    """Create synthetic long context for testing"""

    np.random.seed(seed)

    # Generate random tokens (excluding special tokens)
    vocab_size = tokenizer.vocab_size
    special_token_ids = set([
        tokenizer.pad_token_id, tokenizer.eos_token_id, 
        tokenizer.bos_token_id, tokenizer.unk_token_id
    ])

    # Filter out special tokens
    valid_token_ids = [i for i in range(vocab_size) if i not in special_token_ids]

    # Generate context tokens
    context_token_ids = np.random.choice(valid_token_ids, size=base_length - prompt_length)

    # Generate prompt tokens
    prompt_token_ids = np.random.choice(valid_token_ids, size=prompt_length)

    # Combine
    all_token_ids = np.concatenate([prompt_token_ids, context_token_ids])

    # Decode to text
    text = tokenizer.decode(all_token_ids, skip_special_tokens=True)

    # Create prompt indices tensor
    prompt_indices = torch.arange(prompt_length)

    return text, prompt_indices

def estimate_memory_requirements(
    model_name: str,
    context_length: int,
    batch_size: int = 1,
    precision: str = "fp16"
) -> Dict[str, float]:
    """Estimate memory requirements for model and context"""

    # Model parameter counts (approximate)
    model_sizes = {
        "meta-llama/Llama-2-7b-hf": 7e9,
        "meta-llama/Llama-2-13b-hf": 13e9,
        "meta-llama/Llama-2-70b-hf": 70e9
    }

    if model_name not in model_sizes:
        # Estimate based on name
        if "7b" in model_name.lower():
            param_count = 7e9
        elif "13b" in model_name.lower():
            param_count = 13e9
        elif "70b" in model_name.lower():
            param_count = 70e9
        else:
            param_count = 7e9  # Default
    else:
        param_count = model_sizes[model_name]

    # Bytes per parameter
    bytes_per_param = 2 if precision == "fp16" else 4  # fp32

    # Model memory
    model_memory = param_count * bytes_per_param / (1024**3)  # GB

    # KV cache memory (approximate)
    # Assumes hidden_size = 4096, num_layers = 32 for 7B model
    hidden_size = 4096 if "7b" in model_name.lower() else 5120
    num_layers = 32 if "7b" in model_name.lower() else 40

    kv_cache_memory = (
        2 * batch_size * context_length * num_layers * hidden_size * bytes_per_param / (1024**3)
    )

    # Activation memory (rough estimate)
    activation_memory = (
        batch_size * context_length * hidden_size * bytes_per_param * 4 / (1024**3)
    )

    return {
        'model_memory_gb': model_memory,
        'kv_cache_memory_gb': kv_cache_memory,
        'activation_memory_gb': activation_memory,
        'total_estimated_gb': model_memory + kv_cache_memory + activation_memory
    }

def save_compression_data(
    results: Dict,
    output_path: str,
    format: str = "json"
):
    """Save compression experimental data"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    elif format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_compression_data(
    input_path: str,
    format: str = "json"
) -> Dict:
    """Load compression experimental data"""

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if format == "json":
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif format == "jsonl":
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    else:
        raise ValueError(f"Unsupported format: {format}")