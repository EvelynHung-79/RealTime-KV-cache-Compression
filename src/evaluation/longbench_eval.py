import torch
import json
import os
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import time
import logging # 加入 logging

# 假設 metrics.py 提供了更新後的指標計算或不再需要
# from .metrics import CompressionMetrics
# 假設 utils.eval_utils 提供了所需的 F1, ROUGE, Accuracy 計算
from ..utils.eval_utils import calculate_rouge, calculate_f1, calculate_accuracy
# 引用更新後的 Config
from configs.base_config import CompressionConfig

logger = logging.getLogger(__name__) # Use logger

class LongBenchEvaluator:
    """
    針對使用串流 KVQuant 量化 (無選擇性傳播) 的 LLM 模型，
    執行 LongBench 評估套件。
    """

    # LONGBENCH_TASKS 列表保持不變
    LONGBENCH_TASKS = [
        # Single-Document QA
        'narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh',
        # Multi-Document QA
        'hotpotqa', '2wikimqa', 'musique',
        # Summarization
        'gov_report', 'qmsum', 'multi_news', 'vcsum',
        # Few-shot Learning
        'trec', 'triviaqa',
        # Synthetic Tasks (可能需要根據實際支持情況調整)
        'samsum', 'lsht', 'passage_count', 'passage_retrieval_en', 'passage_retrieval_zh'
    ]

    def __init__(self, model, tokenizer, config: CompressionConfig, output_dir: str = "./longbench_results"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config # 現在是更新後的 CompressionConfig
        self.output_dir = output_dir
        # self.metrics_calculator = CompressionMetrics() # 移除或替換為新的統計聚合邏輯

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"LongBenchEvaluator initialized. Results will be saved to: {output_dir}")

    def load_longbench_dataset(self, task_name: str) -> Dataset:
        """載入指定的 LongBench 任務數據集 (保持不變)"""
        try:
            logger.info(f"Loading LongBench dataset for task: {task_name}")
            dataset = load_dataset("THUDM/LongBench", task_name, split="test")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load {task_name} from HuggingFace: {e}")
            # Fallback to local loading if available
            return self._load_local_longbench(task_name)

    def _load_local_longbench(self, task_name: str) -> Optional[Dataset]:
        """從本地文件載入 LongBench 數據集 (保持不變)"""
        local_path = f"./data/longbench/{task_name}.jsonl" # 假設數據在此路徑
        logger.info(f"Attempting to load local dataset from: {local_path}")

        if not os.path.exists(local_path):
            logger.warning(f"Local dataset not found: {local_path}")
            return None

        data = []
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            logger.info(f"Successfully loaded {len(data)} samples locally for {task_name}.")
            return Dataset.from_list(data)
        except Exception as e:
            logger.error(f"Error loading local dataset {local_path}: {e}")
            return None

    def format_prompt(self, sample: Dict, task_name: str) -> str:
        """根據任務類型格式化輸入提示 (保持不變)"""
        # ... (這部分邏輯與您之前版本相同) ...
        # --- Start of format_prompt logic (unchanged) ---
        if task_name in ['narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh']:
            context = sample.get('context', '')
            question = sample.get('input', '')
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        elif task_name in ['hotpotqa', '2wikimqa', 'musique']:
            context = sample.get('context', '')
            question = sample.get('input', '')
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        elif task_name in ['gov_report', 'qmsum', 'multi_news', 'vcsum']:
            text = sample.get('context', '')
            prompt = f"Please summarize the following text:\n\n{text}\n\nSummary:"
        elif task_name in ['trec', 'triviaqa']:
            context = sample.get('context', '')
            question = sample.get('input', '')
            prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
        else: # Default format
            context = sample.get('context', '')
            input_text = sample.get('input', '')
            if context:
                prompt = f"Context: {context}\n\nInput: {input_text}\n\nOutput:"
            else:
                prompt = f"Input: {input_text}\n\nOutput:"
        # --- End of format_prompt logic ---
        return prompt


    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.1 # 較低的 temperature 以獲得更確定的輸出
    ) -> Tuple[str, Dict]:
        """使用模型生成回應並收集量化統計數據"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_position_embeddings # 使用 config 中的 max_length
        )
        input_length = inputs['input_ids'].shape[1]

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # !!! 重置模型的串流統計數據 !!!
        if hasattr(self.model, 'reset_compression_state'):
            logger.debug("Resetting model compression state.")
            self.model.reset_compression_state() # 確保調用的是新模型的方法

        start_time = time.time()
        ttft = 0 # 初始化 TTFT

        # 使用 generate 進行推論
        with torch.no_grad():
            # --- 測量 TTFT (近似) ---
            # 如果模型內部能記錄 prefill 時間則更好
            # 這裡我們近似測量到第一個 token 生成的時間
            # 注意：Hugging Face generate 包含 prefill 和 decode
            # 要精確測量 prefill，需要在模型 forward 中計時
            # 這裡我們先測量 generate 的總時間
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0 and temperature != 0, # 僅在 temp > 0 時啟用 sample
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generation_time = time.time() - start_time # 這是總時間

        # 解碼回應
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        output_length = len(generated_tokens)

        # !!! 獲取新的量化統計數據 !!!
        quant_stats = {}
        if hasattr(self.model, 'get_compression_stats'):
            try:
                 quant_stats = self.model.get_compression_stats()
                 logger.debug(f"Retrieved quantization stats: {quant_stats}")
            except Exception as e:
                 logger.error(f"Failed to get compression stats: {e}")

        # 計算性能指標
        performance_stats = {
            'total_generation_time': generation_time, # 包含 prefill + decode
            'input_length': input_length,
            'output_length': output_length,
            'total_tokens': input_length + output_length,
            # Tokens per second 應該只計算 decode 階段，但這裡只有總時間
            'approx_tokens_per_second': output_length / generation_time if generation_time > 0 and output_length > 0 else 0,
            # TTFT 仍然難以精確測量，除非模型內部提供
            'ttft_placeholder': quant_stats.get('prefill_time', 0) # 假設模型能提供 prefill 時間
        }

        # 合併統計數據
        # **quant_stats 包含了模型內部返回的所有統計信息
        # （例如 estimated_memory_savings, outlier counts 等）
        combined_stats = {**quant_stats, **performance_stats}

        return response, combined_stats

    def evaluate_task(
        self,
        task_name: str,
        max_samples: Optional[int] = None,
        max_new_tokens: int = 100
    ) -> Dict:
        """在指定的 LongBench 任務上評估模型"""

        logger.info(f"Starting evaluation for task: {task_name}")

        dataset = self.load_longbench_dataset(task_name)
        if dataset is None:
            logger.error(f"Could not load dataset for {task_name}. Skipping.")
            return {"error": f"Could not load dataset for {task_name}"}

        if max_samples and len(dataset) > max_samples:
            logger.info(f"Limiting evaluation to {max_samples} samples for task {task_name}.")
            dataset = dataset.select(range(max_samples))

        results = []
        all_predictions = []
        all_references = []
        all_quant_stats = [] # 儲存新的統計數據

        for i, sample in enumerate(tqdm(dataset, desc=f"Processing {task_name}")):
            try:
                prompt = self.format_prompt(sample, task_name)
                prediction, stats = self.generate_response(
                    prompt, max_new_tokens=max_new_tokens
                )

                # Reference extraction (unchanged)
                reference = sample.get('answers', [''])[0] if isinstance(sample.get('answers'), list) else sample.get('answer', '')
                if not isinstance(reference, str): reference = str(reference) # Ensure string

                task_metrics = self._calculate_task_metrics(prediction, reference, task_name)

                result = {
                    'sample_id': i,
                    'prediction': prediction,
                    'reference': reference,
                    'task_metrics': task_metrics,
                    'quantization_stats': stats # 使用新的 key 名稱
                }

                results.append(result)
                all_predictions.append(prediction)
                all_references.append(reference)
                all_quant_stats.append(stats) # 收集新的統計數據

            except Exception as e:
                logger.error(f"Error processing sample {i} in task {task_name}: {e}", exc_info=True) # Log traceback
                # Optionally add error record
                results.append({'sample_id': i, 'error': str(e)})
                all_quant_stats.append({}) # Add empty dict to maintain list length
                continue

        # 聚合任務結果
        task_results = self._aggregate_task_results(
            all_predictions, all_references, all_quant_stats, task_name # 傳遞新的統計數據
        )

        # 保存詳細結果
        results_file = os.path.join(self.output_dir, f"{task_name}_detailed_results.json")
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Task {task_name} completed. Detailed results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save detailed results for {task_name} to {results_file}: {e}")

        return task_results

    def _calculate_task_metrics(self, prediction: str, reference: str, task_name: str) -> Dict:
        """計算特定任務的評估指標 (保持不變)"""
        # ... (這部分邏輯與您之前版本相同，使用 calculate_f1, calculate_rouge, etc.) ...
        # --- Start of _calculate_task_metrics logic (unchanged) ---
        metrics = {}
        if task_name in ['narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh',
                        'hotpotqa', '2wikimqa', 'musique']:
            metrics['f1'] = calculate_f1(prediction, reference)
            metrics['exact_match'] = calculate_accuracy(prediction, reference) # Use accuracy function
        elif task_name in ['gov_report', 'qmsum', 'multi_news', 'vcsum']:
            rouge_scores = calculate_rouge(prediction, reference)
            metrics.update(rouge_scores)
        elif task_name in ['trec', 'triviaqa']:
             metrics['accuracy'] = calculate_accuracy(prediction, reference)
             # Optionally keep F1 if relevant for TriviaQA
             if task_name == 'triviaqa':
                 metrics['f1'] = calculate_f1(prediction, reference)
        else: # Default for other tasks
            metrics['f1'] = calculate_f1(prediction, reference)
            metrics['rouge_l'] = calculate_rouge(prediction, reference).get('rouge_l', 0)
        # --- End of _calculate_task_metrics logic ---
        return metrics

    def _aggregate_task_results(
        self,
        predictions: List[str],
        references: List[str],
        quant_stats_list: List[Dict], # 接收新的統計數據列表
        task_name: str
    ) -> Dict:
        """聚合整個任務的結果"""

        num_samples = len(predictions)
        if num_samples == 0:
            logger.warning(f"No successful samples to aggregate for task {task_name}.")
            return {'task_name': task_name, 'num_samples': 0}

        # 聚合品質指標 (Quality Metrics) - 與之前邏輯類似
        quality_metrics = {}
        all_task_metrics = [self._calculate_task_metrics(p, r, task_name) for p, r in zip(predictions, references)]

        # Dynamically find all metric keys computed by _calculate_task_metrics
        metric_keys = set(key for sample_metrics in all_task_metrics for key in sample_metrics.keys())

        for key in metric_keys:
             values = [m.get(key, 0.0) for m in all_task_metrics]
             quality_metrics[f'avg_{key}'] = np.mean(values) if values else 0.0
             # Optionally add std, min, max
             # quality_metrics[f'std_{key}'] = np.std(values) if values else 0.0

        # !!! 聚合量化與性能指標 (Quantization & Performance Metrics) !!!
        quant_perf_metrics = {}
        if quant_stats_list:
            # 找出所有樣本報告的統計數據鍵名 (除了 config)
            stat_keys = set(k for stats in quant_stats_list if stats for k in stats.keys() if k != 'config')

            for key in stat_keys:
                values = [s.get(key) for s in quant_stats_list if s and s.get(key) is not None]
                # 只聚合數值類型
                if values and isinstance(values[0], (int, float)):
                    quant_perf_metrics[f'avg_{key}'] = np.mean(values)
                    # Optionally add std, min, max if needed
                    # quant_perf_metrics[f'std_{key}'] = np.std(values)
                elif values: # Handle non-numeric stats if needed (e.g., list or dict summaries)
                     # Example: Count average number of outliers if reported per sample
                     if key == 'key_outliers_detected': # Assuming this key exists
                          quant_perf_metrics['avg_key_outliers'] = np.mean(values)

        # 決定主要得分 (Overall Score) - F1 優先，其次 ROUGE-L，最後 Accuracy
        overall_score = quality_metrics.get('avg_f1',
                            quality_metrics.get('avg_rouge_l',
                                quality_metrics.get('avg_accuracy', 0.0)))

        return {
            'task_name': task_name,
            'num_samples': num_samples,
            'quality_metrics': quality_metrics,
            'quantization_performance': quant_perf_metrics, # 使用新的 key 名稱
            'overall_score': overall_score
        }

    def evaluate_all_tasks(
        self,
        tasks: Optional[List[str]] = None,
        max_samples_per_task: Optional[int] = None,
        max_new_tokens_per_task: Optional[Dict[str, int]] = None # Allow task-specific max_new_tokens
    ) -> Dict:
        """在所有或選定的 LongBench 任務上評估模型"""

        tasks_to_run = tasks or self.LONGBENCH_TASKS
        all_results = {}
        default_max_new_tokens = 100 # Default if not specified per task

        logger.info(f"Starting evaluation on {len(tasks_to_run)} tasks: {tasks_to_run}")

        for task in tasks_to_run:
            if task not in self.LONGBENCH_TASKS:
                 logger.warning(f"Skipping unknown task: {task}")
                 continue

            task_max_new_tokens = default_max_new_tokens
            if max_new_tokens_per_task and task in max_new_tokens_per_task:
                 task_max_new_tokens = max_new_tokens_per_task[task]
                 logger.info(f"Using max_new_tokens={task_max_new_tokens} for task {task}")

            try:
                result = self.evaluate_task(
                    task,
                    max_samples=max_samples_per_task,
                    max_new_tokens=task_max_new_tokens
                )
                all_results[task] = result
            except Exception as e:
                logger.error(f"Evaluation failed catastrophically for task {task}: {e}", exc_info=True)
                all_results[task] = {"error": f"Catastrophic failure: {str(e)}"}

        # 聚合總體結果
        overall_results = self._aggregate_overall_results(all_results)

        # 保存綜合結果
        results_file = os.path.join(self.output_dir, "comprehensive_evaluation_results.json")
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'overall_results': overall_results,
                    'task_results': all_results
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"\nEvaluation completed! Comprehensive results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save comprehensive results to {results_file}: {e}")

        return overall_results

    def _aggregate_overall_results(self, task_results: Dict) -> Dict:
        """聚合所有任務的結果"""

        successful_tasks = {name: r for name, r in task_results.items() if 'error' not in r and r.get('num_samples', 0) > 0}

        if not successful_tasks:
            logger.error("No tasks completed successfully. Cannot aggregate overall results.")
            return {"error": "No tasks completed successfully"}

        num_successful = len(successful_tasks)
        num_total = len(task_results)

        # 平均總體品質得分
        avg_overall_score = np.mean([r['overall_score'] for r in successful_tasks.values()])

        # !!! 平均總體量化與性能指標 !!!
        overall_quant_perf = {}
        # 找出所有成功任務報告的聚合指標鍵名
        all_agg_stat_keys = set(k for r in successful_tasks.values() for k in r.get('quantization_performance', {}).keys())

        for key in all_agg_stat_keys:
             # 計算所有成功任務中該指標的平均值
             values = [r['quantization_performance'].get(key) for r in successful_tasks.values() if r.get('quantization_performance') and r['quantization_performance'].get(key) is not None]
             if values and isinstance(values[0], (int, float)): # Only average numeric values
                 overall_quant_perf[f'overall_{key}'] = np.mean(values)


        return {
            'num_successful_tasks': num_successful,
            'num_total_tasks': num_total,
            'overall_quality_score': avg_overall_score,
            'overall_quantization_performance': overall_quant_perf, # 使用新的 key 名稱
            'task_breakdown': { # Keep task breakdown
                task: {'score': result.get('overall_score', 0)}
                for task, result in successful_tasks.items()
            }
        }