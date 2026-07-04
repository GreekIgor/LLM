#!/usr/bin/env python3
"""
Комплексный бенчмаркинг скрипт для vLLM моделей
Тестирует модели на Q&A задачах и логирует результаты в MLflow

Основано на: https://github.com/Ilia2704/llm_mlflow/blob/main/scripts/benchmark_models.py
"""

import os
import json
import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

import mlflow
import requests

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===== Evaluation Dataset =====
EVAL_SAMPLES: List[Dict[str, str]] = [
    {
        "context": "Kubernetes is an open-source system for automating deployment, scaling, and management of containerized applications.",
        "question": "What does Kubernetes automate?",
        "answer": "deployment, scaling, and management of containerized applications",
    },
    {
        "context": "vLLM is a high-throughput and memory-efficient inference and serving engine for large language models.",
        "question": "What is vLLM designed for?",
        "answer": "inference and serving engine for large language models",
    },
    {
        "context": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle.",
        "question": "What does MLflow manage?",
        "answer": "the end-to-end machine learning lifecycle",
    },
    {
        "context": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
        "question": "What is Python known for?",
        "answer": "simplicity and readability",
    },
    {
        "context": "Docker enables developers to package applications into containers—standardized executable components.",
        "question": "What does Docker enable?",
        "answer": "package applications into containers",
    },
]


# ===== Configuration =====
@dataclass
class BenchmarkConfig:
    """Конфигурация бенчмаркинга"""
    vllm_base_url: str
    mlflow_tracking_uri: str
    mlflow_experiment: str
    models: List[str]
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 60


# ===== Evaluation Metrics =====
def normalize_text(s: str) -> List[str]:
    """Нормализация текста для метрик"""
    return s.strip().lower().replace("\n", " ").split()


def token_f1(pred: str, ref: str) -> float:
    """Вычисление token-level F1 score"""
    p = normalize_text(pred)
    r = normalize_text(ref)
    
    if not p or not r:
        return 0.0
    
    overlap = 0
    r_counts = {}
    for t in r:
        r_counts[t] = r_counts.get(t, 0) + 1
    
    for t in p:
        if r_counts.get(t, 0) > 0:
            overlap += 1
            r_counts[t] -= 1
    
    precision = overlap / len(p) if p else 0
    recall = overlap / len(r) if r else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    """Exact match metric"""
    return 1.0 if normalize_text(pred) == normalize_text(ref) else 0.0


# ===== vLLM Query Functions =====
def build_prompt(context: str, question: str) -> str:
    """Построение промпта для Q&A"""
    return (
        "You are a helpful assistant. Answer succinctly based only on the context.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )


def query_vllm(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    timeout: int = 60
) -> Dict[str, Any]:
    """Запрос к vLLM API (или любому OpenAI-совместимому бэкенду, напр. OpenRouter)"""
    try:
        # Опциональная авторизация: для локального vLLM ключ не нужен,
        # для облачных бэкендов (OpenRouter/OpenAI) берётся из VLLM_API_KEY.
        headers = {}
        api_key = os.getenv("VLLM_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            headers=headers,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error querying vLLM: {e}")
        raise


def extract_answer(text: str) -> str:
    """Извлечение ответа из сгенерированного текста"""
    # Эвристика: берём текст после последнего "Answer:"
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    
    # Очистка
    text = text.strip()
    
    # Берём первое предложение если есть точка
    if "." in text:
        text = text.split(".")[0]
    
    return text.strip()


# ===== Benchmark Functions =====
def evaluate_model(config: BenchmarkConfig, model_id: str) -> Dict[str, Any]:
    """Оценка одной модели на датасете"""
    logger.info(f"Evaluating model: {model_id}")
    
    predictions = []
    f1_scores = []
    em_scores = []
    latencies = []
    
    for i, sample in enumerate(EVAL_SAMPLES):
        logger.info(f"  Processing sample {i+1}/{len(EVAL_SAMPLES)}")
        
        prompt = build_prompt(sample["context"], sample["question"])
        
        try:
            # Запрос к модели
            start_time = time.perf_counter()
            result = query_vllm(
                config.vllm_base_url,
                model_id,
                prompt,
                config.max_tokens,
                config.temperature,
                config.top_p,
                config.timeout
            )
            latency = time.perf_counter() - start_time
            
            # Извлечение ответа
            raw_output = result["choices"][0]["text"]
            predicted = extract_answer(raw_output)
            reference = sample["answer"]
            
            # Вычисление метрик
            f1 = token_f1(predicted, reference)
            em = exact_match(predicted, reference)
            
            f1_scores.append(f1)
            em_scores.append(em)
            latencies.append(latency)
            
            predictions.append({
                "question": sample["question"],
                "context": sample["context"][:100] + "...",  # Truncate for readability
                "reference": reference,
                "predicted": predicted,
                "f1": f1,
                "em": em,
                "latency": latency,
            })
            
        except Exception as e:
            logger.error(f"Error processing sample {i+1}: {e}")
            predictions.append({
                "question": sample["question"],
                "error": str(e),
            })
    
    # Агрегированные метрики
    metrics = {
        "f1_mean": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "f1_std": (sum((x - sum(f1_scores)/len(f1_scores))**2 for x in f1_scores) / len(f1_scores))**0.5 if f1_scores else 0.0,
        "em_mean": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "latency_mean": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
        "samples_evaluated": len(predictions),
        "samples_successful": len([p for p in predictions if "error" not in p]),
    }
    
    return {
        "model_id": model_id,
        "metrics": metrics,
        "predictions": predictions,
    }


def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """Запуск полного бенчмарка для всех моделей"""
    logger.info("="*70)
    logger.info("Starting vLLM Model Benchmark")
    logger.info("="*70)
    logger.info(f"vLLM URL: {config.vllm_base_url}")
    logger.info(f"MLflow URI: {config.mlflow_tracking_uri}")
    logger.info(f"Models: {', '.join(config.models)}")
    logger.info(f"Evaluation samples: {len(EVAL_SAMPLES)}")
    logger.info("="*70)
    
    # Настройка MLflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)
    
    all_results = []
    best_model = None
    best_score = -1.0
    
    for model_id in config.models:
        logger.info(f"\n{'='*70}")
        logger.info(f"Model: {model_id}")
        logger.info(f"{'='*70}")
        
        with mlflow.start_run(run_name=model_id):
            # Логирование параметров
            mlflow.log_params({
                "model_id": model_id,
                "vllm_base_url": config.vllm_base_url,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_samples": len(EVAL_SAMPLES),
            })
            
            # Оценка модели
            results = evaluate_model(config, model_id)
            all_results.append(results)
            
            metrics = results["metrics"]
            
            # Логирование метрик
            mlflow.log_metrics(metrics)
            
            # Сохранение предсказаний как артефакт
            predictions_file = f"predictions_{model_id.replace('/', '_')}.json"
            with open(predictions_file, "w") as f:
                json.dump(results["predictions"], f, indent=2)
            mlflow.log_artifact(predictions_file)
            os.remove(predictions_file)
            
            # Вывод результатов
            logger.info(f"\n📊 Results for {model_id}:")
            logger.info(f"  F1 Score: {metrics['f1_mean']:.4f} (±{metrics['f1_std']:.4f})")
            logger.info(f"  Exact Match: {metrics['em_mean']:.4f}")
            logger.info(f"  Latency (mean): {metrics['latency_mean']:.3f}s")
            logger.info(f"  Latency (p95): {metrics['latency_p95']:.3f}s")
            logger.info(f"  Success rate: {metrics['samples_successful']}/{metrics['samples_evaluated']}")
            
            # Отслеживание лучшей модели
            if metrics["f1_mean"] > best_score:
                best_score = metrics["f1_mean"]
                best_model = model_id
    
    # Сохранение сводки
    summary = {
        "best_model": best_model,
        "best_score": best_score,
        "all_results": all_results,
        "config": {
            "vllm_base_url": config.vllm_base_url,
            "models": config.models,
            "num_samples": len(EVAL_SAMPLES),
        }
    }
    
    with open("benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info("🎉 Benchmark Completed!")
    logger.info(f"{'='*70}")
    logger.info(f"🏆 Best Model: {best_model}")
    logger.info(f"📈 Best F1 Score: {best_score:.4f}")
    logger.info(f"💾 Summary saved to: benchmark_summary.json")
    logger.info(f"📊 MLflow UI: {config.mlflow_tracking_uri}")
    logger.info(f"{'='*70}\n")
    
    # Сохранение имени лучшей модели
    with open("best_model.txt", "w") as f:
        f.write(best_model)
    logger.info("💡 Best model saved to: best_model.txt")
    
    return summary


def main():
    """Главная функция"""
    # Конфигурация из переменных окружения
    config = BenchmarkConfig(
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        mlflow_experiment=os.getenv("MLFLOW_EXPERIMENT", "vllm-model-comparison"),
        models=os.getenv("MODELS", "facebook/opt-1.3b").split(","),
        max_tokens=int(os.getenv("MAX_TOKENS", "64")),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        top_p=float(os.getenv("TOP_P", "0.9")),
        timeout=int(os.getenv("TIMEOUT", "60")),
    )
    
    # Проверка доступности vLLM
    try:
        response = requests.get(f"{config.vllm_base_url}/health", timeout=5)
        response.raise_for_status()
        logger.info("✅ vLLM server is accessible")
    except Exception as e:
        logger.error(f"❌ Cannot connect to vLLM server: {e}")
        logger.error(f"   Make sure vLLM is running at {config.vllm_base_url}")
        return
    
    # Запуск бенчмарка
    run_benchmark(config)


if __name__ == "__main__":
    main()
