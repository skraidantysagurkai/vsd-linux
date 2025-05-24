import gc
import json
import random as rd
import re
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from paths import DATA_DIR
from src.shared.json_tools import load_json_long
from math import ceil
from tqdm import tqdm


def load_data(malicious_path: Path, benign_path: Path):
    """
    Load malicious and benign log data from JSON files.
    """
    malicious_data = [load_json_long(file) for file in malicious_path.glob("*.json")]
    benign_data = [load_json_long(file) for file in benign_path.glob("*.json")]
    return malicious_data, benign_data


def build_log_line(log: dict):
    # Convert timestamp to readable date (UTC)
    try:
        readable_time = datetime.fromtimestamp(float(log.get("timestamp", 0))).strftime('%Y-%m-%d %H:%M:%S UTC')
    except Exception:
        readable_time = "Invalid timestamp"

    # Format command and arguments
    command = log.get("command", "<no command>")
    args = log.get("arguments", [])
    if isinstance(args, list):
        full_command = ' '.join(args)
    elif args is None:
        full_command = command
    else:
        full_command = str(args)

    # Format other fields with labels
    fields = [
        f"Time: {readable_time}",
        f"Success: {'Yes' if log.get('success') == 1 else 'No'}",
        f"UID: {log.get('uid', '?')} | EUID: {log.get('euid', '?')}",
        f"Syscall: {log.get('syscall', '?')}",
        f"PID: {log.get('pid', '?')} | PPID: {log.get('ppid', '?')}",
        f"CWD: {log.get('CWD', '[unknown]')}",
        f"Command: {full_command}"
    ]

    return " ".join(fields).strip()


def build_prompt(data: list):
    log_history = "\n".join([build_log_line(log) for log in data]).strip()
    return (
        f"""
        You are a security analyst reviewing system command logs. 
        Given the following log history, classify it as either 'malicious' or 'benign', and explain the reasoning behind your classification.
        
        Log history:
        {log_history}
        
        Respond in the following format:
        
        ###Classification: <malicious or benign>  
        ###Reason: <brief explanation>
        """)


def parse_answer(answer: str):
    """
    Parse the model's answer to extract classification and reasoning.
    """
    try:
        classification = answer.split("###Classification:")[-1].strip().split("\n")[0].lower()
        binary_class = 1 if "malicious" in classification else 0
        reasoning = answer.split("###Reason:")[-1].strip()
    except Exception as e:
        print(f"Error parsing answer: {e}")
        classification = "unknown"
        reasoning = "unknown"
        binary_class = None
    return classification, binary_class, reasoning


def score_model_one_shot(pred: list, true: list):
    """
    Evaluate a model's predictions against true labels.

    Args:
        pred (list): Predicted labels from the model (e.g., ['malicious', 'benign', ...])
        true (list): Ground-truth labels (same format)

    Returns:
        dict: Dictionary of scores + prints classification report
    """

    # Metrics
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred, zero_division=0)
    recall = recall_score(true, pred, zero_division=0)
    f1 = f1_score(true, pred, zero_division=0)
    cm = confusion_matrix(true, pred)

    print("Classification Report:\n", classification_report(true, pred))
    print("Confusion Matrix (malicious/benign):\n", cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_metrics(pred: list, true: list, model_name: str, metrics: dict, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    safe_model_name = re.sub(r'[^\w\-_.]', '_', model_name)

    result = {
        "model_name": model_name,
        "predictions": pred,
        "true_labels": true,
        "metrics": metrics
    }
    with open(str(output_path / f"{safe_model_name}_evaluation.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)


def classify_logs_batched(pipe, log_data, batch_size=25):
    model_results = []
    total = len(log_data)
    num_batches = ceil(total / batch_size)

    for i in tqdm(range(num_batches), desc="Classifying logs"):
        try:
            batch = log_data[i * batch_size:(i + 1) * batch_size]
            responses = pipe(
                batch,
                max_new_tokens=100,
                do_sample=False,
                top_p=1.0,
            )

            for r in responses:
                _, parsed, _ = parse_answer(r[0]['generated_text'])
                model_results.append(parsed)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            model_results.extend([None] * batch_size)

    print()  # Final newline after last update
    return model_results


def classify_logs(model_name: str, log_data: List[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"--- Testing model: {model_name} ---")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    now = datetime.now()
    model_results = classify_logs_batched(pipe, log_data, batch_size=10)
    time_taken = datetime.now() - now

    del model
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    return model_results, time_taken


if __name__ == "__main__":
    output_path = Path("models/llm_selection_on_logs")
    output_path.mkdir(parents=True, exist_ok=True)

    malicious_data, benign_data = load_data(DATA_DIR / "testing_data/malicious", DATA_DIR / "testing_data/benign")
    malicious_data = [(i, 1) for i in malicious_data][:15]
    benign_data = [(i, 0) for i in benign_data][:15]

    rd.shuffle(malicious_data)
    rd.shuffle(benign_data)

    all_data = malicious_data + benign_data
    rd.shuffle(all_data)

    targets = [i[1] for i in all_data]
    logs = [build_prompt(i[0]) for i in all_data]

    models = [
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct-1M",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mamba-Codestral-7B-v0.1",
        "bigcode/starcoder2-3b",
        "meta-llama/Llama-3.1-8B-Instruct"
    ]


    for model_name in models:
        try:
            local_targets = targets.copy()
            results, time_taken = classify_logs(model_name, logs)

            none_idx = [i for i, j in enumerate(results) if j is None]
            local_targets = [item for i, item in enumerate(local_targets) if i not in none_idx]
            results = [item for i, item in enumerate(results) if i not in none_idx]


            metrics = score_model_one_shot(results, local_targets)
            metrics["time_taken"] = str(time_taken)
            save_metrics(results, local_targets, model_name, metrics, output_path)
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            continue
    print("All models evaluated and results saved.")
