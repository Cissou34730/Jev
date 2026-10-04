"""Evaluate an ONNX NLI model on the local ticket schema."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

DEFAULT_MODEL_ID = "onnx-community/multilingual-MiniLMv2-L6-mnli-xnli-ONNX"
DEFAULT_MODEL_DIR = Path(".laya-cache") / "models" / "minilm-nli"


def hypothesis_map(question: dict[str, Any], use_criteria: bool = True) -> list[tuple[str, str]]:
    """Return (expected value, French NLI hypothesis) pairs in decision order."""
    question_type = question["type"]
    if question_type == "choice":
        if not use_criteria:
            return [
                (key, f"Ce ticket concerne le service {key}.")
                for key in question["criteria"]
            ]
        return [
            (key, f"Ce ticket concerne {key} : {description}.")
            for key, description in question["criteria"].items()
        ]
    if question_type == "noul":
        return [
            (False, "La personne ne demande pas explicitement un remboursement."),
            (True, "La personne demande explicitement un remboursement."),
        ]
    if question_type == "score":
        levels = []
        for criterion in question["criteria"]:
            level, description = criterion.split(":", 1)
            hypothesis = f"Le niveau d'urgence du ticket est {level.strip()}."
            if use_criteria:
                hypothesis = f"{hypothesis[:-1]} :{description}."
            levels.append((level.strip(), hypothesis))
        return levels
    raise ValueError(f"unsupported question type: {question_type}")


def find_entailment_index(config: dict[str, Any]) -> int:
    labels = config.get("id2label", {})
    normalized = {int(index): str(label).lower() for index, label in labels.items()}
    for index, label in normalized.items():
        if "entail" in label:
            return index
    raise ValueError(f"model config has no entailment label: {labels}")


def input_feed(compiled: ov.CompiledModel, encoded: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    for port in compiled.inputs:
        name = port.any_name
        candidates = [name]
        for candidate in candidates:
            if candidate in encoded:
                feed[port] = encoded[candidate]
                break
        else:
            raise RuntimeError(f"model input {name} is not provided by tokenizer: {list(encoded)}")
    return feed


def evaluate(
    compiled: ov.CompiledModel,
    tokenizer: Any,
    data: dict[str, Any],
    entailment_index: int,
    max_length: int,
    batch_size: int,
    use_criteria: bool,
) -> tuple[dict[str, Any], float]:
    questions = data["questions"]
    records: list[tuple[str, str, Any, str, str]] = []
    for ticket in data["tickets"]:
        for question_id, question in questions.items():
            for expected, hypothesis in hypothesis_map(question, use_criteria):
                records.append((ticket["id"], question_id, expected, ticket["state"], hypothesis))

    logits_by_ticket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    started = time.perf_counter()
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        premises = [item[3] for item in batch]
        hypotheses = [item[4] for item in batch]
        encoded = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        result = compiled(input_feed(compiled, dict(encoded)))
        output = np.asarray(result[compiled.outputs[0]])
        for row, (ticket_id, question_id, expected, _, _) in zip(output, batch):
            logits_by_ticket[ticket_id][question_id].append(float(row[entailment_index]))
    elapsed = time.perf_counter() - started

    ticket_by_id = {ticket["id"]: ticket for ticket in data["tickets"]}
    metrics: dict[str, Any] = {"tickets": len(data["tickets"]), "questions": {}}
    total = correct = 0
    confusion: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for ticket_id, question_scores in logits_by_ticket.items():
        ticket = ticket_by_id[ticket_id]
        for question_id, scores in question_scores.items():
            options = hypothesis_map(questions[question_id], use_criteria)
            predicted_index = int(np.argmax(scores))
            predicted = options[predicted_index][0]
            expected = ticket["expected"][question_id]
            if questions[question_id]["type"] == "noul":
                expected = bool(expected)
            expected_text = str(expected)
            predicted_text = str(predicted)
            confusion[question_id][(expected_text, predicted_text)] += 1
            total += 1
            correct += int(expected == predicted)

    for question_id, counts in confusion.items():
        question_total = sum(counts.values())
        question_correct = sum(value for (expected, predicted), value in counts.items() if expected == predicted)
        metrics["questions"][question_id] = {
            "total": question_total,
            "correct": question_correct,
            "accuracy": round(question_correct / question_total, 4),
            "confusion": {
                f"{expected}->{predicted}": count
                for (expected, predicted), count in sorted(counts.items())
            },
        }
    metrics["overall"] = {
        "total_decisions": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
    }
    return metrics, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="run an ONNX NLI model with OpenVINO")
    parser.add_argument("--input", default="tickets-200-minilm.json")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", default="CPU", choices=("CPU", "GPU", "AUTO"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--ignore-criteria",
        action="store_true",
        help="use semantic label hypotheses instead of the descriptions in the JSON schema",
    )
    parser.add_argument("--output", default="resultats-minilm-200.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model_id,
        local_dir=model_dir,
        allow_patterns=("config.json", "tokenizer.json", "tokenizer_config.json",
                         "special_tokens_map.json", "spm.model",
                         "added_tokens.json", "onnx/model.onnx"),
    )
    model_path = model_dir / "onnx" / "model.onnx"
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    core = ov.Core()
    compiled = core.compile_model(core.read_model(model_path), args.device)
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    metrics, elapsed = evaluate(
        compiled,
        tokenizer,
        data,
        find_entailment_index(config),
        args.max_length,
        args.batch_size,
        not args.ignore_criteria,
    )
    metrics["model"] = args.model_id
    metrics["device"] = args.device
    metrics["elapsed_seconds"] = round(elapsed, 3)
    metrics["decisions_per_second"] = round(metrics["overall"]["total_decisions"] / elapsed, 2)
    Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
