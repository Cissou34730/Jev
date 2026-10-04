"""Partially fine-tune MiniLM for urgency classification on CPU.

Only the final transformer layer and classification head are trainable. Synthetic augmentation is
training-only; evaluation uses original tickets grouped by unseen urgency formulations.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from train_minilm_heads import MODEL_ID, grouped_holdout, normalized_sentences, urgency_template_keys


def set_low_priority() -> None:
    if os.name == "nt":
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, 0x00004000)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(tokenizer: Any, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def loader_for(
    encoded: dict[str, torch.Tensor],
    labels: list[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    names = [name for name in ("input_ids", "attention_mask", "token_type_ids") if name in encoded]
    tensors = [encoded[name] for name in names] + [torch.tensor(labels, dtype=torch.long)]
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)
    loader.input_names = names  # type: ignore[attr-defined]
    return loader


def unpack_batch(loader: DataLoader, batch: tuple[torch.Tensor, ...]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    names = loader.input_names  # type: ignore[attr-defined]
    return dict(zip(names, batch[:-1])), batch[-1]


def evaluate(model: torch.nn.Module, loader: DataLoader, classes: list[str]) -> dict[str, Any]:
    model.eval()
    expected: list[int] = []
    predicted: list[int] = []
    confidences: list[float] = []
    losses: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            inputs, labels = unpack_batch(loader, batch)
            output = model(**inputs, labels=labels)
            probabilities = output.logits.softmax(dim=1)
            losses.append(float(output.loss))
            expected.extend(labels.tolist())
            predicted.extend(probabilities.argmax(dim=1).tolist())
            confidences.extend(probabilities.max(dim=1).values.tolist())
    confusion = Counter((classes[a], classes[p]) for a, p in zip(expected, predicted))
    correct = sum(a == p for a, p in zip(expected, predicted))
    return {
        "tickets": len(expected),
        "correct": correct,
        "accuracy": round(correct / len(expected), 4),
        "loss": round(sum(losses) / len(losses), 4),
        "mean_confidence": round(sum(confidences) / len(confidences), 4),
        "confusion": {
            f"{actual}->{prediction}": count
            for (actual, prediction), count in sorted(confusion.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="partially fine-tune MiniLM urgency on CPU")
    parser.add_argument("--tickets", default="tickets-1000.json")
    parser.add_argument("--augmentation", default="tickets-urgence-training-augmentation.json")
    parser.add_argument("--model-dir", default=str(Path(".laya-cache") / "models" / "minilm-train"))
    parser.add_argument("--out-dir", default=str(Path(".laya-cache") / "checkpoints" / "minilm-urgency-partial"))
    parser.add_argument("--metrics", default="resultats-minilm-urgence-partial.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--holdout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    set_low_priority()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    seed_everything(args.seed)

    data = json.loads(Path(args.tickets).read_text(encoding="utf-8"))
    states = [ticket["state"] for ticket in data["tickets"]]
    labels = [str(ticket["expected"]["urgence"]) for ticket in data["tickets"]]
    classes = sorted(set(labels))
    label_to_id = {label: index for index, label in enumerate(classes)}
    groups = urgency_template_keys(states, labels)
    train_indices, test_indices = grouped_holdout(labels, groups, args.holdout, args.seed)
    train_groups = {groups[index] for index in train_indices}
    test_groups = {groups[index] for index in test_indices}
    if train_groups & test_groups:
        raise RuntimeError("urgency formulation leakage between training and test")

    train_states = [states[index] for index in train_indices]
    train_labels = [labels[index] for index in train_indices]
    test_states = [states[index] for index in test_indices]
    test_labels = [labels[index] for index in test_indices]

    augmentation_count = 0
    augmentation_path = Path(args.augmentation)
    if augmentation_path.is_file():
        augmentation = json.loads(augmentation_path.read_text(encoding="utf-8"))["tickets"]
        augmentation_phrases = {normalized_sentences(ticket["state"])[-1] for ticket in augmentation}
        if augmentation_phrases & test_groups:
            raise RuntimeError("augmentation duplicates a held-out urgency formulation")
        train_states.extend(ticket["state"] for ticket in augmentation)
        train_labels.extend(str(ticket["expected"]["urgence"]) for ticket in augmentation)
        augmentation_count = len(augmentation)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.config.id2label = {index: label for index, label in enumerate(classes)}
    model.config.label2id = label_to_id
    model.classifier.apply(model._init_weights)

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.roberta.encoder.layer[-1].parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    train_encoded = encode(tokenizer, train_states, args.max_length)
    test_encoded = encode(tokenizer, test_states, args.max_length)
    train_loader = loader_for(
        train_encoded, [label_to_id[label] for label in train_labels], args.batch_size, True
    )
    test_loader = loader_for(
        test_encoded, [label_to_id[label] for label in test_labels], args.batch_size * 2, False
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, math.floor(total_steps * 0.10)),
        num_training_steps=total_steps,
    )

    started = time.perf_counter()
    epoch_history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            inputs, batch_labels = unpack_batch(train_loader, batch)
            optimizer.zero_grad(set_to_none=True)
            output = model(**inputs, labels=batch_labels)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(output.loss.detach())
            if step % 25 == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={running_loss / step:.4f}",
                    flush=True,
                )
        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(running_loss / len(train_loader), 4),
            "seconds": round(time.perf_counter() - epoch_started, 1),
        })
        model.train()

    test_metrics = evaluate(model, test_loader, classes)
    elapsed = time.perf_counter() - started
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    report = {
        "model": MODEL_ID,
        "task": "urgence",
        "method": "last transformer layer and classification head fine-tuned on CPU",
        "device": "CPU",
        "cpu_threads": args.threads,
        "process_priority": "below normal",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "train_tickets_original": len(train_indices),
        "train_tickets_augmentation": augmentation_count,
        "test_tickets": len(test_indices),
        "train_formulation_groups": len(train_groups),
        "test_formulation_groups": len(test_groups),
        "epoch_history": epoch_history,
        "elapsed_seconds": round(elapsed, 1),
        "held_out": test_metrics,
        "baselines": {
            "minilm_zero_shot": 0.497,
            "minilm_frozen_head": 0.5566,
            "mdeberta_zero_shot": 0.684,
        },
        "checkpoint": str(output_dir),
    }
    Path(args.metrics).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
