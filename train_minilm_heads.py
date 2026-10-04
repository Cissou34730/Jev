"""Train lightweight ticket classifiers on frozen multilingual MiniLM embeddings.

The encoder is never updated. Tickets are embedded once on CPU, then one linear head is trained
per decision. Evaluation groups repeated synthetic formulations so equivalent templates cannot
appear in both training and test sets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
OPENINGS = (
    "bonjour au support",
    "bonjour l équipe",
    "madame monsieur",
    "je vous contacte car",
    "bonjour",
)


def normalized_sentences(text: str) -> list[str]:
    result = []
    for part in re.split(r"[.!?\n]+", text.lower()):
        part = re.sub(r"[a-z]{1,4}-\d+", " ", part)
        part = re.sub(r"\d+", " ", part)
        part = re.sub(r"[^a-zà-ÿ ]+", " ", part)
        part = " ".join(part.split())
        if not part or part.startswith("référence tkt"):
            continue
        for opening in OPENINGS:
            if part.startswith(opening):
                part = part[len(opening):].strip()
                break
        if part:
            result.append(part)
    return result


def problem_template_keys(states: list[str], frequent_share: float = 0.10) -> list[str]:
    counts = Counter(sentence for state in states for sentence in set(normalized_sentences(state)))
    threshold = max(3, round(frequent_share * len(states)))
    boilerplate = {sentence for sentence, count in counts.items() if count >= threshold}
    keys = []
    for state in states:
        kept = [sentence for sentence in normalized_sentences(state) if sentence not in boilerplate]
        keys.append(" | ".join(kept) or "<empty-problem-template>")
    return keys


def urgency_template_keys(states: list[str], labels: list[str]) -> list[str]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for state, label in zip(states, labels):
        for sentence in set(normalized_sentences(state)):
            stats[sentence][label] += 1

    keys = []
    for state, label in zip(states, labels):
        candidates = []
        for sentence in normalized_sentences(state):
            counts = stats[sentence]
            total = sum(counts.values())
            purity = counts[label] / total
            if total >= 20 and purity >= 0.98:
                candidates.append((total, sentence))
        if not candidates:
            raise RuntimeError(f"cannot identify urgency formulation in: {state}")
        keys.append(max(candidates)[1])
    return keys


def grouped_holdout(labels: list[str], groups: list[str], holdout: float, seed: int) -> tuple[list[int], list[int]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (label, group) in enumerate(zip(labels, groups)):
        grouped[(label, group)].append(index)

    rng = random.Random(seed)
    train: list[int] = []
    test: list[int] = []
    classes = sorted(set(labels))
    for label in classes:
        label_groups = [(group, indices) for (group_label, group), indices in grouped.items() if group_label == label]
        if len(label_groups) < 2:
            raise RuntimeError(f"class {label!r} has fewer than two independent formulation groups")
        rng.shuffle(label_groups)
        target = max(1, round(sum(len(indices) for _, indices in label_groups) * holdout))
        held = 0
        for position, (_, indices) in enumerate(label_groups):
            remaining = len(label_groups) - position
            if held < target and remaining > 1:
                test.extend(indices)
                held += len(indices)
            else:
                train.extend(indices)
    return sorted(train), sorted(test)


def corpus_hash(ids: list[str], states: list[str]) -> str:
    digest = hashlib.sha256()
    for ticket_id, state in zip(ids, states):
        digest.update(ticket_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(state.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def encode_tickets(
    model_dir: Path,
    states: list[str],
    ids: list[str],
    cache_path: Path,
    batch_size: int,
    max_length: int,
    pooling: str,
) -> np.ndarray:
    expected_hash = hashlib.sha256(
        f"{MODEL_ID}|{max_length}|{pooling}|{corpus_hash(ids, states)}".encode("utf-8")
    ).hexdigest()
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["corpus_hash"].item()) == expected_hash:
            return cached["embeddings"].astype(np.float32)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=model_dir,
        allow_patterns=(
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "sentencepiece.bpe.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).eval()
    batches = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            encoded = tokenizer(
                states[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            if pooling == "cls":
                pooled = hidden[:, 0]
            else:
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            batches.append(pooled.cpu().numpy().astype(np.float32))
            print(f"encoded {min(start + batch_size, len(states))}/{len(states)}", flush=True)
    embeddings = np.concatenate(batches)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings, corpus_hash=np.array(expected_hash))
    print(f"embedding pass: {time.perf_counter() - started:.1f}s", flush=True)
    return embeddings


def train_head(
    embeddings: np.ndarray,
    labels: list[str],
    train_indices: list[int],
    test_indices: list[int],
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[str]]:
    classes = sorted(set(labels))
    label_to_id = {label: index for index, label in enumerate(classes)}
    x = torch.from_numpy(embeddings)
    y = torch.tensor([label_to_id[label] for label in labels], dtype=torch.long)
    train_idx = torch.tensor(train_indices, dtype=torch.long)
    test_idx = torch.tensor(test_indices, dtype=torch.long)

    torch.manual_seed(seed)
    head = torch.nn.Linear(x.shape[1], len(classes))
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(x[train_idx]), y[train_idx])
        loss.backward()
        optimizer.step()

    head.eval()
    with torch.inference_mode():
        logits = head(x[test_idx])
        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)
    expected = y[test_idx]
    confusion = Counter(
        (classes[int(actual)], classes[int(predicted)])
        for actual, predicted in zip(expected, predictions)
    )
    accuracy = float((predictions == expected).float().mean())
    confidence = probabilities.max(dim=1).values
    metrics = {
        "train_tickets": len(train_indices),
        "test_tickets": len(test_indices),
        "classes": classes,
        "correct": int((predictions == expected).sum()),
        "accuracy": round(accuracy, 4),
        "mean_confidence": round(float(confidence.mean()), 4),
        "confusion": {
            f"{actual}->{predicted}": count
            for (actual, predicted), count in sorted(confusion.items())
        },
    }
    return (
        metrics,
        head.weight.detach().numpy().astype(np.float32),
        head.bias.detach().numpy().astype(np.float32),
        classes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="train CPU ticket heads on frozen MiniLM embeddings")
    parser.add_argument("--tickets", default="tickets-1000.json")
    parser.add_argument("--model-dir", default=str(Path(".laya-cache") / "models" / "minilm-train"))
    parser.add_argument("--cache", default=str(Path(".laya-cache") / "features" / "minilm-tickets.npz"))
    parser.add_argument("--out-dir", default=str(Path(".laya-cache") / "checkpoints" / "minilm-ticket-heads"))
    parser.add_argument("--metrics", default="resultats-minilm-specialise.json")
    parser.add_argument(
        "--urgency-augmentation",
        default="",
        help="training-only urgency examples; an absent file disables augmentation",
    )
    parser.add_argument("--holdout", type=float, default=0.20)
    parser.add_argument("--pooling", choices=("cls", "mean"), default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()

    data = json.loads(Path(args.tickets).read_text(encoding="utf-8"))
    tickets = data["tickets"]
    ids = [ticket["id"] for ticket in tickets]
    states = [ticket["state"] for ticket in tickets]
    embeddings = encode_tickets(
        Path(args.model_dir), states, ids, Path(args.cache), args.batch_size, args.max_length,
        args.pooling
    )

    problem_keys = problem_template_keys(states)
    augmentation_path = Path(args.urgency_augmentation) if args.urgency_augmentation else None
    augmentation_states: list[str] = []
    augmentation_labels: list[str] = []
    augmentation_embeddings: np.ndarray | None = None
    if augmentation_path is not None and augmentation_path.is_file():
        augmentation = json.loads(augmentation_path.read_text(encoding="utf-8"))["tickets"]
        augmentation_ids = [ticket["id"] for ticket in augmentation]
        augmentation_states = [ticket["state"] for ticket in augmentation]
        augmentation_labels = [str(ticket["expected"]["urgence"]) for ticket in augmentation]
        augmentation_cache = Path(args.cache).with_name(
            Path(args.cache).stem + "-urgency-augmentation.npz"
        )
        augmentation_embeddings = encode_tickets(
            Path(args.model_dir), augmentation_states, augmentation_ids, augmentation_cache,
            args.batch_size, args.max_length, args.pooling
        )

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "model": MODEL_ID,
        "training": f"frozen {args.pooling}-pooled encoder with one linear head per question",
        "tickets": len(tickets),
        "seed": args.seed,
        "holdout": args.holdout,
        "questions": {},
    }

    for offset, question_id in enumerate(("service", "remboursement", "urgence")):
        labels = [str(ticket["expected"][question_id]) for ticket in tickets]
        groups = urgency_template_keys(states, labels) if question_id == "urgence" else problem_keys
        train_indices, test_indices = grouped_holdout(labels, groups, args.holdout, args.seed + offset)
        if set(groups[index] for index in train_indices) & set(groups[index] for index in test_indices):
            raise RuntimeError(f"group leakage detected for {question_id}")
        head_embeddings = embeddings
        head_labels = labels
        if question_id == "urgence" and augmentation_embeddings is not None:
            original_count = len(head_labels)
            head_embeddings = np.concatenate((embeddings, augmentation_embeddings))
            head_labels = labels + augmentation_labels
            train_indices = train_indices + list(
                range(original_count, original_count + len(augmentation_labels))
            )
        metrics, weight, bias, classes = train_head(
            head_embeddings,
            head_labels,
            train_indices,
            test_indices,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.seed + offset,
        )
        original_train_indices = [index for index in train_indices if index < len(groups)]
        metrics["train_groups"] = len(set(groups[index] for index in original_train_indices))
        if question_id == "urgence":
            metrics["augmentation_tickets"] = len(augmentation_labels)
        metrics["test_groups"] = len(set(groups[index] for index in test_indices))
        result["questions"][question_id] = metrics
        np.savez_compressed(
            output_dir / f"{question_id}.npz",
            weight=weight,
            bias=bias,
            classes=np.array(classes),
        )
        print(f"{question_id}: {metrics['accuracy']:.2%} on {metrics['test_tickets']} held-out tickets")

    total = sum(question["test_tickets"] for question in result["questions"].values())
    correct = sum(question["correct"] for question in result["questions"].values())
    result["overall"] = {
        "test_decisions": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
    }
    result["artifact_dir"] = str(output_dir)
    Path(args.metrics).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

