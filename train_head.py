"""Retrain Laya's decision head for one question, with the encoder frozen.

Why only the head: on the multilingual checkpoint every question whose options are not a plain
yes/no collapsed onto one option, and the collapse survived a renamed label, a replaced criterion
description and a change of question type. That is a property of the decision path -- `type_emb`,
the `head` layers and the `scorer` -- not of the encoder, which the same tests showed does carry
usable signal (the facturation/livraison duel is right 77% of the time).

Why it fits on a CPU: the encoder is frozen, so its output for a given (state, question) never
changes. It is computed once into a cache and the training loop never runs it again. The head is
a couple of transformer layers over frozen vectors.

The head pass below is copied from `laya.common.DecisionModel.forward` rather than reaching into
it, so that the trained head is the head that runs at inference. `check_parity` asserts that: it
compares the copy against the checkpoint's own `forward` on the same cache before any training
starts, and refuses to continue if they disagree.

Prerequisites: `laya` ships the model, the objective (`proper_reward`) and the frozen-encoder
switch (`detach_encoder`), but no training loop. This is that loop.

    uv run python train_head.py --question service
    uv run python train_head.py --question service --epochs 30 --lr 5e-4
    uv run python train_head.py --question service,remboursement --out .laya-cache/checkpoints/laya-head

The first run downloads the PyTorch checkpoint (`convaiinnovations/laya-multilingual`); only the
ONNX export lives in `.laya-cache/checkpoints` today.

Then export the fine-tuned checkpoint to ONNX and point the server at it:

    uv run python export_laya_master.py --repo .laya-cache/checkpoints/laya-head --out .laya-cache/checkpoints/laya-head
    $env:LAYA_OV_ONNX_PATH = ".laya-cache\checkpoints\laya-head\laya.onnx"

Two things to know before trusting a number this prints:

  * the split is by template, not by ticket. The tickets come from a generator with a couple of
    dozen phrasings per class, so a ticket-level split puts the same phrasing on both sides and
    the held-out score measures nothing. `--seed` picks the grouping and is printed. The exact
    held-out tickets are always written to `<out>/heldout-tickets.json`; `--split-only` can
    recreate that file without loading the model or computing the encoder cache.
  * `--label-bias none` when you serve the fine-tuned model. Training happens on raw logits; a
    bias applied at inference afterwards is a correction fitted on the old model.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

import laya
from laya.agent import Agent
from laya.common import QTYPES, build_sequence, proper_reward

MAX_MARKERS = 5

# The generator's five openings, read off the corpus. They are glued to the problem statement in
# one sentence ("bonjour je souhaite corriger l adresse..."), so they survive the sentence-level
# pass below and have to come off as prefixes. Longest first: "bonjour au support" would be cut
# down to "au support" if "bonjour" were tried before it.
OPENINGS = ("bonjour au support", "bonjour l équipe", "madame monsieur", "je vous contacte car",
            "bonjour")


# -- inputs, built exactly as inference builds them -------------------------------------------

def option_keys(q: Dict[str, Any]) -> List[str]:
    """The option names in rendered order, the same list `render_options` walks."""
    if q["t"] == "choice":
        return [str(key) for key in q["crit"]]
    if q["t"] == "score":
        return [str(i) for i in range(len(q["crit"]))]
    return ["false", "true"]


def target_index(q: Dict[str, Any], expected: Any) -> int:
    """Where the expected answer sits among the options."""
    if q["t"] == "choice":
        return list(q["crit"].keys()).index(expected)
    if q["t"] == "noul":
        return 1 if expected else 0
    for i, criterion in enumerate(q["crit"]):
        # A score level is labelled "Faible : ..." in the ticket and "Faible" is what is expected.
        if str(criterion).split(":")[0].strip() == expected:
            return i
    raise ValueError("expected level %r is not one of %r" % (expected, list(q["crit"])))


def encode_row(agent, state: str, q: Dict[str, Any]) -> Dict[str, Any]:
    """One row: the sequence, the marker positions and the question type."""
    seq, markers = build_sequence(
        agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"],
    )
    if len(markers) > MAX_MARKERS:
        raise ValueError("question has %d options; MAX_MARKERS=%d" % (len(markers), MAX_MARKERS))
    return {"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]}


def collate(rows: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    """The five tensors `DecisionModel.forward` takes.

    `marker_mask` stays a float tensor here and is cast to bool at the call site: the checkpoint's
    own `forward` does `masked_fill(~marker_mask, ...)`, and `~` needs bool, while the ONNX export
    needs float.
    """
    n = len(rows)
    length = max(len(row["ids"]) for row in rows)
    input_ids = torch.full((n, length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, length), dtype=torch.long)
    marker_pos = torch.zeros((n, MAX_MARKERS), dtype=torch.long)
    marker_mask = torch.zeros((n, MAX_MARKERS), dtype=torch.float32)
    for i, row in enumerate(rows):
        input_ids[i, : len(row["ids"])] = torch.tensor(row["ids"])
        attention_mask[i, : len(row["ids"])] = 1
        k = len(row["markers"])
        marker_pos[i, :k] = torch.tensor(row["markers"])
        marker_mask[i, :k] = 1.0
    qtype = torch.tensor([row["qtype"] for row in rows])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "marker_pos": marker_pos,
            "marker_mask": marker_mask, "qtype": qtype}


# -- the head pass ----------------------------------------------------------------------------

def head_forward(model, h, attention_mask, marker_pos, marker_mask, qtype):
    """`DecisionModel.forward` from just after the encoder to the option logits.

    Copied deliberately: replacing `model.encoder` with the cache would drop the encoder out of
    `state_dict()`, and the trained checkpoint has to carry it. `check_parity` holds this copy
    against the real forward.
    """
    h = h + model.type_emb(qtype)[:, None, :]
    pad = ~attention_mask.bool()
    if model.head is not None:
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=pad)
    idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    gathered = torch.gather(h, 1, idx)
    logits = model.scorer(gathered).squeeze(-1).float()
    return logits.masked_fill(~marker_mask, -1e4)


def check_parity(model, batch, cache) -> float:
    """max|Δ| between `head_forward` on the cache and the checkpoint's own forward.

    In eval mode on purpose: with dropout active the two forwards differ by its randomness and
    the check would fail on a copy that is in fact correct.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            reference, _ = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                                 batch["marker_mask"].bool(), batch["qtype"])
            copy = head_forward(model, cache, batch["attention_mask"], batch["marker_pos"],
                                batch["marker_mask"].bool(), batch["qtype"])
    finally:
        model.train(was_training)
    return float((reference - copy).abs().max())


# -- data -------------------------------------------------------------------------------------

def sentences(text: str) -> List[str]:
    """The ticket's sentences, each reduced to letters and spaces.

    References (`CMD-500452`, `TKT-0972`) and every other number go, so two wordings of the same
    sentence collapse to one key instead of looking like two.
    """
    out = []
    for part in re.split(r"[.!?\n]+", text.lower()):
        part = re.sub(r"[a-z]{1,4}-\d+", " ", part)
        part = re.sub(r"\d+", " ", part)
        part = re.sub(r"[^a-zà-ÿ ]+", " ", part)
        part = " ".join(part.split())
        if part:
            out.append(part)
    return out


def boilerplate(states: List[str], share: float = 0.10) -> set:
    """The generator's fixed parts, measured rather than listed.

    A ticket is composed from an opening, a problem statement, an urgency phrasing and a closing
    reference. On the 1000-ticket file that makes 192 distinct whole-ticket keys out of 200
    tickets, so splitting on the whole text separates nothing. What tells them apart is frequency:
    an opening or an urgency phrasing appears in tens of percent of the tickets, while a problem
    statement appears once per template -- a few percent. Anything above `share` is therefore
    decoration, and what is left is the problem statement.

    `share=0.10` sits in the gap: the most common urgency phrasing covers about a quarter of the
    tickets, the rarest problem statement a few percent.
    """
    counts: Dict[str, int] = {}
    for state in states:
        for sentence in set(sentences(state)):
            counts[sentence] = counts.get(sentence, 0) + 1
    threshold = max(3, share * len(states))
    return {sentence for sentence, count in counts.items() if count >= threshold}


def key_of(state: str, boiler: set) -> str:
    """What is left of a ticket once the decoration is gone: its problem statement."""
    kept = []
    for sentence in sentences(state):
        if sentence in boiler:
            continue
        for opening in OPENINGS:
            if sentence.startswith(opening):
                sentence = sentence[len(opening):].strip()
                break
        if sentence:
            kept.append(sentence)
    return " ".join(kept)


def split_by_template(states: List[str], holdout: float, seed: int,
                      share: float = 0.10) -> Tuple[List[int], List[int], set]:
    """Ticket indices split so no problem statement appears on both sides."""
    boiler = boilerplate(states, share)
    groups: Dict[str, List[int]] = {}
    for i, state in enumerate(states):
        groups.setdefault(key_of(state, boiler), []).append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = holdout * len(states)
    train: List[int] = []
    test: List[int] = []
    for key in keys:
        # Fill the smaller side first, so the last group does not land entirely on one side.
        if len(train) <= len(test) or len(test) >= target:
            train.extend(groups[key])
        else:
            test.extend(groups[key])
    return sorted(train), sorted(test), boiler


# -- training ---------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="retrain Laya's head with the encoder frozen")
    ap.add_argument("--checkpoint", default="convaiinnovations/laya-multilingual",
                    help="a Hub id or a checkpoint directory")
    ap.add_argument("--tickets", default="tickets-1000.json",
                    help="a {questions, tickets} file, as the client reads it")
    ap.add_argument("--question", default="service",
                    help="comma-separated question ids to train on. More than one is how the "
                         "shared head keeps what it already does well")
    ap.add_argument("--out", default=os.path.join(".laya-cache", "checkpoints", "laya-head"),
                    help="where the fine-tuned checkpoint is written")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cache-batch", type=int, default=16, help="rows per encoder forward")
    ap.add_argument("--holdout", type=float, default=0.5, help="fraction of tickets held out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-out",
                    help="write the client-compatible held-out dataset here; defaults to "
                         "<out>/heldout-tickets.json")
    ap.add_argument("--split-only", action="store_true",
                    help="write the held-out dataset and exit without loading the model")
    ap.add_argument("--freeze-type-emb", action="store_true",
                    help="leave the shared per-question-type embedding alone")
    ap.add_argument("--save-best", action="store_true",
                    help="write the epoch with the best held-out accuracy instead of the last one. "
                         "The held-out half then also picks the model, so its number is a "
                         "validation number and not a clean test; evaluating every epoch roughly "
                         "doubles the run")
    args = ap.parse_args()

    # -- questions ----------------------------------------------------------------------------
    data = json.load(open(args.tickets, encoding="utf-8"))
    wanted = [name.strip() for name in args.question.split(",") if name.strip()]
    questions: Dict[str, Any] = {}
    internal: Dict[str, Any] = {}
    for name in wanted:
        if name not in data["questions"]:
            raise SystemExit("question %r is not in %s" % (name, args.tickets))
        qdef = data["questions"][name]
        Agent._check_question(name, qdef)
        questions[name] = qdef
        internal[name] = Agent._to_internal(qdef)
    tickets = [t for t in data["tickets"] if all(name in (t.get("expected") or {}) for name in wanted)]
    if not tickets:
        raise SystemExit("no ticket carries an expected value for %s" % ", ".join(wanted))
    print("%d ticket(s), %d question(s): %s" % (len(tickets), len(wanted), ", ".join(wanted)))

    # -- split, before loading the model -------------------------------------------------------
    states = [t["state"] for t in tickets]
    train_rows, test_rows, boiler = split_by_template(states, args.holdout, args.seed)
    keys = {key_of(state, boiler) for state in states}
    print("split by problem statement: %d train ticket(s) / %d held out, %d problem statement(s), "
          "%d boilerplate sentence(s) dropped, seed %d"
          % (len(train_rows), len(test_rows), len(keys), len(boiler), args.seed))
    if len(keys) < 10:
        print("warning: only %d distinct problem statement(s); the split is coarse" % len(keys))

    heldout_out = args.heldout_out or os.path.join(args.out, "heldout-tickets.json")
    heldout_parent = os.path.dirname(os.path.abspath(heldout_out))
    os.makedirs(heldout_parent, exist_ok=True)
    with open(heldout_out, "w", encoding="utf-8") as handle:
        json.dump({"questions": questions, "tickets": [tickets[i] for i in test_rows]},
                  handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("wrote %d held-out ticket(s) to %s" % (len(test_rows), heldout_out))
    if args.split_only:
        return

    device = torch.device("cpu")
    started = time.time()
    agent = laya.load(args.checkpoint, device="cpu")
    model = agent.model.to(device)
    model.train()
    print("loaded %s in %.0fs: max_len %s, head_max_len %s, head_layers %s"
          % (args.checkpoint, time.time() - started, agent.cfg.get("max_len"),
             agent.cfg.get("head_max_len"), len(model.head.layers) if model.head else 0))

    # -- rows, with the encoder output cached once --------------------------------------------
    rows: List[Dict[str, Any]] = []
    for i, ticket in enumerate(tickets):
        for name in wanted:
            row = encode_row(agent, states[i], internal[name])
            row["question"] = name
            row["target"] = target_index(internal[name], ticket["expected"][name])
            rows.append(row)

    # Everything trainable, then the encoder frozen. `requires_grad_` on a module walks its own
    # parameters, so one call each is enough -- and the order matters.
    model.requires_grad_(True)
    model.encoder.requires_grad_(False)
    if args.freeze_type_emb:
        model.type_emb.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("trainable: %d tensor(s), %d parameter(s); encoder frozen"
          % (len(trainable), sum(p.numel() for p in trainable)))

    # The cache is the whole reason this fits on a CPU: one encoder pass per row, ever.
    started = time.time()
    cache: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), args.cache_batch):
            part = rows[start:start + args.cache_batch]
            batch = collate(part, agent.tok.pad_token_id)
            hidden = model.encoder(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"]).last_hidden_state
            for i in range(len(part)):
                # One row at a time: the head attends over the sequence, and every row here has
                # its own length, so the cache cannot share one padded tensor.
                length = int(batch["attention_mask"][i].sum())
                cache.append(hidden[i, :length].detach().clone())
    print("encoder cache: %d row(s) in %.0fs, %.0f MB"
          % (len(cache), time.time() - started,
             sum(t.numel() * t.element_size() for t in cache) / 1e6))
    model.train()

    # -- parity, before anything is trained ----------------------------------------------------
    sample = collate(rows[: min(len(rows), args.batch)], agent.tok.pad_token_id)
    sample_cache = torch.zeros(len(sample["input_ids"]), sample["input_ids"].size(1),
                               cache[0].size(-1))
    for i, row in enumerate(rows[: min(len(rows), args.batch)]):
        sample_cache[i, : cache[i].size(0)] = cache[i]
    delta = check_parity(model, sample, sample_cache)
    print("parity: max|delta| head_forward vs DecisionModel.forward = %.2e" % delta)
    if delta > 1e-3:
        raise SystemExit("the copied head pass does not match the checkpoint's own forward; "
                         "training on it would train something else")

    def one_batch(picks: List[int]):
        """One tensor per batch: the cached hidden states padded, plus target, mask and qtype."""
        item_rows = [rows[i] for i in picks]
        width = max(cache[i].size(0) for i in picks)
        hidden = torch.zeros(len(picks), width, cache[0].size(-1))
        attention = torch.zeros(len(picks), width, dtype=torch.long)
        marker_pos = torch.zeros(len(picks), MAX_MARKERS, dtype=torch.long)
        marker_mask = torch.zeros(len(picks), MAX_MARKERS, dtype=torch.float32)
        target = torch.zeros(len(picks), MAX_MARKERS)
        qtype = torch.zeros(len(picks), dtype=torch.long)
        for j, i in enumerate(picks):
            length = cache[i].size(0)
            hidden[j, :length] = cache[i]
            attention[j, :length] = 1
            markers = item_rows[j]["markers"]
            marker_pos[j, : len(markers)] = torch.tensor(markers)
            marker_mask[j, : len(markers)] = 1.0
            target[j, item_rows[j]["target"]] = 1.0
            qtype[j] = item_rows[j]["qtype"]
        return hidden, attention, marker_pos, marker_mask, target, qtype

    def batches_of(indices: List[int], chunk: int):
        for start in range(0, len(indices), chunk):
            yield one_batch(indices[start:start + chunk])

    def evaluate(indices: List[int]) -> Dict[str, float]:
        model.eval()
        correct = total = duel_right = duel_total = 0
        per_question: Dict[str, List[int]] = {}
        with torch.no_grad():
            for start in range(0, len(indices), 64):
                picks = indices[start:start + 64]
                hidden, attention, marker_pos, marker_mask, _, qtype = one_batch(picks)
                logits = head_forward(model, hidden, attention, marker_pos, marker_mask.bool(), qtype)
                picked = logits.argmax(-1)
                for j, i in enumerate(picks):
                    row = rows[i]
                    truth = row["target"]
                    hit = int(picked[j] == truth)
                    correct += hit
                    total += 1
                    tally = per_question.setdefault(row["question"], [0, 0])
                    tally[0] += hit
                    tally[1] += 1
                    keys = option_keys(internal[row["question"]])
                    if "facturation" in keys and "livraison" in keys:
                        # The duel the label bias buried: facturation against livraison only. Found
                        # by name, not by position, so it survives a reordering of the criteria.
                        left, right = keys.index("facturation"), keys.index("livraison")
                        if truth in (left, right):
                            duel_total += 1
                            duel_right += int((logits[j, left] > logits[j, right]) == (truth == left))
        model.train()
        return {"n": total, "correct": correct,
                "accuracy": round(correct / max(1, total), 4),
                "duel_right": duel_right, "duel_total": duel_total,
                "duel": round(duel_right / max(1, duel_total), 4),
                "by_question": {q: {"n": n, "correct": c, "accuracy": round(c / max(1, n), 4)}
                                for q, (c, n) in sorted(per_question.items())}}

    print("held out, before: %s" % json.dumps(evaluate(test_rows), ensure_ascii=False))

    # -- the loop -----------------------------------------------------------------------------
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    best: Dict[str, Any] = {"accuracy": -1.0}
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_duel = (0.0, 0)
    for epoch in range(1, args.epochs + 1):
        order = list(train_rows)
        random.Random(args.seed + epoch).shuffle(order)
        total_reward = 0.0
        batches = 0
        for hidden, attention, marker_pos, marker_mask, target, qtype in batches_of(order, args.batch):
            logits = head_forward(model, hidden, attention, marker_pos, marker_mask.bool(), qtype)
            probabilities = torch.softmax(logits, -1)
            reward = proper_reward(probabilities, target, qtype, marker_mask)
            loss = -reward.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_reward += float(reward.mean().detach())
            batches += 1
        if args.save_best:
            # Every epoch, because the best one is not necessarily the last: on this checkpoint the
            # duel peaked around the middle of the run while the accuracy wobbled.
            metrics = evaluate(test_rows)
            if metrics["accuracy"] > best["accuracy"]:
                best = dict(metrics, epoch=epoch)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if metrics["duel"] > best_duel[0]:
                best_duel = (metrics["duel"], epoch)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print("epoch %3d  mean reward %+.4f  held out %s"
                  % (epoch, total_reward / max(1, batches), json.dumps(evaluate(test_rows))))
    if best_state is not None:
        model.load_state_dict(best_state)
        print("keeping the best epoch: %s" % json.dumps(best))
    if best_duel[1]:
        print("best duel %.4f at epoch %d" % best_duel)

    print("train, after : %s" % json.dumps(evaluate(train_rows)))
    print("held out, after: %s" % json.dumps(evaluate(test_rows)))

    # -- a checkpoint `laya.load` and `export_laya_master.py` can read -------------------------
    os.makedirs(args.out, exist_ok=True)
    model.eval().to("cpu")
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              os.path.join(args.out, "model.safetensors"))
    with open(os.path.join(args.out, "rl_agent_config.json"), "w", encoding="utf-8") as handle:
        json.dump(agent.cfg, handle, indent=2)
    agent.tok.save_pretrained(os.path.join(args.out, "tokenizer"))
    # `build_model` rebuilds the encoder from this config alone; the weights come from the file
    # above, so no encoder weights are written here.
    model.encoder.config.save_pretrained(os.path.join(args.out, "encoder"))
    print("wrote %s" % args.out)
    print("next: uv run python export_laya_master.py --repo %s --out %s" % (args.out, args.out))


if __name__ == "__main__":
    main()
