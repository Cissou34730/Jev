"""Local OpenVINO inference path for laya's typed questions (choice/score/noul).

Mirrors laya_runner.py's stdin/stdout JSON contract exactly, but compiles the reusable master
ONNX export (see export_laya_master.py: from-source export off the official
convaiinnovations/laya-multilingual checkpoint, dynamic n/L axes, fixed 5-wide marker axis,
GPU-safe -- no boolean/u8 tensor in the graph, verified OpenVINO GPU compile across several
(batch, seq_len) profiles) with OpenVINO instead of running the PyTorch model or ONNX Runtime.
Reuses laya's own tokenizer, sequence-building, serialization and answer-decoding (choice/
score/noul) so encoding and decoding match the CPU path exactly.

Two request shapes on stdin, matching jev-test.ps1's two invocation modes:
  - `questions`: a dict of question id -> {type, instructions, criteria}, exactly Laya's own
    multi-question contract (see laya.agent.Agent.predict_batch) -- every ticket is answered
    against every question in this dict, in one shared forward pass per batch.
  - `instruction`: the legacy single-question mode (still used for -InputFormat excel), which
    builds one hardcoded noul "automatable" question. Handled by the exact same machinery with
    a one-question dict, so there is only one code path to keep correct.
"""
import json
import math
import os
import sys
from datetime import datetime
from time import perf_counter

import numpy as np
import openvino as ov
from transformers import AutoTokenizer

from laya.agent import Agent
from laya.common import (
    QTYPES, answer_confidence, build_sequence, confidence_from_probs, encode_text,
    render_options, serialize_state, temp_bucket,
)

DEFAULT_MAX_LEN = 128


def _build_fixed_shape_feeds(chunk_items, pad_token_id, ov_batch_size, max_len, num_markers):
    # Every call feeds exactly (ov_batch_size, max_len): the model is compiled for this one
    # static shape, so the GPU plugin never has to JIT-compile kernels for a new shape and the
    # persistent OpenVINO/driver caches are actually reused across batches and process runs.
    input_ids = np.full((ov_batch_size, max_len), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((ov_batch_size, max_len), dtype=np.int64)
    marker_pos = np.zeros((ov_batch_size, num_markers), dtype=np.int64)
    # float32, not bool: the master export's marker_mask input is a plain 0.0/1.0 additive-mask
    # float tensor (see export_laya_master.py) so no boolean/u8 tensor ever reaches OpenVINO.
    marker_mask = np.zeros((ov_batch_size, num_markers), dtype=np.float32)
    qtype = np.zeros((ov_batch_size,), dtype=np.int64)
    for i, item in enumerate(chunk_items):
        ids = item["ids"]
        markers = item["markers"]
        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = 1
        marker_pos[i, : len(markers)] = markers
        marker_mask[i, : len(markers)] = 1.0
        qtype[i] = item["qtype"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": qtype,
    }


def _encode_ticket(tokenizer, internal, qids, ticket, max_len, head_max_len):
    """One sequence item per question id for this ticket, tokenizing its state once."""
    state = {"description": ticket["description"]}
    state_ids = encode_text(
        tokenizer, serialize_state(state).replace(tokenizer.mask_token, " "),
        add_special_tokens=False,
    )["input_ids"]
    items = []
    for qid in qids:
        q = internal[qid]
        seq, markers = build_sequence(
            tokenizer, state, q, max_len, head_max_len,
            truncate_left=False, state_ids=state_ids,
        )
        n_opts = len(render_options(q))
        if len(markers) != n_opts:
            raise ValueError(
                f"row {ticket['rowNumber']}: question '{qid}' needs {n_opts} option markers, got "
                f"{len(markers)}; the question text is too long for max_len={max_len}/head_max_len={head_max_len}"
            )
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    return items


def _decode_ticket_answers(logits, items, qids, internal, offset, temperature, temperature_by_options):
    """Turn `len(qids)` logit rows starting at `offset` into this ticket's typed answers.

    Mirrors laya.agent.Agent._decode_answers / laya.onnx_agent.ONNXAgent._decode_answers exactly,
    so a choice/score/noul answer here has the same fields and the same meaning as the official
    PyTorch/ONNX Runtime backends.
    """
    answers = {}
    for j, qid in enumerate(qids):
        r = offset + j
        q = internal[qid]
        k = len(items[j]["markers"])
        qt = QTYPES[q["t"]]
        t_scale = temperature_by_options.get(temp_bucket(qt, k), temperature[qt])
        z = logits[r, :k] / t_scale
        p = np.exp(z - z.max())
        p = p / p.sum()

        conf_score = round(confidence_from_probs(p, k), 4)
        ans_conf = round(answer_confidence(p, k), 4)

        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            answers[qid] = {
                "type": "choice",
                "choice": keys[int(p.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                "confidence": conf_score,
                "answer_confidence": ans_conf,
            }
        elif q["t"] == "score":
            exp_score = float((np.arange(k) * p).sum())
            answers[qid] = {
                "type": "score",
                "score": round(exp_score, 4),
                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": conf_score,
                "answer_confidence": ans_conf,
            }
        else:
            answers[qid] = {
                "type": "noul",
                "noul": round(float(p[1]), 4),
                "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                "answer_confidence": ans_conf,
            }
    return answers


def _format_answer_summary(answers, qids):
    """One compact line per ticket, e.g. 'service=technique, remboursement=0.83, urgence=2'."""
    parts = []
    for qid in qids:
        a = answers[qid]
        if a["type"] == "choice":
            parts.append(f"{qid}={a['choice']}")
        elif a["type"] == "score":
            parts.append(f"{qid}={a['score']:g}")
        else:
            parts.append(f"{qid}={a['noul']:g}")
    return ", ".join(parts)


def _infer_batch_answers(compiled, tokenizer, internal, qids, batch, ov_batch_size, max_len, head_max_len,
                         num_markers, temperature, temperature_by_options):
    # batch is already sized to at most (ov_batch_size / len(qids)) tickets (see main()), so this
    # is one inference call: every ticket contributes len(qids) rows, one per question.
    per_ticket_items = [
        _encode_ticket(tokenizer, internal, qids, ticket, max_len, head_max_len) for ticket in batch
    ]
    flat_items = [item for items in per_ticket_items for item in items]

    feeds = _build_fixed_shape_feeds(flat_items, tokenizer.pad_token_id, ov_batch_size, max_len, num_markers)
    result = compiled(feeds)
    logits = result["logits"][: len(flat_items)]

    answers_per_ticket = []
    offset = 0
    for items in per_ticket_items:
        answers_per_ticket.append(
            _decode_ticket_answers(logits, items, qids, internal, offset, temperature, temperature_by_options)
        )
        offset += len(items)
    return answers_per_ticket


def main():
    payload = json.load(sys.stdin)
    tickets = payload["tickets"]
    questions = payload.get("questions")

    if questions:
        if not isinstance(questions, dict):
            raise ValueError("payload 'questions' must be a dict of question id -> definition")
        for qid, qdef in questions.items():
            Agent._check_question(qid, qdef)
        qids = list(questions.keys())
        internal = {qid: Agent._to_internal(questions[qid]) for qid in qids}
    else:
        instruction = payload["instruction"].strip()
        qids = ["automatable"]
        internal = {
            "automatable": {
                "t": "noul",
                "ins": f"{instruction} Apply this only to the ticket description.",
                "crit": {
                    "false": "The ticket does not meet the instruction and cannot be automated.",
                    "true": "The ticket meets the instruction and can be automated.",
                },
            }
        }
    num_questions = len(qids)

    device = os.environ["LAYA_OV_DEVICE"]
    tickets_per_batch = int(os.environ["LAYA_OV_BATCH_SIZE"])
    onnx_path = os.environ["LAYA_OV_ONNX_PATH"]
    tokenizer_dir = os.environ["LAYA_OV_TOKENIZER_DIR"]
    cache_dir = os.environ["LAYA_OV_CACHE_DIR"]
    auto_max_len = os.environ.get("LAYA_OV_AUTO_MAX_LEN", "0").strip() == "1"
    manual_max_len = int(os.environ.get("LAYA_OV_MAX_LEN", DEFAULT_MAX_LEN))
    truncate_percent = float(os.environ.get("LAYA_OV_MAX_LEN_TRUNCATE_PERCENT", "0"))
    force_fp32 = os.environ.get("LAYA_OV_FORCE_FP32", "0").strip() == "1"

    checkpoint_dir = os.path.dirname(onnx_path)
    with open(os.path.join(checkpoint_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        agent_cfg = json.load(f)
    head_max_len = agent_cfg["head_max_len"]
    num_markers = agent_cfg["max_markers"]
    temperature = agent_cfg.get("temperature", [1.0, 1.0, 1.0])
    temperature_by_options = agent_cfg.get("temperature_by_options", {})

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # Rows per forward pass: every ticket in a batch contributes one row per question, so the
    # compiled OpenVINO batch axis is tickets_per_batch * num_questions, not just tickets_per_batch.
    ov_batch_size = tickets_per_batch * num_questions

    # Populated below only when auto_max_len is set; kept at these defaults otherwise so the JSON
    # output always has the same shape regardless of which max_len mode was used.
    truncated = 0
    longest = None
    target_len = None
    measure_seconds = 0.0

    if auto_max_len:
        # Measure the real sequence length this exact set of tickets/questions needs (uncapped, so
        # nothing is truncated during measurement): one length per ticket (the longest of its
        # question rows). truncate_percent lets max_len target a lower percentile instead of the
        # true max, at the cost of description truncation (never marker/instruction loss, see
        # laya.common.build_sequence: the state is always the part that gets cut) for the tickets
        # above that percentile.
        ticket_lengths = []
        total_tickets = len(tickets)
        report_every = max(1, total_tickets // 10)
        measure_started = perf_counter()
        for i, ticket in enumerate(tickets, start=1):
            items = _encode_ticket(tokenizer, internal, qids, ticket, max_len=4096, head_max_len=head_max_len)
            ticket_lengths.append(max(len(item["ids"]) for item in items))
            if i % report_every == 0 or i == total_tickets:
                elapsed = perf_counter() - measure_started
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (total_tickets - i) / rate if rate > 0 else 0.0
                print(
                    f"Measuring sequence lengths: {i}/{total_tickets} tickets "
                    f"({100.0 * i / total_tickets:.0f}%), {elapsed:.1f}s elapsed, {rate:.1f} tickets/s, "
                    f"ETA {eta:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
        measure_seconds = perf_counter() - measure_started
        ticket_lengths.sort()
        longest = ticket_lengths[-1]
        percentile = max(0.0, min(100.0, 100.0 - truncate_percent))
        rank = max(1, math.ceil(percentile / 100.0 * len(ticket_lengths)))
        target_len = ticket_lengths[rank - 1]
        max_len = min(4096, ((target_len + 8) + 7) // 8 * 8)
        truncated = sum(1 for length in ticket_lengths if length > max_len)
        budget_desc = "no truncation budget" if truncate_percent <= 0 else f"{truncate_percent:g}% truncation budget"
        print(
            f"OpenVINO auto max_len: {len(tickets)} tickets x {num_questions} question(s) measured in "
            f"{measure_seconds:.1f}s, longest={longest} tokens, {budget_desc} -> p{percentile:g} "
            f"length={target_len}, using max_len={max_len} (+8 margin, rounded up to a multiple of 8); "
            f"{truncated}/{len(tickets)} ticket(s) ({100.0 * truncated / len(tickets):.1f}%) will have their "
            "description truncated",
            file=sys.stderr,
            flush=True,
        )
    else:
        max_len = manual_max_len

    core = ov.Core()
    available = core.available_devices
    print(f"OpenVINO available devices: {', '.join(available)}", file=sys.stderr, flush=True)
    print(f"OpenVINO device requested: {device}", file=sys.stderr, flush=True)

    if device == "GPU" and "GPU" not in available:
        raise RuntimeError(
            f"OpenVINO device 'GPU' was requested but is not available. "
            f"Available devices: {', '.join(available)}"
        )

    os.makedirs(cache_dir, exist_ok=True)
    core.set_property({"CACHE_DIR": cache_dir})

    if force_fp32:
        # GPU plugin defaults INFERENCE_PRECISION_HINT to f16 for speed even though this ONNX
        # export is fp32; this restores true fp32 compute (CPU already defaults to f32/bf16, so
        # this only has an effect on GPU).
        core.set_property(device, {"INFERENCE_PRECISION_HINT": ov.Type.f32})

    def _cache_blob_sizes():
        return {name: os.path.getsize(os.path.join(cache_dir, name)) for name in os.listdir(cache_dir)}

    blobs_before = _cache_blob_sizes()
    print(
        f"OpenVINO cache_dir has {len(blobs_before)} blob(s) "
        f"({sum(blobs_before.values()) / 1_048_576:.1f} MB) before compiling: {cache_dir}",
        file=sys.stderr,
        flush=True,
    )

    compile_started = perf_counter()
    model = core.read_model(onnx_path)
    # Fix batch and sequence length so every inference call uses the exact same shape: with
    # dynamic shapes the GPU plugin JIT-compiles new kernels for every distinct (batch, seq_len)
    # it sees, which defeats CACHE_DIR across batches and process runs. The marker axis is
    # already static (5) in the master export, so it is not part of this reshape.
    model.reshape({
        "input_ids": ov.PartialShape([ov_batch_size, max_len]),
        "attention_mask": ov.PartialShape([ov_batch_size, max_len]),
        "marker_pos": ov.PartialShape([ov_batch_size, num_markers]),
        "marker_mask": ov.PartialShape([ov_batch_size, num_markers]),
        "qtype": ov.PartialShape([ov_batch_size]),
    })
    compiled = core.compile_model(model, device)
    compile_seconds = perf_counter() - compile_started
    # The uncompiled IR graph (can be sizeable in fp32) is no longer needed once `compiled`
    # exists; drop the reference now instead of keeping it alive for the whole inference loop.
    del model
    try:
        actual_device = compiled.get_property("EXECUTION_DEVICES")
    except Exception:
        actual_device = device
    try:
        actual_precision = str(compiled.get_property("INFERENCE_PRECISION_HINT"))
    except Exception:
        actual_precision = "unknown"

    new_blobs = [name for name in _cache_blob_sizes() if name not in blobs_before]
    cache_status = (
        f"cache MISS, wrote {len(new_blobs)} new blob(s)" if new_blobs else "cache HIT, reused an existing blob"
    )
    print(
        f"OpenVINO model compiled in {compile_seconds:.2f}s "
        f"(requested={device}, actual={actual_device}, runtime_precision={actual_precision}, "
        f"cache_dir={cache_dir}, {cache_status})",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"OpenVINO inference settings: tickets_per_batch={tickets_per_batch}, "
        f"questions_per_ticket={num_questions} ({', '.join(qids)}), rows_per_forward_pass={ov_batch_size}, "
        f"max_len={max_len}, head_max_len={head_max_len}, num_markers={num_markers}, "
        f"fixed_input_shape=({ov_batch_size}, {max_len}) "
        "(changing LayaBatchSize, the question schema, or the max_len setting forces a recompile)",
        file=sys.stderr,
        flush=True,
    )

    total_batches = (len(tickets) + tickets_per_batch - 1) // tickets_per_batch
    results = []
    inference_started_at = None
    inference_finished_at = None
    inference_seconds = 0.0
    tickets_done = 0
    loop_started = perf_counter()

    for batch_number, start in enumerate(range(0, len(tickets), tickets_per_batch), start=1):
        batch = tickets[start : start + tickets_per_batch]
        try:
            call_started_at = datetime.now().astimezone()
            call_started = perf_counter()
            if inference_started_at is None:
                inference_started_at = call_started_at
            try:
                answers_per_ticket = _infer_batch_answers(
                    compiled, tokenizer, internal, qids, batch, ov_batch_size, max_len, head_max_len,
                    num_markers, temperature, temperature_by_options,
                )
            finally:
                inference_seconds += perf_counter() - call_started
                inference_finished_at = datetime.now().astimezone()

            for ticket, answers in zip(batch, answers_per_ticket):
                results.append({"rowNumber": ticket["rowNumber"], "answers": answers, "error": None})
                print(
                    f"  ticket {ticket['rowNumber']}: {_format_answer_summary(answers, qids)}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for ticket in batch:
                results.append({"rowNumber": ticket["rowNumber"], "answers": None, "error": error})
            print(f"Batch {batch_number} failed: {error}", file=sys.stderr, flush=True)

        tickets_done += len(batch)
        elapsed = perf_counter() - loop_started
        rate = tickets_done / elapsed if elapsed > 0 else 0.0
        remaining = len(tickets) - tickets_done
        eta = remaining / rate if rate > 0 else 0.0
        print(
            f"Completed batch {batch_number}/{total_batches} "
            f"({tickets_done}/{len(tickets)} tickets, {100.0 * tickets_done / len(tickets):.0f}%), "
            f"{elapsed:.1f}s elapsed, {rate:.1f} tickets/s, ETA {eta:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    if inference_seconds > 0:
        print(
            f"OpenVINO throughput: {len(tickets) / inference_seconds:.1f} tickets/s "
            f"over {inference_seconds:.2f}s",
            file=sys.stderr,
            flush=True,
        )

    json.dump(
        {
            "results": results,
            "inference": {
                "startedAt": inference_started_at.isoformat(timespec="seconds"),
                "finishedAt": inference_finished_at.isoformat(timespec="seconds"),
                "seconds": inference_seconds,
                "checkpoints": [
                    f"convaiinnovations/laya-multilingual (laya-master, onnx-fp32, "
                    f"runtime_precision={actual_precision}, openvino:{actual_device})"
                ],
                "compileSeconds": compile_seconds,
                "cacheStatus": cache_status,
                "maxLen": max_len,
                "maxLenAuto": auto_max_len,
                "maxLenTruncatedTickets": truncated,
                "maxLenTruncatedPercent": (100.0 * truncated / len(tickets)) if tickets else 0.0,
            },
        },
        sys.stdout,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()

