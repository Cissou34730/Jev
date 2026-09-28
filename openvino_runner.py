"""Local OpenVINO inference path for laya's noul questions.

Mirrors laya_runner.py's stdin/stdout JSON contract exactly, but compiles the pinned
English FP16 ONNX checkpoint (Mattepiu/laya-onnx@a7f385b, fp16_onlygpu_unverified/laya_fp16.onnx)
with OpenVINO instead of running the PyTorch model or ONNX Runtime. Reuses laya's own
tokenizer, sequence-building and serialization so encoding matches the CPU path exactly.
"""
import json
import os
import sys
from datetime import datetime
from time import perf_counter

import numpy as np
import openvino as ov
from transformers import AutoTokenizer

from laya.common import QTYPES, build_sequence, collate_items, encode_text, serialize_state

MAX_LEN = 512
HEAD_MAX_LEN = 192
# This checkpoint's exported graph fixes marker_pos/marker_mask to exactly 2 options,
# which matches the noul question below (false/true) and no other question type.
NUM_MARKERS = 2


def _infer_noul_scores(compiled, tokenizer, internal_question, batch, ov_batch_size):
    items = []
    for ticket in batch:
        state = {"description": ticket["description"]}
        state_ids = encode_text(
            tokenizer, serialize_state(state).replace(tokenizer.mask_token, " "),
            add_special_tokens=False,
        )["input_ids"]
        seq, markers = build_sequence(
            tokenizer, state, internal_question, MAX_LEN, HEAD_MAX_LEN,
            truncate_left=False, state_ids=state_ids,
        )
        if len(markers) != NUM_MARKERS:
            raise ValueError(
                f"row {ticket['rowNumber']}: expected {NUM_MARKERS} option markers, got {len(markers)}; "
                f"the instruction text is too long for max_len={MAX_LEN}/head_max_len={HEAD_MAX_LEN}"
            )
        items.append({"ids": seq, "markers": markers})

    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    scores = [None] * len(items)

    for start in range(0, len(order), ov_batch_size):
        chunk_indices = order[start : start + ov_batch_size]
        chunk_items = [items[i] for i in chunk_indices]
        collated = collate_items([[it] for it in chunk_items], tokenizer.pad_token_id)
        feeds = {
            "input_ids": collated["input_ids"].numpy().astype(np.int64),
            "attention_mask": collated["attention_mask"].numpy().astype(np.int64),
            "marker_pos": collated["marker_pos"].numpy().astype(np.int64),
            "marker_mask": collated["marker_mask"].numpy().astype(np.bool_),
            "qtype": np.full((len(chunk_items),), QTYPES["noul"], dtype=np.int64),
        }
        result = compiled(feeds)
        logits = result["logits"]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        for local_i, global_i in enumerate(chunk_indices):
            scores[global_i] = float(probs[local_i, 1])

    return scores


def main():
    payload = json.load(sys.stdin)
    tickets = payload["tickets"]
    outer_batch_size = int(payload["batchSize"])
    instruction = payload["instruction"].strip()

    device = os.environ["LAYA_OV_DEVICE"]
    ov_batch_size = int(os.environ["LAYA_OV_BATCH_SIZE"])
    onnx_path = os.environ["LAYA_OV_ONNX_PATH"]
    tokenizer_dir = os.environ["LAYA_OV_TOKENIZER_DIR"]
    cache_dir = os.environ["LAYA_OV_CACHE_DIR"]

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

    compile_started = perf_counter()
    model = core.read_model(onnx_path)
    compiled = core.compile_model(model, device)
    compile_seconds = perf_counter() - compile_started
    try:
        actual_device = compiled.get_property("EXECUTION_DEVICES")
    except Exception:
        actual_device = device
    print(
        f"OpenVINO model compiled in {compile_seconds:.2f}s "
        f"(requested={device}, actual={actual_device}, cache_dir={cache_dir})",
        file=sys.stderr,
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    internal_question = {
        "t": "noul",
        "ins": f"{instruction} Apply this only to the ticket description.",
        "crit": {
            "false": "The ticket does not meet the instruction and cannot be automated.",
            "true": "The ticket meets the instruction and can be automated.",
        },
    }

    print(
        f"OpenVINO inference settings: outer_ticket_batch_size={outer_batch_size}, "
        f"ov_internal_batch_size={ov_batch_size}, max_len={MAX_LEN}, head_max_len={HEAD_MAX_LEN} "
        "(checkpoint defaults; this repository does not ship rl_agent_config.json), "
        "sort_by_length=True",
        file=sys.stderr,
        flush=True,
    )

    total_batches = (len(tickets) + outer_batch_size - 1) // outer_batch_size
    results = []
    inference_started_at = None
    inference_finished_at = None
    inference_seconds = 0.0

    for batch_number, start in enumerate(range(0, len(tickets), outer_batch_size), start=1):
        batch = tickets[start : start + outer_batch_size]
        try:
            call_started_at = datetime.now().astimezone()
            call_started = perf_counter()
            if inference_started_at is None:
                inference_started_at = call_started_at
            try:
                scores = _infer_noul_scores(compiled, tokenizer, internal_question, batch, ov_batch_size)
            finally:
                inference_seconds += perf_counter() - call_started
                inference_finished_at = datetime.now().astimezone()

            for ticket, score in zip(batch, scores):
                results.append({"rowNumber": ticket["rowNumber"], "score": score, "error": None})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for ticket in batch:
                results.append({"rowNumber": ticket["rowNumber"], "score": None, "error": error})
            print(f"Batch {batch_number} failed: {error}", file=sys.stderr, flush=True)

        print(
            f"Completed OpenVINO batch {batch_number} / {total_batches}",
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
                "checkpoints": [f"Mattepiu/laya-onnx@a7f385b (english, fp16, openvino:{actual_device})"],
            },
        },
        sys.stdout,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()
