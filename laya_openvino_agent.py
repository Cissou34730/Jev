"""A Laya Agent backed by the local OpenVINO build, for `laya-serve`.

Standalone on purpose: it duplicates the encoding, padding and decoding of openvino_runner.py
rather than importing it, so a change here cannot break the runner or the other server.

It exposes what `laya.router.Router` calls:
  system_one(state, questions, lang=, max_len=, head_max_len=)
  predict_batch(states, questions, batch_size=, lang=, max_len=, head_max_len=, sort_by_length=)
`lang` and `sort_by_length` are accepted and ignored; `max_len` / `head_max_len` are accepted and
ignored too, because the model is compiled once per shape and recompiling per request is what we
are avoiding. Sequences longer than that length have their state truncated.

No `lang_temperatures` attribute: that is what keeps the Router from forwarding `lang` into
`predict_batch`, and its absence is checked with `getattr`.

Faithful to Laya's own batching: one forward pass per call, holding exactly the rows that call
needs -- states x questions for `predict_batch`, questions for `system_one`. There is no padding,
and therefore no rows of work paid for rows nobody asked about. OpenVINO needs a fixed shape per
compiled model, so each distinct row count is compiled on first use and kept, which for a fixed
question schema is exactly one shape. A call above `LAYA_OV_MAX_ROWS` is split into passes of that
size, so memory stays bounded.

Configuration, all environment variables:
  LAYA_OV_ONNX_PATH       the exported model               (default .laya-cache/.../laya.onnx)
  LAYA_OV_TOKENIZER_DIR   the tokenizer directory          (default next to the model)
  LAYA_OV_CACHE_DIR       the OpenVINO compile cache       (default .laya-cache/openvino-cache)
  LAYA_OV_DEVICE          GPU or CPU                       (default GPU)
  LAYA_OV_FORCE_FP32      1 to compute in fp32 on the GPU  (default 0)
  LAYA_OV_MAX_LEN         the compiled sequence length     (default 128)
  LAYA_OV_MAX_ROWS        rows per forward pass at most    (default 256)
  LAYA_OV_SHAPE_CACHE     compiled shapes kept resident    (default 8)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import openvino as ov
from transformers import AutoTokenizer

from laya.agent import Agent
from laya.common import (
    QTYPES, _reuse_question_tokens, answer_confidence, build_sequence, confidence_from_probs,
    encode_text, render_options, serialize_state, temp_bucket,
)

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / ".laya-cache" / "checkpoints" / "laya-master"


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


# When this process started and how much memory it holds. A pass that slows down has to be read
# against both: with one fixed input shape, the shape cache cannot be the cause, but a growing
# working set or a long uptime can.
_PROCESS_STARTED = time.perf_counter()

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    def _resident_mb() -> float:
        """Working set of this process in MB, or 0 when it cannot be read."""
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        try:
            call = ctypes.windll.psapi.GetProcessMemoryInfo
            # Without these the pseudo-handle from GetCurrentProcess (-1) is passed as 32 bits and
            # the call fails, which is how this reported 0 MB for every pass.
            call.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
            call.restype = wintypes.BOOL
            ok = call(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        except OSError:
            return 0.0
        return counters.WorkingSetSize / (1024 * 1024) if ok else 0.0

else:
    def _resident_mb() -> float:
        return 0.0


class OpenVINOAgent:
    """Same surface as `laya.Agent` for the calls the Router makes, on the OpenVINO model."""

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        tokenizer_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        force_fp32: Optional[bool] = None,
        max_len: Optional[int] = None,
        max_rows: Optional[int] = None,
        shape_cache: Optional[int] = None,
    ) -> None:
        self.onnx_path = onnx_path or _env("LAYA_OV_ONNX_PATH", str(CHECKPOINT / "laya.onnx"))
        checkpoint = Path(self.onnx_path).parent
        self.tokenizer_dir = tokenizer_dir or _env("LAYA_OV_TOKENIZER_DIR", str(checkpoint))
        self.cache_dir = cache_dir or _env("LAYA_OV_CACHE_DIR", str(ROOT / ".laya-cache" / "openvino-cache"))
        self.device = device or _env("LAYA_OV_DEVICE", "GPU")
        self.force_fp32 = (
            _env("LAYA_OV_FORCE_FP32", "0") == "1" if force_fp32 is None else bool(force_fp32)
        )
        self.max_len = int(max_len or _env("LAYA_OV_MAX_LEN", "128"))
        self.max_rows = int(max_rows or _env("LAYA_OV_MAX_ROWS", "256"))
        self.shape_cache_size = int(shape_cache or _env("LAYA_OV_SHAPE_CACHE", "8"))
        if not Path(self.onnx_path).is_file():
            raise RuntimeError(f"ONNX model not found: {self.onnx_path}. Run export_laya_master.py first.")

        config = json.loads((checkpoint / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.head_max_len = int(config["head_max_len"])
        self.num_markers = int(config["max_markers"])
        self.temperature = config.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = config.get("temperature_by_options", {})
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        self.model_id = "laya-master-openvino"
        self._core = None
        self._compiled: Dict[int, Any] = {}
        # Counted here, not by the caller: one forward pass is one call to the compiled model, and
        # with several states per call that is not the same as one request and not the same as one
        # batch. Printed as it happens, so a server log says whether anything is going on at all.
        self.pass_count = 0

    # -- engine ------------------------------------------------------------------------------

    def _compile(self, rows: int):
        """The model compiled for exactly this many rows, kept for reuse."""
        if rows in self._compiled:
            return self._compiled[rows]
        if self._core is None:
            core = ov.Core()
            available = core.available_devices
            if self.device == "GPU" and "GPU" not in available:
                raise RuntimeError(f"OpenVINO device 'GPU' requested but not available: {', '.join(available)}")
            os.makedirs(self.cache_dir, exist_ok=True)
            core.set_property({"CACHE_DIR": self.cache_dir})
            if self.force_fp32:
                # The GPU plugin defaults to f16 even for an fp32 export.
                core.set_property(self.device, {"INFERENCE_PRECISION_HINT": ov.Type.f32})
            self._core = core
        model = self._core.read_model(self.onnx_path)
        # A fixed shape per compiled model, so the GPU plugin never JIT-compiles kernels per call
        # and CACHE_DIR is reused across calls and process runs. One shape per row count seen.
        model.reshape({
            "input_ids": ov.PartialShape([rows, self.max_len]),
            "attention_mask": ov.PartialShape([rows, self.max_len]),
            "marker_pos": ov.PartialShape([rows, self.num_markers]),
            "marker_mask": ov.PartialShape([rows, self.num_markers]),
            "qtype": ov.PartialShape([rows]),
        })
        compiled = self._core.compile_model(model, self.device)
        del model
        if len(self._compiled) >= self.shape_cache_size:
            self._compiled.pop(next(iter(self._compiled)))
        self._compiled[rows] = compiled
        return compiled

    def _feeds(self, items: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        rows, pad = len(items), self.tokenizer.pad_token_id
        input_ids = np.full((rows, self.max_len), pad, dtype=np.int64)
        attention_mask = np.zeros((rows, self.max_len), dtype=np.int64)
        marker_pos = np.zeros((rows, self.num_markers), dtype=np.int64)
        marker_mask = np.zeros((rows, self.num_markers), dtype=np.float32)
        qtype = np.zeros((rows,), dtype=np.int64)
        for i, item in enumerate(items):
            ids, markers = item["ids"], item["markers"]
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

    # -- encoding and decoding ---------------------------------------------------------------

    @_reuse_question_tokens
    def _encode_state(self, state: Any, internal: Dict[str, Any], ids: List[str]) -> List[Dict[str, Any]]:
        """One row per question of this state; its tokens are tokenized once and shared."""
        state_ids = encode_text(
            self.tokenizer, serialize_state(state).replace(self.tokenizer.mask_token, " "),
            add_special_tokens=False,
        )["input_ids"]
        items = []
        for qid in ids:
            q = internal[qid]
            seq, markers = build_sequence(
                self.tokenizer, state, q, self.max_len, self.head_max_len,
                truncate_left=isinstance(state, list), state_ids=state_ids,
            )
            k = len(render_options(q))
            if len(markers) != k:
                raise ValueError(
                    f"question {qid!r}: needs {k} option markers, got {len(markers)}; the question "
                    f"text is too long for max_len={self.max_len}/head_max_len={self.head_max_len}"
                )
            items.append({"qid": qid, "ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
        return items

    def _decode(self, logits: np.ndarray, item: Dict[str, Any], q: Dict[str, Any]) -> Dict[str, Any]:
        k = len(item["markers"])
        qtype = item["qtype"]
        scale = self.temperature_by_options.get(temp_bucket(qtype, k), self.temperature[qtype])
        z = logits[:k] / scale
        p = np.exp(z - z.max())
        p = p / p.sum()
        confidence = round(confidence_from_probs(p, k), 4)
        answer_confidence_value = round(answer_confidence(p, k), 4)
        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            return {
                "type": "choice",
                "choice": keys[int(p.argmax())],
                "probabilities": {key: round(float(value), 4) for key, value in zip(keys, p)},
                "confidence": confidence,
                "answer_confidence": answer_confidence_value,
            }
        if q["t"] == "score":
            return {
                "type": "score",
                "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                "probabilities": {str(i): round(float(value), 4) for i, value in enumerate(p)},
                "confidence": confidence,
                "answer_confidence": answer_confidence_value,
            }
        return {
            "type": "noul",
            "noul": round(float(p[1]), 4),
            "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
            "answer_confidence": answer_confidence_value,
        }

    # -- inference ---------------------------------------------------------------------------

    def _run(self, per_state_items: List[List[Dict[str, Any]]], internal: Dict[str, Any], ids: List[str]):
        """Answer every (state, question) row; returns (per-state answers, input tokens)."""
        flat = [(state_index, item) for state_index, items in enumerate(per_state_items) for item in items]
        input_tokens = sum(len(item["ids"]) for _, item in flat)
        answers: List[Dict[str, Any]] = [{} for _ in per_state_items]
        for start in range(0, len(flat), self.max_rows):
            part = flat[start:start + self.max_rows]
            rows = len(part)
            states = len({state_index for state_index, _ in part})
            compiled = self._compile(rows)
            # The phases are timed apart because one total says nothing about the GPU: building the
            # feeds, copying the logits and decoding are CPU work on this same thread, and only the
            # middle number is the model.
            feeds_started = time.perf_counter()
            feeds = self._feeds([item for _, item in part])
            feeds_seconds = time.perf_counter() - feeds_started
            infer_started = time.perf_counter()
            raw_logits = compiled(feeds)["logits"]
            infer_seconds = time.perf_counter() - infer_started
            copy_started = time.perf_counter()
            logits = np.asarray(raw_logits)
            copy_seconds = time.perf_counter() - copy_started
            decode_started = time.perf_counter()
            for offset, (state_index, item) in enumerate(part):
                qid = item["qid"]
                answers[state_index][qid] = self._decode(logits[offset], item, internal[qid])
            decode_seconds = time.perf_counter() - decode_started
            self.pass_count += 1
            total_seconds = feeds_seconds + infer_seconds + copy_seconds + decode_seconds
            print(
                f"forward pass {self.pass_count}: {states} state(s), {rows} items | "
                f"feeds {feeds_seconds:.2f}s infer {infer_seconds:.2f}s "
                f"copy {copy_seconds:.2f}s decode {decode_seconds:.2f}s | "
                f"total {total_seconds:.2f}s | up {time.perf_counter() - _PROCESS_STARTED:.0f}s, "
                f"rss {_resident_mb():.0f} MB",
                file=sys.stderr, flush=True,
            )
        return answers, input_tokens

    def _prepare(self, questions: Any):
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty dict of question id -> definition")
        for qid, qdef in questions.items():
            Agent._check_question(qid, qdef)
        return list(questions.keys()), {qid: Agent._to_internal(qdef) for qid, qdef in questions.items()}

    def predict_batch(
        self,
        states: List[Any],
        questions: Dict[str, Any],
        batch_size: Optional[int] = None,
        lang: Optional[str] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        sort_by_length: bool = False,
    ) -> List[Dict[str, Any]]:
        ids, internal = self._prepare(questions)
        states = list(states)
        chunk = batch_size if batch_size and batch_size > 0 else max(1, len(states))
        results: List[Dict[str, Any]] = []
        for start in range(0, len(states), chunk):
            part = states[start:start + chunk]
            # Tokenizing the states and assembling their sequences happens before `_run`, so it is
            # the one part of a request the pass line cannot show. It is not free.
            encode_started = time.perf_counter()
            encoded = [self._encode_state(state, internal, ids) for state in part]
            print(
                f"encode: {len(part)} state(s) x {len(ids)} question(s), "
                f"{time.perf_counter() - encode_started:.2f}s",
                file=sys.stderr, flush=True,
            )
            answers, input_tokens = self._run(encoded, internal, ids)
            tokens = max(1, input_tokens // max(1, len(part)))
            for state_answers in answers:
                results.append({
                    "model": self.model_id,
                    "answers": state_answers,
                    # output_tokens is the answer count: this model is not autoregressive.
                    "usage": {"input_tokens": tokens, "output_tokens": len(ids)},
                })
        return results

    def system_one(
        self,
        state: Any,
        questions: Dict[str, Any],
        lang: Optional[str] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.predict_batch([state], questions)[0]

    def __repr__(self) -> str:
        return f"OpenVINOAgent(device={self.device!r}, max_len={self.max_len}, shapes={sorted(self._compiled)})"
