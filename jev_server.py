"""Jev-compatible HTTP server backed by the local OpenVINO Laya engine.

Routes (contract of https://api.typesafe.ai/openapi.json):
  POST /v1/systemone   {state, questions, model?} -> {model, answers, usage}
  GET  /v1/models
plus GET /health and the admin routes below.

A request may carry many questions, and a structured `state` holding many items that questions
address by backticked path (`tickets[3].description`). Laya reads one "question + state" row at a
time and cuts the state to max_len, so each question is given a state reduced to the values it
references; a question without references gets the whole state. The state of a row longer than
JEV_SERVER_MAX_LEN tokens is truncated (the question and its options never are).

Concurrent requests share forward passes (micro-batching). The model lives in a separate worker
Process (jev_worker.py), compiled once at the length JEV_SERVER_MAX_LEN fixes, or, when that is 0,
from the lengths of the traffic it sees (JEV_SERVER_MAX_LEN_AUTO_PERCENT of the longest rows are
truncated instead of forcing a recompile).

Admin routes:
  GET  /admin/status
  POST /admin/unload  {"force": bool}
  POST /admin/cancel  {"request_id": str | "all": true, "force": bool}

Every number is an environment variable, JEV_SERVER_* (see Settings); engine paths and device use
the LAYA_OV_* names of openvino_runner.py.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import secrets
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from transformers import AutoTokenizer

from laya.agent import Agent
from laya.common import QTYPES, _reuse_question_tokens, build_sequence, encode_text, render_options, serialize_state
from openvino_runner import _build_fixed_shape_feeds, _decode_ticket_answers

ROOT = Path(__file__).resolve().parent
WORKER_PATH = ROOT / "jev_worker.py"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def log(message: str) -> None:
    print(f"{datetime.now().astimezone():%H:%M:%S} {message}", file=sys.stderr, flush=True)


def _env(name: str, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw.strip())
    except ValueError as exc:
        raise SystemExit(f"invalid value for {name}: {raw!r}") from exc


def _flag(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    backlog: int
    log_level: str
    model_name: str
    device: str
    onnx_path: str
    tokenizer_dir: str
    cache_dir: str
    force_fp32: bool
    max_len: int
    max_len_cap: int
    max_len_step: int
    max_len_auto_percent: float
    max_len_auto_rows: int
    max_len_auto_wait_ms: float
    rows_per_pass: int
    batch_wait_ms: float
    batch_max_wait_ms: float
    max_questions: int
    max_body_bytes: int
    max_inflight_requests: int
    max_pending_rows: int
    retry_after_seconds: int
    request_timeout_seconds: float
    pass_timeout_seconds: float
    load_timeout_seconds: float
    watchdog_interval_seconds: float
    disconnect_poll_seconds: float
    max_pass_retries: int
    encode_concurrency: int
    worker_start_timeout_seconds: float
    worker_stop_timeout_seconds: float
    status_max_requests: int

    @classmethod
    def from_env(cls) -> "Settings":
        p = "JEV_SERVER_"
        checkpoint = ROOT / ".laya-cache" / "checkpoints" / "laya-master"
        return cls(
            host=_env(p + "HOST", "127.0.0.1"),
            port=_env(p + "PORT", 8000, int),
            backlog=_env(p + "BACKLOG", 2048, int),
            log_level=_env(p + "LOG_LEVEL", "info"),
            model_name=_env(p + "MODEL_NAME", "laya-master-openvino"),
            device=_env("LAYA_OV_DEVICE", "GPU"),
            onnx_path=_env("LAYA_OV_ONNX_PATH", str(checkpoint / "laya.onnx")),
            tokenizer_dir=_env("LAYA_OV_TOKENIZER_DIR", str(checkpoint)),
            cache_dir=_env("LAYA_OV_CACHE_DIR", str(ROOT / ".laya-cache" / "openvino-cache")),
            force_fp32=_env("LAYA_OV_FORCE_FP32", False, _flag),
            max_len=_env(p + "MAX_LEN", 0, int),
            max_len_cap=_env(p + "MAX_LEN_CAP", 4096, int),
            max_len_step=_env(p + "MAX_LEN_STEP", 8, int),
            max_len_auto_percent=_env(p + "MAX_LEN_AUTO_PERCENT", 3.0, float),
            max_len_auto_rows=_env(p + "MAX_LEN_AUTO_ROWS", 150, int),
            max_len_auto_wait_ms=_env(p + "MAX_LEN_AUTO_WAIT_MS", 2000.0, float),
            rows_per_pass=_env(p + "ROWS_PER_PASS", 150, int),
            batch_wait_ms=_env(p + "BATCH_WAIT_MS", 50.0, float),
            batch_max_wait_ms=_env(p + "BATCH_MAX_WAIT_MS", 1000.0, float),
            max_questions=_env(p + "MAX_QUESTIONS", 100_000, int),
            max_body_bytes=_env(p + "MAX_BODY_BYTES", 64 * 1024 * 1024, int),
            max_inflight_requests=_env(p + "MAX_INFLIGHT_REQUESTS", 50_000, int),
            max_pending_rows=_env(p + "MAX_PENDING_ROWS", 500_000, int),
            retry_after_seconds=_env(p + "RETRY_AFTER_SECONDS", 1, int),
            request_timeout_seconds=_env(p + "REQUEST_TIMEOUT_SECONDS", 3600.0, float),
            pass_timeout_seconds=_env(p + "PASS_TIMEOUT_SECONDS", 600.0, float),
            load_timeout_seconds=_env(p + "LOAD_TIMEOUT_SECONDS", 900.0, float),
            watchdog_interval_seconds=_env(p + "WATCHDOG_INTERVAL_SECONDS", 1.0, float),
            disconnect_poll_seconds=_env(p + "DISCONNECT_POLL_SECONDS", 0.5, float),
            max_pass_retries=_env(p + "MAX_PASS_RETRIES", 2, int),
            encode_concurrency=_env(p + "ENCODE_CONCURRENCY", 2, int),
            worker_start_timeout_seconds=_env(p + "WORKER_START_TIMEOUT_SECONDS", 120.0, float),
            worker_stop_timeout_seconds=_env(p + "WORKER_STOP_TIMEOUT_SECONDS", 10.0, float),
            status_max_requests=_env(p + "STATUS_MAX_REQUESTS", 200, int),
        )

    def validate(self) -> None:
        # No key is configured, so a host anything other than this machine could reach is refused.
        if self.host not in LOOPBACK_HOSTS:
            raise SystemExit(f"JEV_SERVER_HOST={self.host} is not a loopback address; this server is local-only")
        for name in (
            "rows_per_pass", "max_questions", "encode_concurrency", "max_pass_retries",
            "max_len_step", "max_len_auto_rows",
        ):
            if getattr(self, name) < 1:
                raise SystemExit(f"JEV_SERVER_{name.upper()} must be at least 1")
        if self.max_len < 0 or self.max_len_cap < self.max_len_step:
            raise SystemExit("need JEV_SERVER_MAX_LEN >= 0 (0 = automatic) and MAX_LEN_CAP >= MAX_LEN_STEP")
        if not 0.0 <= self.max_len_auto_percent < 100.0:
            raise SystemExit("JEV_SERVER_MAX_LEN_AUTO_PERCENT must be in [0, 100)")


# --- references: `path` tokens in questions select what part of the state a question sees ---------

_BACKTICK = re.compile(r"`([^`]+)`")
_SEGMENT = re.compile(r"^([^.\[\]]*)((?:\[\d+\])*)$")
_INDEX = re.compile(r"\[(\d+)\]")
_IDENTIFIER = re.compile(r"^[A-Za-z_][\w\-]*$")


class RequestError(Exception):
    def __init__(self, qid: str, message: str):
        super().__init__(message)
        self.qid = qid
        self.message = message


def _collect_backticked(value: Any, out: list) -> None:
    if isinstance(value, str):
        out.extend(_BACKTICK.findall(value))
    elif isinstance(value, dict):
        for item in value.values():
            _collect_backticked(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_backticked(item, out)


def _parse_path(token: str) -> Optional[list]:
    segments: list = []
    for part in token.split("."):
        match = _SEGMENT.match(part)
        if match is None:
            return None
        if match.group(1):
            segments.append(match.group(1))
        segments.extend(int(i) for i in _INDEX.findall(match.group(2)))
    return segments or None


def _walk(state: Any, segments: list) -> Any:
    current = state
    for segment in segments:
        if isinstance(segment, int):
            if not isinstance(current, list) or segment >= len(current):
                raise LookupError(segment)
            current = current[segment]
        else:
            if not isinstance(current, dict) or segment not in current:
                raise LookupError(segment)
            current = current[segment]
    return current


def question_refs(qdef: dict, state: Any) -> dict:
    """Backticked paths of a question that point into the state, mapped to their values."""
    if not isinstance(state, (dict, list)):
        return {}
    tokens: list = []
    _collect_backticked(qdef.get("instructions"), tokens)
    _collect_backticked(qdef.get("criteria"), tokens)
    instructions = qdef.get("instructions")
    # Fields of a structured `instructions` object are named by the question itself, not by the state.
    local_keys = set(instructions) if isinstance(instructions, dict) else set()
    refs: dict = {}
    for token in tokens:
        if token in refs:
            continue
        segments = _parse_path(token)
        if segments is None:
            continue
        if isinstance(segments[0], str) and segments[0] in local_keys:
            continue
        shaped_like_path = all(isinstance(s, int) or _IDENTIFIER.match(s) for s in segments)
        is_path = len(segments) > 1 or "[" in token
        if not shaped_like_path or (not is_path and not (isinstance(state, dict) and token in state)):
            continue
        try:
            refs[token] = _walk(state, segments)
        except LookupError as exc:
            raise ValueError(f"reference `{token}` does not exist in state") from exc
    return refs


# --- rows and jobs --------------------------------------------------------------------------------


class Row:
    __slots__ = ("qid", "ids", "markers", "qtype")

    def __init__(self, qid: str, ids: np.ndarray, markers: list, qtype: int):
        self.qid = qid
        self.ids = ids
        self.markers = markers
        self.qtype = qtype


class Job:
    def __init__(self, state: Any, questions: dict, qids: list, internal: dict, rows: list):
        self.id = secrets.token_hex(6)
        self.state = state
        self.questions = questions
        self.qids = qids
        self.internal = internal
        self.rows = rows
        self.total = len(rows)
        self.remaining = len(rows)
        self.measure()
        self.trimmed = False
        self.pending = deque(range(len(rows)))
        self.answers: dict = {}
        self.created = time.monotonic()
        self.done = asyncio.Event()
        self.closed = False
        self.queued = False
        self.error: Optional[tuple] = None
        self.retries = 0

    def measure(self) -> None:
        self.needed_len = max(len(row.ids) for row in self.rows)
        self.input_tokens = sum(len(row.ids) for row in self.rows)


@_reuse_question_tokens
def _encode_request(tokenizer, head_max_len: int, max_len: int, state: Any, questions: dict):
    """One row per question of this state; the state is tokenized once and reused for all of them."""
    internal: dict = {}
    rows: list = []
    state_ids_by_refs: dict = {}
    mask = tokenizer.mask_token
    for qid, qdef in questions.items():
        try:
            Agent._check_question(qid, qdef)
            q = Agent._to_internal(qdef)
            refs = question_refs(qdef, state)
            row_state = refs if refs else state
            key = tuple(refs) if refs else None
            state_ids = state_ids_by_refs.get(key)
            if state_ids is None:
                state_ids = encode_text(
                    tokenizer, serialize_state(row_state).replace(mask, " "), add_special_tokens=False
                )["input_ids"]
                state_ids_by_refs[key] = state_ids
            seq, markers = build_sequence(
                tokenizer, row_state, q, max_len, head_max_len,
                truncate_left=isinstance(row_state, list), state_ids=state_ids,
            )
            if len(markers) != len(render_options(q)):
                raise ValueError(f"options exceed head_max_len={head_max_len}")
        except Exception as exc:
            raise RequestError(qid, str(exc)) from exc
        internal[qid] = q
        rows.append(Row(qid, np.asarray(seq, dtype=np.int32), markers, QTYPES[q["t"]]))
    return internal, rows


# --- engine worker process -----------------------------------------------------------------------


class EngineKilled(Exception):
    pass


class EngineError(Exception):
    pass


class EngineWorker:
    def __init__(self, settings: Settings):
        self.s = settings
        self.proc: Optional[subprocess.Popen] = None
        self.conn = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-engine-io")

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self.conn is not None

    async def start(self) -> None:
        s = self.s
        self._reap()
        loop = asyncio.get_running_loop()
        authkey = secrets.token_bytes(32)
        listener = Listener(("127.0.0.1", 0), authkey=authkey)
        host, port = listener.address
        env = dict(os.environ)
        env.update(
            JEV_WORKER_ADDRESS=f"{host}:{port}",
            JEV_WORKER_AUTHKEY=authkey.hex(),
            LAYA_OV_DEVICE=s.device,
            LAYA_OV_ONNX_PATH=s.onnx_path,
            LAYA_OV_CACHE_DIR=s.cache_dir,
            LAYA_OV_FORCE_FP32="1" if s.force_fp32 else "0",
            PYTHONUTF8="1",
        )
        proc = subprocess.Popen([sys.executable, str(WORKER_PATH)], env=env, cwd=str(ROOT))
        accept = loop.run_in_executor(None, listener.accept)
        deadline = time.monotonic() + s.worker_start_timeout_seconds
        try:
            while not accept.done():
                if proc.poll() is not None:
                    raise EngineError(f"worker exited with code {proc.returncode} before connecting")
                if time.monotonic() > deadline:
                    raise EngineError(f"worker did not connect within {s.worker_start_timeout_seconds:g}s")
                await asyncio.wait([accept], timeout=0.2)
            conn = accept.result()
        except BaseException:
            if proc.poll() is None:
                proc.kill()
            raise
        finally:
            listener.close()
        self.proc, self.conn = proc, conn
        log(f"engine worker started (pid {proc.pid})")

    async def call(self, message: tuple, timeout: float = 0.0):
        conn = self.conn
        if conn is None or not self.alive:
            raise EngineKilled("engine worker is not running")

        def io():
            conn.send(message)
            return conn.recv()

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self.pool, io)
        try:
            reply = await (asyncio.wait_for(future, timeout) if timeout > 0 else future)
        except asyncio.TimeoutError as exc:
            self.kill()
            raise EngineError(f"engine call {message[0]} exceeded {timeout:g}s; worker killed") from exc
        except (EOFError, OSError) as exc:
            self._reap()
            raise EngineKilled(f"engine worker died during {message[0]}") from exc
        if reply[0] == "error":
            raise EngineError(reply[1])
        return reply

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            try:
                # Reaped now, so `alive` is False on the next check instead of racing the exit.
                self.proc.wait(timeout=self.s.worker_stop_timeout_seconds)
            except subprocess.TimeoutExpired:
                pass

    async def stop(self) -> None:
        if self.proc is None:
            return
        loop = asyncio.get_running_loop()
        if self.proc.poll() is None and self.conn is not None:
            try:
                await loop.run_in_executor(self.pool, self.conn.send, ("exit",))
                await loop.run_in_executor(None, self.proc.wait, self.s.worker_stop_timeout_seconds)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._reap()

    def _reap(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
            self.conn = None
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
            try:
                self.proc.wait(timeout=self.s.worker_stop_timeout_seconds)
            except subprocess.TimeoutExpired:
                pass
            self.proc = None


# --- server: admission, micro-batching, watchdog -------------------------------------------------


class Server:
    def __init__(self, settings: Settings):
        self.s = settings
        self.worker = EngineWorker(settings)
        self.jobs: dict = {}
        self.queue: deque = deque()
        self.queued_rows = 0
        self.pending_rows = 0
        self.admitting = 0
        self.last_admit = 0.0
        self.wake = asyncio.Event()
        self.engine_lock = asyncio.Lock()
        self.encode_sem = asyncio.Semaphore(settings.encode_concurrency)
        self.loaded_len = 0
        self.load_info: Optional[dict] = None
        self.pass_started: Optional[float] = None
        self.pass_jobs: tuple = ()
        self.passes_done = 0
        self.tasks: list = []

    async def start(self) -> None:
        s = self.s
        if not Path(s.onnx_path).is_file():
            raise RuntimeError(f"ONNX model not found: {s.onnx_path}. Run export_laya_master.py first.")
        cfg = json.loads((Path(s.onnx_path).parent / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.head_max_len = cfg["head_max_len"]
        self.num_markers = cfg["max_markers"]
        self.temperature = cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = cfg.get("temperature_by_options", {})
        self.tokenizer = AutoTokenizer.from_pretrained(s.tokenizer_dir)
        self.pad_id = self.tokenizer.pad_token_id
        self.release_date = datetime.fromtimestamp(Path(s.onnx_path).stat().st_mtime).strftime("%Y-%m-%d")
        shape = s.max_len if s.max_len else f"auto (top {s.max_len_auto_percent:g}% truncated, cap {s.max_len_cap})"
        self.tasks = [asyncio.create_task(self._batcher()), asyncio.create_task(self._watchdog())]
        async with self.engine_lock:
            await self._ensure_loaded()
        log(f"server ready: device={s.device} rows_per_pass={s.rows_per_pass} max_len={shape}")

    async def shutdown(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.worker.stop()

    # -- body --------------------------------------------------------------------------------

    async def read_body(self, request: Request) -> bytes:
        limit = self.s.max_body_bytes
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise HTTPException(status_code=413, detail=f"body exceeds {limit} bytes")
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise HTTPException(status_code=413, detail=f"body exceeds {limit} bytes")
            chunks.append(chunk)
        return b"".join(chunks)

    def busy(self, reason: str) -> HTTPException:
        return HTTPException(
            status_code=429, detail=reason, headers={"Retry-After": str(self.s.retry_after_seconds)}
        )

    # -- admission ---------------------------------------------------------------------------

    async def admit(self, state: Any, questions: dict) -> Job:
        s = self.s
        rows_wanted = len(questions)
        if len(self.jobs) + self.admitting >= s.max_inflight_requests:
            raise self.busy("too many requests in flight")
        if self.pending_rows + rows_wanted > s.max_pending_rows:
            raise self.busy("too many rows pending")
        self.admitting += 1
        self.pending_rows += rows_wanted
        try:
            async with self.encode_sem:
                internal, rows = await asyncio.to_thread(
                    _encode_request, self.tokenizer, self.head_max_len, s.max_len or s.max_len_cap, state, questions
                )
        except RequestError as exc:
            self.pending_rows -= rows_wanted
            raise HTTPException(
                status_code=422,
                detail=[{"loc": ["body", "questions", exc.qid], "msg": exc.message, "type": "value_error"}],
            ) from exc
        except BaseException:
            self.pending_rows -= rows_wanted
            raise
        finally:
            self.admitting -= 1
        job = Job(state, questions, list(questions), internal, rows)
        self.jobs[job.id] = job
        self.queue.append(job)
        job.queued = True
        self.queued_rows += job.total
        self.last_admit = time.monotonic()
        self.wake.set()
        log(f"request {job.id}: {job.total} question(s), needs {job.needed_len} tokens")
        return job

    async def wait_for(self, job: Job, request: Request) -> None:
        poll = self.s.disconnect_poll_seconds
        while not job.done.is_set():
            try:
                await asyncio.wait_for(job.done.wait(), timeout=poll)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    self.finalize(job, 499, "client disconnected")

    def release(self, job: Job) -> None:
        self.finalize(job, 499, "request abandoned")
        self.jobs.pop(job.id, None)

    # -- job bookkeeping ---------------------------------------------------------------------

    def finalize(self, job: Job, status: Optional[int] = None, message: str = "") -> None:
        if job.closed:
            return
        job.closed = True
        self.queued_rows -= len(job.pending)
        job.pending.clear()
        self.pending_rows -= job.remaining
        job.remaining = 0
        if status is not None:
            job.error = (status, message)
        job.done.set()

    def cancel(self, job: Job, status: int, message: str) -> None:
        self.finalize(job, status, message)

    # -- micro-batching ----------------------------------------------------------------------

    def _take_rows(self) -> list:
        taken: list = []
        budget = self.s.rows_per_pass
        share = max(1, budget // max(1, len(self.queue)))
        while budget > 0 and self.queue:
            job = self.queue.popleft()
            job.queued = False
            if job.closed:
                continue
            count = min(share, len(job.pending), budget)
            for _ in range(count):
                taken.append((job, job.pending.popleft()))
            budget -= count
            self.queued_rows -= count
            if job.pending:
                self.queue.append(job)
                job.queued = True
        return taken

    async def _batcher(self) -> None:
        wait = self.s.batch_wait_ms / 1000.0
        while True:
            try:
                if not self.queue:
                    self.wake.clear()
                    await self.wake.wait()
                if wait > 0 and self.queued_rows < self.s.rows_per_pass:
                    await asyncio.sleep(wait)
                    # A pass costs the full padded shape: wait until arrivals go quiet or the pass is full.
                    deadline = time.monotonic() + self.s.batch_max_wait_ms / 1000.0
                    while self.queued_rows < self.s.rows_per_pass and time.monotonic() < deadline and (
                        self.admitting > 0 or time.monotonic() - self.last_admit < wait
                    ):
                        await asyncio.sleep(wait)
                taken = self._take_rows()
                if taken:
                    await self._run_pass(taken)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log(f"batcher error: {exc!r}")
                await asyncio.sleep(self.s.watchdog_interval_seconds)

    async def _measure_max_len(self) -> int:
        """Length covering every row but the largest `max_len_auto_percent`% of those queued."""
        s = self.s
        deadline = time.monotonic() + s.max_len_auto_wait_ms / 1000.0
        while True:
            lengths = sorted(len(row.ids) for job in self.jobs.values() if not job.closed for row in job.rows)
            if not lengths:
                return 0
            if len(lengths) >= s.max_len_auto_rows or time.monotonic() >= deadline:
                break
            await asyncio.sleep(s.batch_wait_ms / 1000.0)
        covered = max(0.0, min(100.0, 100.0 - s.max_len_auto_percent))
        rank = max(1, math.ceil(covered / 100.0 * len(lengths)))
        target = min(s.max_len_cap, math.ceil(lengths[rank - 1] / s.max_len_step) * s.max_len_step)
        log(
            f"measured {len(lengths)} row(s): longest={lengths[-1]}, p{covered:g}={lengths[rank - 1]}, "
            f"compiling max_len={target}"
        )
        return target

    async def _ensure_loaded(self) -> None:
        s = self.s
        if self.worker.alive and self.loaded_len:
            return
        self.loaded_len = 0
        self.load_info = None
        target = s.max_len or await self._measure_max_len()
        if not target:
            return
        if not self.worker.alive:
            await self.worker.start()
        log(f"compiling model for max_len={target}, rows_per_pass={s.rows_per_pass}")
        reply = await self.worker.call(
            ("load", s.rows_per_pass, target, self.num_markers), timeout=s.load_timeout_seconds
        )
        self.loaded_len = target
        self.load_info = reply[1]
        log(f"model ready: {reply[1]}")

    async def _fit_jobs(self, jobs: list) -> None:
        """Re-encode what the compiled length cannot hold, so the state is cut and the options never are."""
        for job in jobs:
            if job.trimmed or job.needed_len <= self.loaded_len:
                continue
            before = job.needed_len
            internal, rows = await asyncio.to_thread(
                _encode_request, self.tokenizer, self.head_max_len, self.loaded_len, job.state, job.questions
            )
            job.internal, job.rows = internal, rows
            job.measure()
            job.trimmed = True
            log(f"request {job.id}: state truncated from {before} to {job.needed_len} tokens")

    async def _run_pass(self, taken: list) -> None:
        s = self.s
        async with self.engine_lock:
            live = [(job, i) for job, i in taken if not job.closed]
            if not live:
                return
            jobs = list({id(job): job for job, _ in live}.values())
            try:
                await self._ensure_loaded()
                await self._fit_jobs(jobs)
                items = [
                    {"ids": job.rows[i].ids, "markers": job.rows[i].markers, "qtype": job.rows[i].qtype}
                    for job, i in live
                ]
                feeds = _build_fixed_shape_feeds(items, self.pad_id, s.rows_per_pass, self.loaded_len, self.num_markers)
                self.pass_started = started = time.monotonic()
                self.pass_jobs = tuple(jobs)
                try:
                    reply = await self.worker.call(("infer", feeds))
                finally:
                    self.pass_started = None
                    self.pass_jobs = ()
                log(
                    f"pass {self.passes_done + 1}: {len(live)}/{s.rows_per_pass} rows, "
                    f"{len(jobs)} request(s), {time.monotonic() - started:.2f}s"
                )
            except EngineKilled as exc:
                log(f"pass interrupted: {exc}")
                self._requeue(live)
                return
            except Exception as exc:
                log(f"pass failed: {exc!r}")
                for job in jobs:
                    self.finalize(job, 500, f"engine failure: {exc}")
                return
        try:
            logits = reply[1]
            for position, (job, i) in enumerate(live):
                if job.closed:
                    continue
                row = job.rows[i]
                decoded = _decode_ticket_answers(
                    logits, [{"markers": row.markers}], [row.qid], {row.qid: job.internal[row.qid]},
                    position, self.temperature, self.temperature_by_options,
                )
                job.answers[row.qid] = decoded[row.qid]
                job.remaining -= 1
                self.pending_rows -= 1
                if job.remaining == 0:
                    self.finalize(job)
            self.passes_done += 1
        except Exception as exc:
            log(f"decode failed: {exc!r}")
            for job in jobs:
                self.finalize(job, 500, f"decode failure: {exc}")

    def _requeue(self, live: list) -> None:
        by_job: dict = {}
        for job, i in live:
            if not job.closed:
                by_job.setdefault(job, []).append(i)
        for job, indexes in by_job.items():
            job.retries += 1
            if job.retries > self.s.max_pass_retries:
                self.finalize(job, 503, "engine interrupted too many times")
                continue
            for i in reversed(indexes):
                job.pending.appendleft(i)
            self.queued_rows += len(indexes)
            if not job.queued:
                self.queue.append(job)
                job.queued = True
        self.wake.set()

    # -- watchdog and admin ------------------------------------------------------------------

    async def _watchdog(self) -> None:
        s = self.s
        while True:
            await asyncio.sleep(s.watchdog_interval_seconds)
            now = time.monotonic()
            if s.request_timeout_seconds > 0:
                for job in list(self.jobs.values()):
                    if not job.closed and now - job.created > s.request_timeout_seconds:
                        log(f"request {job.id} timed out after {s.request_timeout_seconds:g}s")
                        self.cancel(job, 504, "request timeout")
            if s.pass_timeout_seconds > 0 and self.pass_started and now - self.pass_started > s.pass_timeout_seconds:
                log(f"forward pass exceeded {s.pass_timeout_seconds:g}s; killing the engine worker")
                self.worker.kill()

    def status(self) -> dict:
        now = time.monotonic()
        settings = asdict(self.s)
        requests = [
            {
                "id": job.id,
                "age_seconds": round(now - job.created, 1),
                "rows_total": job.total,
                "rows_done": job.total - job.remaining,
                "needed_len": job.needed_len,
                "retries": job.retries,
                "state": "cancelled" if job.error else ("done" if job.closed else "running"),
            }
            for job in list(self.jobs.values())[: self.s.status_max_requests]
        ]
        return {
            "engine": {
                "running": self.worker.alive,
                "pid": self.worker.proc.pid if self.worker.proc is not None else None,
                "max_len": self.loaded_len,
                "load": self.load_info,
                "pass_running_seconds": round(now - self.pass_started, 1) if self.pass_started else None,
                "passes_done": self.passes_done,
            },
            "queue": {
                "requests": len(self.jobs),
                "pending_rows": self.pending_rows,
                "queued_rows": self.queued_rows,
            },
            "requests": requests,
            "settings": settings,
        }

    async def unload(self, force: bool) -> dict:
        if force:
            for job in list(self.jobs.values()):
                self.cancel(job, 499, "cancelled by administrator")
            self.worker.kill()
        elif any(not job.closed for job in self.jobs.values()):
            raise HTTPException(status_code=409, detail="requests in progress; cancel them or use force")
        async with self.engine_lock:
            await self.worker.stop()
            self.loaded_len = 0
            self.load_info = None
        log("model unloaded")
        return {"unloaded": True}

    def cancel_requests(self, request_id: Optional[str], everything: bool, force: bool) -> dict:
        if everything:
            targets = [job for job in self.jobs.values() if not job.closed]
        else:
            job = self.jobs.get(request_id or "")
            if job is None:
                raise HTTPException(status_code=404, detail="unknown request_id")
            targets = [] if job.closed else [job]
        in_flight = any(job in self.pass_jobs for job in targets)
        for job in targets:
            self.cancel(job, 499, "cancelled by administrator")
        if force and in_flight:
            self.worker.kill()
        return {"cancelled": [job.id for job in targets], "worker_killed": bool(force and in_flight)}


# --- HTTP ------------------------------------------------------------------------------------------

SETTINGS = Settings.from_env()
server: Optional[Server] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global server
    SETTINGS.validate()
    server = Server(SETTINGS)
    await server.start()
    try:
        yield
    finally:
        await server.shutdown()


app = FastAPI(
    title="Jev-compatible Laya server",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    # FastAPI otherwise exports spans to OTEL_EXPORTER_OTLP_ENDPOINT, which VS Code sets in this shell.
    telemetry={"auto_configure": False},
)


def _unprocessable(loc: list, message: str) -> HTTPException:
    return HTTPException(status_code=422, detail=[{"loc": loc, "msg": message, "type": "value_error"}])


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {
        "models": [
            {
                "name": server.s.model_name,
                "description": "Laya System 1 decisions on the local OpenVINO engine.",
                "release_date": server.release_date,
            }
        ]
    }


@app.post("/v1/systemone")
async def systemone(request: Request):
    srv = server
    body = await srv.read_body(request)
    try:
        payload = await asyncio.to_thread(json.loads, body)
    except ValueError as exc:
        raise _unprocessable(["body"], f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise _unprocessable(["body"], "body must be a JSON object")
    for field in ("state", "questions"):
        if field not in payload:
            raise _unprocessable(["body", field], "Field required")
    state, questions, model = payload["state"], payload["questions"], payload.get("model")
    if not isinstance(state, (str, dict, list)):
        raise _unprocessable(["body", "state"], "state must be a string, object or array")
    if not isinstance(questions, dict) or not questions:
        raise _unprocessable(["body", "questions"], "questions must be a non-empty object")
    if len(questions) > srv.s.max_questions:
        raise _unprocessable(["body", "questions"], f"at most {srv.s.max_questions} questions per request")
    if model is not None and not isinstance(model, str):
        raise _unprocessable(["body", "model"], "model must be a string")

    started = time.monotonic()
    job = await srv.admit(state, questions)
    try:
        await srv.wait_for(job, request)
    finally:
        srv.release(job)
    if job.error is not None:
        status, message = job.error
        raise HTTPException(status_code=status, detail=message)
    result = {
        "model": srv.s.model_name,
        "answers": {qid: job.answers[qid] for qid in job.qids},
        # Not autoregressive: output_tokens is the answer count, input_tokens sums every row's tokens.
        "usage": {"input_tokens": job.input_tokens, "output_tokens": job.total},
    }
    log(f"request {job.id}: done in {time.monotonic() - started:.2f}s")
    content = await asyncio.to_thread(lambda: json.dumps(result, ensure_ascii=False).encode("utf-8"))
    return Response(content=content, media_type="application/json", headers={"x-request-id": job.id})


async def _admin_body(request: Request) -> dict:
    raw = await request.body()
    if not raw:
        return {}
    try:
        body = json.loads(raw)
    except ValueError as exc:
        raise _unprocessable(["body"], f"invalid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise _unprocessable(["body"], "body must be a JSON object")
    return body


@app.get("/admin/status")
async def admin_status():
    return server.status()


@app.post("/admin/unload")
async def admin_unload(request: Request):
    body = await _admin_body(request)
    return await server.unload(bool(body.get("force", False)))


@app.post("/admin/cancel")
async def admin_cancel(request: Request):
    body = await _admin_body(request)
    everything = bool(body.get("all", False))
    request_id = body.get("request_id")
    if not everything and not isinstance(request_id, str):
        raise _unprocessable(["body"], "give request_id or all=true")
    return server.cancel_requests(request_id, everything, bool(body.get("force", False)))


if __name__ == "__main__":
    SETTINGS.validate()
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, backlog=SETTINGS.backlog, log_level=SETTINGS.log_level)
