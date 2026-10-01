"""`laya-serve` on the local OpenVINO engine: Laya's own HTTP server, our model behind it.

    .venv\\Scripts\\python.exe laya_serve_openvino.py
    .venv\\Scripts\\python.exe laya_serve_openvino.py --batch-items 150
    .venv\\Scripts\\python.exe laya_serve_openvino.py --batch off --port 8001

Two modes, chosen by --batch / JEV_BATCH:

  off   `laya.serve.create_app(router)`, the stock application, untouched. One request is one
        forward pass, requests are serialised, and nothing here changes that.
  on    the same two routes, in this file, with a batcher in front of the engine. A request parks
        in a queue; one consumer takes everything that arrived inside the fill window, up to
        `--batch-states` states, and hands the lot to `Router.predict_batch`, whose
        `states x questions` items become one forward pass.

A third route, `POST /v1/systemone/batch`, takes many states in one request:
`{model, states: [...], questions, batch_size}` answers
`{model, results: [{answers, usage}, ...], usage}`, with one result per state in the order they
were sent. It is what laya-client.ps1 uses, so a file of a thousand states costs one request.
`batch_size` (optional, a positive integer) is how many states the client wants per forward pass;
a request without one uses `--batch-states`.

The contract is identical in both modes: `GET /health` and `POST /v1/systemone` taking
`{model, state, questions}` and answering `{model, answers, usage}`. Answers are unchanged,
`max_len` / `head_max_len` are accepted and ignored (our agent compiles one length, and
recompiling per request is exactly what it avoids), and validation, admission, auth and error
mapping are taken from `laya.serve` rather than rewritten.

Batching only pays with several requests in flight at once, so the client has to send them
concurrently. In `on` mode the figures that matter are items per forward pass and the number of
passes, not the latency of one request.

The Router only knows three names, `english`, `multilingual` and `typed-decisions`, so the agent
is attached under all three and set as the default. Every request shape -- an explicit `model=`, a
detected language, or nothing at all -- then lands on our engine, and no PyTorch checkpoint is ever
downloaded or loaded.

Every setting has a command-line flag and an environment variable, the flag winning:

  --batch on|off          batch requests into shared forward passes   (JEV_BATCH, default on)
  --batch-states N        states per forward pass for a request with
                          no batch_size of its own                   (JEV_BATCH_STATES, default 50)
  --window-ms N           how long a batch waits to fill              (JEV_BATCH_WINDOW_MS, default 20)
  --max-concurrent N      requests admitted at once; beyond it the
                          server answers 503 with Retry-After       (LAYA_MAX_CONCURRENT, default 2)
  --max-len N             compiled sequence length for every question (LAYA_OV_MAX_LEN, default 128)
  --host H, --port N      where to listen                             (LAYA_HOST, LAYA_PORT)
  LAYA_LOG_LEVEL          uvicorn log level                           (default info)
  LAYA_ACCESS_LOG         uvicorn's one line per request              (default off)
  LAYA_API_KEY            bearer token, unchecked when unset
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

# FastAPI's own app does not turn the OTLP auto-configuration off, and the VS Code terminal
# exports OTEL_EXPORTER_OTLP_ENDPOINT, which makes it try and fail at startup. OTEL_SDK_DISABLED
# is the SDK's documented off switch; set it before the app is built, keep any user value.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from laya import serve as stock
from laya.mcp.device import agent_device, resolve_device, router_agent
from laya.router import Router

from laya_openvino_agent import OpenVINOAgent

# Every name the Router can route to, all pointing at the same agent.
ROUTER_NAMES = ("english", "multilingual", "typed-decisions")
DEFAULT_NAME = "multilingual"


def build_router(max_len: Optional[int] = None) -> Router:
    agent = OpenVINOAgent(max_len=max_len)
    # No `models=` map: the three names already exist, and `load()` returns an attached agent
    # before it would ever look a repository up.
    router = Router(default=DEFAULT_NAME, max_loaded=len(ROUTER_NAMES), preload=False)
    for name in ROUTER_NAMES:
        router.attach(name, agent)
    return router


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "on", "yes")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def build_app(router: Router, batching: bool, window_ms: int, batch_states: int, max_concurrent: int):
    """The stock application when batching is off, ours when it is on."""
    if not batching:
        print("laya-serve/OpenVINO: batching off, stock app, one request per forward pass", flush=True)
        return stock.create_app(router)
    print(
        f"laya-serve/OpenVINO: batching on, {batch_states} states per forward pass "
        f"({batch_states} x questions items), {window_ms} ms fill window, "
        f"{max_concurrent} requests admitted at once",
        flush=True,
    )
    return _build_batched_app(router, window_ms, batch_states, max_concurrent)


def _build_batched_app(router: Router, window_ms: int, batch_states: int,
                       max_concurrent: int) -> FastAPI:
    api_key = os.environ.get("LAYA_API_KEY") or None
    expected_auth = ("Bearer " + api_key).encode("utf-8", "surrogateescape") if api_key else b""
    admission_limit = max_concurrent

    # One worker: a forward pass at a time is what this engine wants, and the batcher is the only
    # thing that hands it work.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-infer")
    queue: asyncio.Queue = asyncio.Queue()
    admission: Optional[asyncio.Semaphore] = None
    # Requests seen and items sent, counted per HTTP request. Forward passes are counted by the
    # agent, which is the only place that knows how many the call really became.
    passes = {"requests": 0, "items": 0}

    def _check_auth(authorization: Optional[str]) -> None:
        if api_key is None:
            return
        supplied = (authorization or "").encode("utf-8", "surrogateescape")
        if not stock.hmac.compare_digest(supplied, expected_auth):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    async def _run_batch(batch: list) -> None:
        """One forward pass for the whole batch; every parked request gets its own result."""
        requests = [
            {"state": state, "questions": questions, "model": model}
            for state, questions, model, _ in batch
        ]
        items = sum(len(questions) for _, questions, _, _ in batch)
        started = time.perf_counter()
        try:
            results = await asyncio.get_running_loop().run_in_executor(
                pool, lambda: router.predict_batch(requests, batch_size=batch_states)
            )
        except BaseException as exc:  # includes CancelledError on shutdown
            for *_, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return
        seconds = time.perf_counter() - started
        ms = seconds * 1000.0
        passes["requests"] += 1
        passes["items"] += items
        print(
            f"request: {len(batch)} state(s), {items} items, {seconds:.2f}s",
            file=sys.stderr, flush=True,
        )
        for (*_, future), result in zip(batch, results):
            if not future.done():
                future.set_result((result, ms))

    async def _batcher() -> None:
        while True:
            batch = [await queue.get()]
            # Fill the pass with requests already waiting, up to batch_states states. A request
            # arriving alone waits window_ms and then goes on its own, which is the only latency
            # this mode adds.
            deadline = time.monotonic() + window_ms / 1000.0
            while len(batch) < batch_states:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
            await _run_batch(batch)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(_batcher())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(
        title="laya-serve (OpenVINO, batched)",
        summary="Laya System-1 decisions over the TypeSafe Jev /v1/systemone protocol",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        devices = {}
        for name in router.loaded or []:
            device = agent_device(router_agent(router, name))
            if device:
                devices[name] = device
        return {
            "status": "ok",
            "loaded": router.loaded,
            "device": next(iter(devices.values()), None) or resolve_device(),
            "checkpoint_devices": devices,
            "batching": {
                "enabled": True,
                "window_ms": window_ms,
                "states_per_forward_pass": batch_states,
                "requests_admitted": admission_limit,
                "requests_served": passes["requests"],
                "forward_passes": getattr(router_agent(router, DEFAULT_NAME), "pass_count", None),
                "items_through_engine": passes["items"],
                "queued": queue.qsize(),
            },
        }

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
        nonlocal admission
        _check_auth(authorization)
        if admission is None:
            admission = asyncio.Semaphore(admission_limit)
        if admission.locked():
            raise HTTPException(status_code=503, detail="server busy, try again later",
                                headers={"Retry-After": "1"})
        await admission.acquire()
        try:
            return await _systemone_inner(request)
        finally:
            admission.release()

    async def _systemone_inner(request: Request):
        if request.headers.get("content-length"):
            try:
                if int(request.headers["content-length"]) > stock.MAX_BODY_BYTES:
                    raise HTTPException(status_code=413, detail="request body too large")
            except ValueError:
                pass
        raw = await stock._read_body_capped(request)
        try:
            body = stock.json.loads(raw)
        except (ValueError, RecursionError):
            raise HTTPException(status_code=400, detail="request body must be valid JSON")
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(
                status_code=400, detail="request body must be an object with a 'questions' field"
            )
        state = body.get("state")
        questions = body["questions"]
        stock._check_request_limits(state, questions)
        model = stock._resolve_model(body.get("model"))
        # Accepted, validated, then ignored: the token budget is fixed at compile time, so
        # honouring these per request would mean recompiling per request.
        cap = stock._resolve_max_token_budget()
        stock._validate_budget_param(body, "max_len", cap)
        stock._validate_budget_param(body, "head_max_len", cap)

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        await queue.put((state, questions, model, future))
        try:
            result, ms = await future
        except HTTPException:
            raise
        except ValueError as exc:
            # Question validation errors name the question and what to fix: safe for clients.
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception:
            stock._log.exception("inference failed for model=%s",
                                 str(model).replace("\n", "\\n").replace("\r", "\\r"))
            raise HTTPException(status_code=500, detail="inference failed")
        return JSONResponse(
            content=result,
            headers={
                "Server-Timing": f"inference;dur={ms:.2f}",
                "X-Inference-Time-Ms": f"{ms:.2f}",
            },
        )

    @app.post("/v1/systemone/batch")
    async def systemone_batch(request: Request, authorization: Optional[str] = Header(default=None)):
        """One request carrying many states, answered in forward passes of `--batch-states`."""
        nonlocal admission
        _check_auth(authorization)
        if admission is None:
            admission = asyncio.Semaphore(admission_limit)
        if admission.locked():
            raise HTTPException(status_code=503, detail="server busy, try again later",
                                headers={"Retry-After": "1"})
        await admission.acquire()
        try:
            return await _systemone_batch_inner(request)
        finally:
            admission.release()

    async def _systemone_batch_inner(request: Request):
        if request.headers.get("content-length"):
            try:
                if int(request.headers["content-length"]) > stock.MAX_BODY_BYTES:
                    raise HTTPException(status_code=413, detail="request body too large")
            except ValueError:
                pass
        raw = await stock._read_body_capped(request)
        try:
            body = stock.json.loads(raw)
        except (ValueError, RecursionError):
            raise HTTPException(status_code=400, detail="request body must be valid JSON")
        if not isinstance(body, dict) or "states" not in body or "questions" not in body:
            raise HTTPException(
                status_code=400,
                detail="request body must be an object with 'states' and 'questions' fields",
            )
        states = body["states"]
        questions = body["questions"]
        if not isinstance(states, list) or not states:
            raise HTTPException(status_code=400, detail="'states' must be a non-empty array")
        if not all(isinstance(state, (str, dict, list)) for state in states):
            raise HTTPException(
                status_code=400, detail="every state must be a string, an object or an array"
            )
        for state in states:
            stock._check_request_limits(state, questions)
        raw_batch_size = body.get("batch_size")
        if raw_batch_size is None:
            batch_size = batch_states
        elif isinstance(raw_batch_size, bool) or not isinstance(raw_batch_size, int) or raw_batch_size < 1:
            raise HTTPException(status_code=400, detail="'batch_size' must be a positive integer")
        else:
            batch_size = raw_batch_size
        model = stock._resolve_model(body.get("model"))
        requests = [{"state": state, "questions": questions, "model": model} for state in states]

        started = time.perf_counter()
        try:
            # Laya splits the request's states into forward passes of the client's batch_size.
            results = await asyncio.get_running_loop().run_in_executor(
                pool, lambda: router.predict_batch(requests, batch_size=batch_size)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception:
            stock._log.exception("batch inference failed for model=%s",
                                 str(model).replace("\n", "\\n").replace("\r", "\\r"))
            raise HTTPException(status_code=500, detail="inference failed")
        seconds = time.perf_counter() - started
        ms = seconds * 1000.0
        items = len(states) * len(questions)
        passes["requests"] += 1
        passes["items"] += items
        print(
            f"batch request: {len(states)} state(s), {items} items, {seconds:.2f}s",
            file=sys.stderr, flush=True,
        )

        input_tokens = sum(int(r.get("usage", {}).get("input_tokens", 0)) for r in results)
        output_tokens = sum(int(r.get("usage", {}).get("output_tokens", 0)) for r in results)
        return JSONResponse(
            content={
                "model": results[0].get("model") if results else DEFAULT_NAME,
                "results": [
                    {"answers": r.get("answers", {}), "usage": r.get("usage", {})}
                    for r in results
                ],
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            },
            headers={
                "Server-Timing": f"inference;dur={ms:.2f}",
                "X-Inference-Time-Ms": f"{ms:.2f}",
            },
        )

    return app


def _parse_args(argv: Optional[list] = None):
    """Command line first, environment second, defaults last."""
    import argparse

    parser = argparse.ArgumentParser(
        description="laya-serve on the local OpenVINO engine",
    )
    parser.add_argument("--batch", choices=("on", "off"), default=None,
                        help=f"batch requests into shared forward passes (env JEV_BATCH, "
                             f"default {'on' if _env_bool('JEV_BATCH', True) else 'off'})")
    parser.add_argument("--batch-states", type=int, default=None,
                        help="states per forward pass "
                             f"(env JEV_BATCH_STATES, default {_env_int('JEV_BATCH_STATES', 50)})")
    parser.add_argument("--window-ms", type=int, default=None,
                        help="how long a batch waits for more requests "
                             f"(env JEV_BATCH_WINDOW_MS, default {_env_int('JEV_BATCH_WINDOW_MS', 20)})")
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="requests admitted at once; beyond it the server answers 503 "
                             f"(env LAYA_MAX_CONCURRENT, default {_env_int('LAYA_MAX_CONCURRENT', 2)})")
    parser.add_argument("--max-len", type=int, default=None,
                        help="compiled sequence length for every question; the question and its "
                             "options share this budget with the state, so a question with little "
                             "room left truncates the state, possibly to nothing "
                             f"(env LAYA_OV_MAX_LEN, default {_env_int('LAYA_OV_MAX_LEN', 128)})")
    parser.add_argument("--host", default=None,
                        help=f"bind address (env LAYA_HOST, default {os.environ.get('LAYA_HOST', '127.0.0.1')})")
    parser.add_argument("--port", type=int, default=None,
                        help=f"port (env LAYA_PORT, default {os.environ.get('LAYA_PORT', '8001')})")
    return parser.parse_args(argv)


def main() -> None:
    import uvicorn

    args = _parse_args()
    batching = _env_bool("JEV_BATCH", True) if args.batch is None else args.batch == "on"
    window_ms = args.window_ms or _env_int("JEV_BATCH_WINDOW_MS", 20)
    batch_states = args.batch_states or _env_int("JEV_BATCH_STATES", 50)
    max_concurrent = args.max_concurrent or _env_int("LAYA_MAX_CONCURRENT", 2)
    max_len = args.max_len or _env_int("LAYA_OV_MAX_LEN", 128)
    if batch_states < 1 or window_ms < 0 or max_concurrent < 1 or max_len < 1:
        raise SystemExit("--batch-states, --max-concurrent and --max-len must be at least 1, "
                         "and --window-ms cannot be negative")

    uvicorn.run(
        build_app(build_router(max_len=max_len), batching, window_ms, batch_states, max_concurrent),
        host=args.host or os.environ.get("LAYA_HOST", "127.0.0.1"),
        port=args.port or int(os.environ.get("LAYA_PORT", "8001")),
        log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
        # One line per request buries the forward pass lines; LAYA_ACCESS_LOG=1 puts them back.
        access_log=_env_bool("LAYA_ACCESS_LOG", False),
    )


if __name__ == "__main__":
    main()
