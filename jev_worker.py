"""Inference worker started by jev_server.py; it alone owns the OpenVINO model so the server can kill it.

Protocol: pickled tuples over an authenticated loopback connection.
  ("load", rows, max_len, num_markers) -> ("loaded", info)
  ("infer", feeds)                     -> ("logits", ndarray)
  ("exit",)                            -> the process ends
A failed request is answered with ("error", message).
"""
import gc
import os
import sys
from multiprocessing.connection import Client
from time import perf_counter

import numpy as np
import openvino as ov


def _log(message):
    print(f"[worker] {message}", file=sys.stderr, flush=True)


def _compile(core, device, onnx_path, cache_dir, rows, max_len, num_markers):
    available = core.available_devices
    if device == "GPU" and "GPU" not in available:
        raise RuntimeError(f"OpenVINO device 'GPU' requested but not available: {', '.join(available)}")
    blobs_before = set(os.listdir(cache_dir))
    started = perf_counter()
    model = core.read_model(onnx_path)
    # One static shape keeps the GPU plugin from JIT-compiling kernels per shape and lets CACHE_DIR be reused.
    model.reshape({
        "input_ids": ov.PartialShape([rows, max_len]),
        "attention_mask": ov.PartialShape([rows, max_len]),
        "marker_pos": ov.PartialShape([rows, num_markers]),
        "marker_mask": ov.PartialShape([rows, num_markers]),
        "qtype": ov.PartialShape([rows]),
    })
    compiled = core.compile_model(model, device)
    del model
    seconds = perf_counter() - started
    try:
        precision = str(compiled.get_property("INFERENCE_PRECISION_HINT"))
    except Exception:
        precision = "unknown"
    try:
        actual_device = str(compiled.get_property("EXECUTION_DEVICES"))
    except Exception:
        actual_device = device
    new_blobs = len(set(os.listdir(cache_dir)) - blobs_before)
    info = {
        "rows": rows,
        "max_len": max_len,
        "compile_seconds": round(seconds, 2),
        "device": actual_device,
        "precision": precision,
        "cache": f"MISS, wrote {new_blobs} blob(s)" if new_blobs else "HIT",
    }
    return compiled, info


def main():
    host, port = os.environ["JEV_WORKER_ADDRESS"].rsplit(":", 1)
    conn = Client((host, int(port)), authkey=bytes.fromhex(os.environ["JEV_WORKER_AUTHKEY"]))
    device = os.environ["LAYA_OV_DEVICE"]
    onnx_path = os.environ["LAYA_OV_ONNX_PATH"]
    cache_dir = os.environ["LAYA_OV_CACHE_DIR"]
    force_fp32 = os.environ.get("LAYA_OV_FORCE_FP32", "0").strip() == "1"

    core = ov.Core()
    os.makedirs(cache_dir, exist_ok=True)
    core.set_property({"CACHE_DIR": cache_dir})
    if force_fp32:
        # The GPU plugin defaults to f16 even for an fp32 export; this restores true fp32 compute.
        core.set_property(device, {"INFERENCE_PRECISION_HINT": ov.Type.f32})

    compiled = None
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            break
        kind = message[0]
        if kind == "exit":
            break
        try:
            if kind == "load":
                compiled = None
                gc.collect()
                compiled, info = _compile(core, device, onnx_path, cache_dir, *message[1:])
                reply = ("loaded", info)
            elif kind == "infer":
                if compiled is None:
                    raise RuntimeError("no model loaded")
                reply = ("logits", np.array(compiled(message[1])["logits"]))
            else:
                raise ValueError(f"unknown message {kind!r}")
        except Exception as exc:
            _log(f"{kind} failed: {type(exc).__name__}: {exc}")
            reply = ("error", f"{type(exc).__name__}: {exc}")
        try:
            conn.send(reply)
        except OSError:
            break


if __name__ == "__main__":
    main()
