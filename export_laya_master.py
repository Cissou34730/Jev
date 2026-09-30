"""Export one reusable, OpenVINO/Intel-GPU-compatible ONNX master model from the official
Laya multilingual checkpoint (source of truth: `laya.load(...)`, not a community ONNX port).

Why a custom forward instead of patching laya.common._DynamicMultiheadAttention:
nn.TransformerEncoderLayer.forward has a fused native fast path
(torch._transformer_encoder_layer_fwd) that bypasses self_attn.forward()/_sa_block entirely
whenever eval mode + batch_first + bias hold (confirmed with an instrumented call-counter: 0
calls into a patched self_attn.forward with the fast path enabled). Even with that fast path
disabled, PyTorch's own F._canonical_mask converts a bool key_padding_mask to a float mask via
`masked_fill_`, which decomposes to the same Where/GatherND-on-u8 pattern OpenVINO's Intel GPU
plugin refuses to compile (verified: "[GPU] No layout format available for gathernd:...,
data_type: u8"). So this module never calls nn.TransformerEncoderLayer/nn.MultiheadAttention at
all: it re-runs the trained head's own submodules (self_attn.in_proj_weight/out_proj, norm1/2,
linear1/2) directly, building the padding mask as a plain additive float bias
`(1 - attention_mask) * -1e9` -- no bool/u8 tensor ever exists in the traced graph. Marker
masking is rewritten the same way (arithmetic bias instead of `logits.masked_fill(~mask, ...)`).
This is a from-source export: same weights, same maths (verified against the checkpoint's own
unmodified forward below), different (GPU-safe) wiring.

Marker/option count: fixed at MAX_MARKERS=5 (choice supports 2-5 options, score 3+ levels, noul
always 2) so the ONNX graph never needs a boolean-driven dynamic gather over ragged option
counts; a question with fewer than 5 options pads marker_pos with 0 and marker_mask with 0.0 in
the unused slots.

Dynamic axes: only batch ("n") and sequence length ("L") stay dynamic in the master ONNX --
enough to let a deployment step (see openvino_deploy.py) reshape to whatever static length it
wants (128/192/256/384/512/...) before compiling for the GPU. The marker/option axis is a fixed
size (5), never dynamic.

Run: .venv\\Scripts\\python.exe export_laya_master.py [--out .laya-cache\\checkpoints\\laya-master]
Needs: torch, laya, onnx, onnxruntime, openvino.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import laya
from laya.common import QTYPES, build_sequence

MAX_MARKERS = 5


def load_agent(repo, subfolder):
    try:
        return laya.load(repo, device="cpu", subfolder=subfolder)
    except Exception as exc:
        if subfolder is None and repo == "convaiinnovations/laya-multilingual":
            print(f"laya.load({repo!r}) failed ({exc}); falling back to "
                  "laya.load('convaiinnovations/laya', subfolder='multilingual')", file=sys.stderr)
            return laya.load("convaiinnovations/laya", device="cpu", subfolder="multilingual")
        raise


QUESTIONS = {
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "department_2": {"type": "choice", "instructions": "Billing or technical issue?",
                     "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages"}},
    "department_3": {"type": "choice", "instructions": "Which department should handle this?",
                     "criteria": {"billing": "invoices, payments", "technical": "bugs, outages", "sales": "pricing"}},
    "department_4": {"type": "choice", "instructions": "Which department should handle this?",
                     "criteria": {"billing": "invoices", "technical": "bugs", "sales": "pricing", "other": "everything else"}},
    "department_5": {"type": "choice", "instructions": "Which department should handle this?",
                     "criteria": {"billing": "invoices", "technical": "bugs", "sales": "pricing",
                                   "legal": "contracts, compliance", "other": "everything else"}},
    "urgency_3": {"type": "score", "instructions": "How urgent is this request?",
                 "criteria": ["not urgent", "soon", "critical"]},
    "urgency_4": {"type": "score", "instructions": "How urgent is this request?",
                 "criteria": ["not urgent", "soon", "urgent", "critical"]},
}

STATE = {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
         "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."}


def encode_items(agent, state, questions):
    items = []
    for qdef in questions.values():
        q = agent._to_internal(qdef)
        seq, markers = build_sequence(agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
        if len(markers) > MAX_MARKERS:
            raise ValueError(f"question has {len(markers)} options; MAX_MARKERS={MAX_MARKERS} exceeded")
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    return items


def collate_fixed(items, pad_id):
    n = len(items)
    L = max(len(it["ids"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, MAX_MARKERS), dtype=torch.long)
    mmask = torch.zeros((n, MAX_MARKERS), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = 1.0
    qtype = torch.tensor([it["qtype"] for it in items])
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "qtype": qtype}


def batch_for(agent, state, questions):
    return collate_fixed(encode_items(agent, state, questions), agent.tok.pad_token_id)


def _encoder_attn_mask_dict(attention_mask, config, dtype):
    """Pre-built float additive attention-mask dict for `transformers`' ModernBERT-family encoder
    (mmBERT reuses `modeling_modernbert.py`).

    Root cause of the GPU GatherND/u8 failure (traced through `transformers/masking_utils.py`):
    `ModernBertModel.forward` builds its masks via `create_bidirectional_mask`/
    `create_bidirectional_sliding_window_mask` -> `sdpa_mask`, which ANDs in
    `padding_mask_function(padding_mask)` whose `inner_mask` does
    `return padding_mask[batch_idx, kv_idx]` -- advanced/fancy indexing with index tensors on a
    bool 2D mask (`_preprocess_mask_arguments` casts it via `.to(dtype=torch.bool)`). That
    indexing is what decomposes into the unsupported `GatherND` node on u8 data (shape
    `(n,1,1,L)`, i.e. the padding mask before its later `.expand()` to `(n,1,L,L)`) -- this is a
    *different* node than the Laya head's own attention/marker masking (already fixed above), and
    exists regardless of any patch to `laya.common`.

    Fix: build the same masks ourselves with pure elementwise arithmetic (comparisons + casts on
    `torch.arange` position grids, never indexing a data tensor by another tensor) and hand them
    to the encoder as an already-4D dict; `ModernBertModel.forward` takes a dict verbatim
    (`if not isinstance(attention_mask_mapping := attention_mask, dict): ...`), so this bypasses
    `masking_utils` entirely -- no GatherND, no bool/u8 tensor anywhere in the traced graph.
    """
    n, L = attention_mask.shape
    device = attention_mask.device
    min_value = torch.finfo(dtype).min
    pad_bias = ((1.0 - attention_mask.to(dtype)) * min_value)[:, None, None, :].expand(n, 1, L, L)
    masks = {"full_attention": pad_bias}
    if "sliding_attention" in set(getattr(config, "layer_types", None) or ()):
        window = config.sliding_window
        q_idx = torch.arange(L, device=device)[:, None]
        kv_idx = torch.arange(L, device=device)[None, :]
        out_of_window = ((q_idx - kv_idx).abs() > window).to(dtype) * min_value
        window_bias = out_of_window[None, None, :, :].expand(n, 1, L, L)
        masks["sliding_attention"] = torch.minimum(pad_bias, window_bias)
    return masks


class GPUFriendlyDecisionModel(torch.nn.Module):
    """Bit-for-bit re-expression of laya.common.DecisionModel.forward's head loop and marker
    masking, using only additive float tensors -- reuses the trained model's own submodules and
    weights verbatim, never calling nn.TransformerEncoderLayer/nn.MultiheadAttention.forward."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        m = self.m
        enc_dtype = next(m.encoder.parameters()).dtype
        enc_masks = _encoder_attn_mask_dict(attention_mask, m.encoder.config, enc_dtype)
        h = m.encoder(input_ids=input_ids, attention_mask=enc_masks).last_hidden_state
        h = h + m.type_emb(qtype)[:, None, :]
        if m.head is not None:
            attn_bias = (1.0 - attention_mask.to(h.dtype))[:, None, None, :] * -1e9
            for layer in m.head.layers:
                sa = layer.self_attn
                x = layer.norm1(h)
                q, k, v = F.linear(x, sa.in_proj_weight, sa.in_proj_bias).chunk(3, dim=-1)
                q = q.unflatten(-1, (sa.num_heads, sa.head_dim)).permute(0, 2, 1, 3)
                k = k.unflatten(-1, (sa.num_heads, sa.head_dim)).permute(0, 2, 1, 3)
                v = v.unflatten(-1, (sa.num_heads, sa.head_dim)).permute(0, 2, 1, 3)
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0)
                attn = attn.permute(0, 2, 1, 3).flatten(-2)
                h = h + sa.out_proj(attn)
                y = layer.norm2(h)
                y = layer.linear2(layer.activation(layer.linear1(y)))
                h = h + y
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        gathered = torch.gather(h, 1, idx)
        logits = m.scorer(gathered).squeeze(-1).float()
        # Exact arithmetic equivalent of `logits.masked_fill(~marker_mask, -1e4)` (a value
        # *replacement*, not an additive bias) -- `logits + (mask-1)*1e4` was wrong here because
        # it only shifts the padded slot's already-arbitrary raw logit, instead of pinning it to
        # -1e4 like masked_fill does; harmless post-softmax but it broke the raw-logit parity
        # check against the checkpoint's own forward.
        logits = logits * marker_mask + (-1e4) * (1.0 - marker_mask)
        p = torch.softmax(logits.detach(), -1)
        kf = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(kf)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, kf / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = m.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def copy_sidecars(agent, out_dir):
    tok = agent.tok
    tmp = os.path.join(out_dir, "_tok")
    tok.save_pretrained(tmp)
    shutil.copy(os.path.join(tmp, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))
    cfg = {k: getattr(tok, k) for k in ("cls_token", "sep_token", "mask_token", "pad_token", "unk_token") if getattr(tok, k, None)}
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    shutil.rmtree(tmp, ignore_errors=True)
    rl_cfg = dict(agent.cfg)
    rl_cfg["max_markers"] = MAX_MARKERS
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(rl_cfg, f, indent=2)


def check_openvino_gpu(onnx_path, profiles):
    import openvino as ov
    core = ov.Core()
    if "GPU" not in core.available_devices:
        print("no GPU device visible to OpenVINO; skipping the GPU compile check", file=sys.stderr)
        return
    model = core.read_model(onnx_path)
    for batch, seq_len in profiles:
        m = model.clone()
        m.reshape({"input_ids": [batch, seq_len], "attention_mask": [batch, seq_len],
                   "marker_pos": [batch, MAX_MARKERS], "marker_mask": [batch, MAX_MARKERS], "qtype": [batch]})
        t0 = time.time()
        core.compile_model(m, "GPU")
        print(f"OpenVINO GPU compile OK: batch={batch} seq_len={seq_len} ({time.time() - t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="convaiinnovations/laya-multilingual")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--out", default=os.path.join(".laya-cache", "checkpoints", "laya-master"))
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--skip-gpu-check", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    agent = load_agent(args.repo, args.subfolder)
    model = agent.model.eval()
    print(f"loaded in {time.time() - t0:.1f}s (max_len {agent.cfg['max_len']}, head_max_len {agent.cfg['head_max_len']})")

    b = batch_for(agent, STATE, QUESTIONS)
    mmask_bool = b["marker_mask"].bool()
    with torch.no_grad():
        ref_logits, ref_act = model(b["input_ids"], b["attention_mask"], b["marker_pos"], mmask_bool, b["qtype"])

    gpu_model = GPUFriendlyDecisionModel(model)
    with torch.no_grad():
        gpu_logits, gpu_act = gpu_model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    d_logits = float((gpu_logits - ref_logits).abs().max())
    d_act = float((gpu_act - ref_act).abs().max())
    print(f"max|Δ| GPU-friendly forward vs official DecisionModel.forward: logits={d_logits:.2e} act={d_act:.2e}")
    if d_logits > 1e-3 or d_act > 1e-3:
        print("GPU-friendly forward does not match the official checkpoint's own forward; aborting", file=sys.stderr)
        sys.exit(2)

    names = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
    dyn = {"input_ids": {0: "n", 1: "L"}, "attention_mask": {0: "n", 1: "L"},
           "marker_pos": {0: "n"}, "marker_mask": {0: "n"}, "qtype": {0: "n"}}
    inputs = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    onnx_path = os.path.join(args.out, "laya.onnx")
    t0 = time.time()
    prog = torch.onnx.export(gpu_model, inputs, input_names=names, output_names=["logits", "act_logits"],
                             dynamic_shapes=dyn, dynamo=True, opset_version=args.opset)
    prog.save(onnx_path, external_data=False)
    if os.path.exists(onnx_path + ".data"):
        import onnx
        m = onnx.load(onnx_path, load_external_data=True)
        onnx.save_model(m, onnx_path, save_as_external_data=False)
        os.remove(onnx_path + ".data")
        del m
    print(f"exported in {time.time() - t0:.1f}s: {os.path.getsize(onnx_path) / 1e9:.2f} GB")

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {k: v.numpy() for k, v in b.items() if k in names}
    feed["input_ids"] = feed["input_ids"].astype(np.int64)
    feed["attention_mask"] = feed["attention_mask"].astype(np.int64)
    feed["marker_pos"] = feed["marker_pos"].astype(np.int64)
    feed["marker_mask"] = feed["marker_mask"].astype(np.float32)
    feed["qtype"] = feed["qtype"].astype(np.int64)
    onnx_logits, onnx_act = sess.run(None, feed)
    d_onnx = float(np.abs(onnx_logits - gpu_logits.numpy()).max())

    b1 = batch_for(agent, {"body": "Der Kunde wurde zweimal belastet und droht zu kündigen"},
                   {"churn_risk": QUESTIONS["churn_risk"], "department_5": QUESTIONS["department_5"]})
    feed1 = {"input_ids": b1["input_ids"].numpy().astype(np.int64), "attention_mask": b1["attention_mask"].numpy().astype(np.int64),
             "marker_pos": b1["marker_pos"].numpy().astype(np.int64), "marker_mask": b1["marker_mask"].numpy().astype(np.float32),
             "qtype": b1["qtype"].numpy().astype(np.int64)}
    with torch.no_grad():
        l1, _ = gpu_model(b1["input_ids"], b1["attention_mask"], b1["marker_pos"], b1["marker_mask"], b1["qtype"])
    lo1, _ = sess.run(None, feed1)
    d_onnx1 = float(np.abs(lo1 - l1.numpy()).max())
    print(f"max|Δlogits| onnx vs GPU-friendly PyTorch: batch={d_onnx:.2e} other-shape={d_onnx1:.2e}")
    if d_onnx > 1e-3 or d_onnx1 > 1e-3:
        print("ONNX output does not match the GPU-friendly PyTorch forward; aborting", file=sys.stderr)
        sys.exit(2)

    copy_sidecars(agent, args.out)
    print("sidecars written:", sorted(f for f in os.listdir(args.out) if not f.startswith("_")))

    if not args.skip_gpu_check:
        check_openvino_gpu(onnx_path, profiles=[(4, 128), (16, 256), (8, 384), (2, 512)])
    print("done")


if __name__ == "__main__":
    main()
