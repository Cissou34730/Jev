"""Export Laya's English checkpoint to an OpenVINO-GPU-compatible ONNX graph.

Adapted from mariojcr/laya-onnx's export_onnx.py (same source checkpoint, same dynamic n/L/k
axes, same opset). The upstream export represents the encoder's key-padding mask as a *boolean*
tensor fed into scaled_dot_product_attention; torch.onnx's dynamo exporter decomposes that into a
GatherND node operating on u8 data, which OpenVINO's Intel GPU plugin refuses to compile
(verified: "[GPU] No layout format available for gathernd:..., data_type: u8"). CPU has no such
restriction -- this is a GPU-plugin-specific gap in mariojcr's export, not a bug in our runner.

This script monkeypatches laya.common._DynamicMultiheadAttention to build an additive float mask
instead of a boolean one before calling scaled_dot_product_attention (same maths: PyTorch's SDPA
converts a bool mask to an additive -inf mask internally anyway, so this only changes what the
*exporter* sees, not what the model computes). It then verifies, in order:
  1. patched PyTorch forward vs the original (unpatched) PyTorch forward -- confirms the patch
     doesn't change the model's actual outputs;
  2. exported ONNX (onnxruntime CPU) vs the patched PyTorch forward -- confirms the export is
     faithful (same check torch.onnx does upstream);
  3. native OpenVINO reshape+compile on GPU at several (batch, max_len) profiles -- confirms this
     export actually solves the problem the patch was written for.

Run: .venv\\Scripts\\python.exe export_onnx_gpu.py [--out .laya-cache\\checkpoints\\laya-onnx-gpu]
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
from laya.common import QTYPES, _DynamicMultiheadAttention, build_sequence, collate_items

STATE = {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
         "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."}
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                                 "sales": "pricing, new contracts", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
}


def _patched_forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                      attn_mask=None, average_attn_weights=True, is_causal=False):
    # Verbatim copy of laya.common._DynamicMultiheadAttention.forward up to the SDPA call; only
    # the mask passed to scaled_dot_product_attention changes (bool -> additive float).
    if (need_weights or self.in_proj_weight is None or self.bias_k is not None
            or self.bias_v is not None
            or (attn_mask is not None and attn_mask.dtype != torch.bool)):
        return nn_multihead_forward(self, query, key, value, key_padding_mask=key_padding_mask,
                                     need_weights=need_weights, attn_mask=attn_mask,
                                     average_attn_weights=average_attn_weights, is_causal=is_causal)
    if self.batch_first:
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
    if query is key is value:
        q, k, v = (part.unflatten(-1, (self.num_heads, self.head_dim)).permute(1, 2, 0, 3)
                   for part in F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1))
    else:
        embed_dim = query.shape[-1]
        wq, wk, wv = self.in_proj_weight.split(embed_dim, dim=0)
        bq, bk, bv = ((None, None, None) if self.in_proj_bias is None
                      else self.in_proj_bias.split(embed_dim, dim=0))
        q, k, v = (
            F.linear(t, w, b).unflatten(-1, (self.num_heads, self.head_dim)).permute(1, 2, 0, 3)
            for t, w, b in ((query, wq, bq), (key, wk, bk), (value, wv, bv))
        )
    # First attempt just swapped the boolean "keep" mask for an additive one built with
    # masked_fill -- but masked_fill still traces to a Where/GatherND node keyed on the same u8
    # boolean tensor, so the GPU plugin choked in the same place. Building the bias with pure
    # arithmetic (no torch.bool tensor ever created) avoids that decomposition entirely: only
    # Cast/Mul/Add nodes reach the graph, which the GPU plugin handles fine.
    bias = None
    if attn_mask is not None:
        # Rare path (causal/custom mask); not used by this encoder (only key_padding_mask is
        # ever passed), so faithfulness matters more than GPU-friendliness here.
        mask = ~attn_mask
        if key_padding_mask is not None:
            keep = key_padding_mask[:, None, None, :] == 0
            mask = mask & keep
        bias = torch.zeros(mask.shape, dtype=q.dtype, device=q.device).masked_fill(~mask, -1e9)
    elif key_padding_mask is not None:
        bias = key_padding_mask.to(q.dtype)[:, None, None, :] * -1e9
    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=is_causal and bias is None,
                                          dropout_p=self.dropout if self.training else 0.0)
    attn = attn.permute(2, 0, 1, 3).flatten(-2)
    if self.batch_first:
        attn = attn.transpose(0, 1)
    return self.out_proj(attn), None


def nn_multihead_forward(self, *args, **kwargs):
    return torch.nn.MultiheadAttention.forward(self, *args, **kwargs)


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        return self.m(input_ids, attention_mask, marker_pos, marker_mask, qtype)


def batch_for(agent, state, questions):
    items = []
    for qdef in questions.values():
        q = agent._to_internal(qdef)
        seq, markers = build_sequence(agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    b = collate_items([items], agent.tok.pad_token_id)
    feed = {"input_ids": b["input_ids"].numpy().astype(np.int64), "attention_mask": b["attention_mask"].numpy().astype(np.int64),
            "marker_pos": b["marker_pos"].numpy().astype(np.int64), "marker_mask": b["marker_mask"].numpy(),
            "qtype": b["qtype"].numpy().astype(np.int64)}
    return b, feed, items


def copy_sidecars(agent, out_dir):
    tok = agent.tok
    tmp = os.path.join(out_dir, "_tok")
    tok.save_pretrained(tmp)
    shutil.copy(os.path.join(tmp, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))
    cfg = {k: getattr(tok, k) for k in ("cls_token", "sep_token", "mask_token", "pad_token", "unk_token") if getattr(tok, k, None)}
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(agent.cfg, f, indent=2)


def check_openvino_gpu(onnx_path, profiles):
    import openvino as ov
    core = ov.Core()
    if "GPU" not in core.available_devices:
        print("no GPU device visible to OpenVINO; skipping the GPU compile check", file=sys.stderr)
        return
    model = core.read_model(onnx_path)
    for batch, max_len, k in profiles:
        m = model.clone()
        m.reshape({"input_ids": [batch, max_len], "attention_mask": [batch, max_len],
                   "marker_pos": [batch, k], "marker_mask": [batch, k], "qtype": [batch]})
        t0 = time.time()
        core.compile_model(m, "GPU")
        print(f"OpenVINO GPU compile OK: batch={batch} max_len={max_len} k={k} ({time.time() - t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="convaiinnovations/laya")
    ap.add_argument("--out", default=os.path.join(".laya-cache", "checkpoints", "laya-onnx-gpu", "english"))
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--skip-gpu-check", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # nn.TransformerEncoderLayer has a fused native fast path (torch._transformer_encoder_layer_fwd)
    # that bypasses self_attn.forward() entirely whenever eval mode + batch_first + bias, etc. all
    # hold -- which they do here. That fused kernel builds its own mask via self_attn.merge_masks(),
    # ignoring our patched _DynamicMultiheadAttention.forward completely (confirmed with an
    # instrumented call-counter: 0 calls into the patch with the fast path enabled). Both export
    # attempts so far silently traced that fused path, so the mask patch never took effect --
    # disabling it here is what actually makes the patch (and thus the export) honor it.
    torch.backends.mha.set_fastpath_enabled(False)

    t0 = time.time()
    agent = laya.load(args.repo, device="cpu", subfolder=None)
    model = agent.model.eval()
    print(f"loaded in {time.time() - t0:.1f}s (max_len {agent.cfg['max_len']}, head_max_len {agent.cfg['head_max_len']})")

    b, feed, items = batch_for(agent, STATE, QUESTIONS)
    inputs = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    with torch.no_grad():
        ref_logits, ref_act = model(*inputs)

    # Patch AFTER computing the unpatched reference, so we can prove the patch changes nothing.
    _DynamicMultiheadAttention.forward = _patched_forward
    with torch.no_grad():
        patched_logits, patched_act = model(*inputs)
    d_patch = float(np.abs(patched_logits.numpy() - ref_logits.numpy()).max())
    print(f"max|Δlogits| patched vs original PyTorch forward: {d_patch:.2e}")
    if d_patch > 1e-3:
        print("patched attention changes model output; aborting", file=sys.stderr)
        sys.exit(2)

    names = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
    dyn = {"input_ids": {0: "n", 1: "L"}, "attention_mask": {0: "n", 1: "L"}, "marker_pos": {0: "n", 1: "k"},
           "marker_mask": {0: "n", 1: "k"}, "qtype": {0: "n"}}
    onnx_path = os.path.join(args.out, "laya.onnx")
    t0 = time.time()
    prog = torch.onnx.export(Wrapper(model), inputs, input_names=names, output_names=["logits", "act_logits"],
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
    logits, act = sess.run(None, feed)
    d_logits = float(np.abs(logits - patched_logits.numpy()).max())
    b1, feed1, _ = batch_for(agent, {"body": "Der Kunde wurde zweimal belastet"}, {"churn_risk": QUESTIONS["churn_risk"]})
    with torch.no_grad():
        l1, _ = model(b1["input_ids"], b1["attention_mask"], b1["marker_pos"], b1["marker_mask"], b1["qtype"])
    lo, _ = sess.run(None, feed1)
    d1 = float(np.abs(lo - l1.numpy()).max())
    print(f"max|Δlogits| onnx vs patched PyTorch: batch={d_logits:.2e} single={d1:.2e}")
    if d_logits > 1e-3 or d1 > 1e-3:
        print("ONNX output does not match patched PyTorch; aborting", file=sys.stderr)
        sys.exit(2)

    copy_sidecars(agent, args.out)
    print("sidecars written:", sorted(f for f in os.listdir(args.out) if not f.startswith("_")))

    if not args.skip_gpu_check:
        check_openvino_gpu(onnx_path, profiles=[(4, 128, 2), (16, 256, 2), (50, 128, 2)])
    print("done")


if __name__ == "__main__":
    main()
