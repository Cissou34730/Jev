"""Local, read-only: does the ONNX export agree with the original PyTorch checkpoint?

Runs the same tickets through both engines on CPU (so the device cannot explain a difference)
and prints the urgence answer side by side. If PyTorch varies with the ticket while OpenVINO
sits on one level, the export is at fault. If they agree, the checkpoint is the limit.
No server, no HTTP, no GPU.
"""
import json
import sys

sys.path.insert(0, ".")

import laya  # noqa: E402

from laya_openvino_agent import OpenVINOAgent  # noqa: E402

MAX_LEN = 256
HEAD_MAX_LEN = 256

data = json.loads(open("tickets-1000.json", "r", encoding="utf-8").read())
questions = {"urgence": data["questions"]["urgence"]}

by_level = {}
for t in data["tickets"]:
    by_level.setdefault(t["expected"]["urgence"], []).append(t)
sample = []
for level in ("Faible", "Modérée", "Élevée"):
    sample.extend(by_level[level][:2])

print("loading PyTorch checkpoint (convaiinnovations/laya, subfolder=multilingual) ...", flush=True)
ref_agent = laya.load("convaiinnovations/laya", device="cpu", subfolder="multilingual")
print("loading OpenVINO agent (device=CPU) ...", flush=True)
ov_agent = OpenVINOAgent(device="CPU", max_len=MAX_LEN)

print()
print(f"{'id':<10}{'expected':<10}{'pytorch score':>14}{'pytorch probs':>26}"
      f"{'openvino score':>15}{'openvino probs':>26}")
for t in sample:
    ref = ref_agent.system_one(t["state"], questions, max_len=MAX_LEN,
                               head_max_len=HEAD_MAX_LEN)["answers"]["urgence"]
    got = ov_agent.system_one(t["state"], questions)["answers"]["urgence"]
    ref_probs = " ".join(f"{ref['probabilities'][k]:.2f}" for k in ("0", "1", "2"))
    got_probs = " ".join(f"{got['probabilities'][k]:.2f}" for k in ("0", "1", "2"))
    print(f"{t['id']:<10}{t['expected']['urgence']:<10}{ref['score']:>14.3f}{ref_probs:>26}"
          f"{got['score']:>15.3f}{got_probs:>26}")
