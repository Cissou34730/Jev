"""Local, read-only: same PyTorch vs OpenVINO comparison as scratch-compare-engines.py, but on
the English urgency test (tickets-50-en.json). Only the language changes; everything else
(criteria structure, checkpoint, device) is identical to the French run.
"""
import json
import sys

sys.path.insert(0, ".")

import laya  # noqa: E402

from laya_openvino_agent import OpenVINOAgent  # noqa: E402

MAX_LEN = 256
HEAD_MAX_LEN = 256

data = json.loads(open("tickets-50-en.json", "r", encoding="utf-8").read())
question_name = next(iter(data["questions"]))
questions = {question_name: data["questions"][question_name]}

by_level = {}
for t in data["tickets"]:
    by_level.setdefault(t["expected"][question_name], []).append(t)
sample = []
for level, tickets in by_level.items():
    sample.extend(tickets[:2])

print("loading PyTorch checkpoint (convaiinnovations/laya, subfolder=multilingual) ...", flush=True)
ref_agent = laya.load("convaiinnovations/laya", device="cpu", subfolder="multilingual")
print("loading OpenVINO agent (device=CPU) ...", flush=True)
ov_agent = OpenVINOAgent(device="CPU", max_len=MAX_LEN)

print()
print(f"{'id':<10}{'expected':<10}{'pytorch score':>14}{'pytorch probs':>26}"
      f"{'openvino score':>15}{'openvino probs':>26}")
for t in sample:
    ref = ref_agent.system_one(t["state"], questions, max_len=MAX_LEN,
                               head_max_len=HEAD_MAX_LEN)["answers"][question_name]
    got = ov_agent.system_one(t["state"], questions)["answers"][question_name]
    ref_probs = " ".join(f"{ref['probabilities'][k]:.2f}" for k in ("0", "1", "2"))
    got_probs = " ".join(f"{got['probabilities'][k]:.2f}" for k in ("0", "1", "2"))
    print(f"{t['id']:<10}{t['expected'][question_name]:<10}{ref['score']:>14.3f}{ref_probs:>26}"
          f"{got['score']:>15.3f}{got_probs:>26}")
