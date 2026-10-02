"""Is our GatherND workaround result-neutral? Full-precision comparison of the official unmodified
PyTorch checkpoint against our exported, patched OpenVINO graph -- same tickets, same questions,
same max_len.

Why this matters: the multilingual encoder is 2/3 sliding-window attention (local_attention=128,
layer_types alternating full/sliding), and export_laya_master.py rebuilds that mask by hand instead
of calling transformers.masking_utils, to keep u8 GatherND out of the traced graph. That rebuild is
the one place a divergence could hide. This measures it instead of asserting it.

Both sides run on CPU so the comparison isolates the graph/mask rewrite from GPU-vs-CPU numerics.
OpenVINO's _decode rounds to 4 decimals, so resolution is 1e-4: enough to separate a faithful
export (diff ~1e-5) from a broken sliding-window mask (diff >> 1e-2).

The decision count is the number that matters: a tiny numeric difference that never flips an answer
means every accuracy figure we measured describes the checkpoint, not our export.
"""
import json
import sys

sys.path.insert(0, ".")

import laya  # noqa: E402

from laya_openvino_agent import OpenVINOAgent  # noqa: E402

MAX_LEN = 256
HEAD_MAX_LEN = 256
SAMPLE = 200

data = json.load(open("tickets-1000.json", encoding="utf-8"))
questions = data["questions"]
tickets = data["tickets"][:SAMPLE]
print(f"tickets: {len(tickets)}   questions: {list(questions)}")

print("loading the official checkpoint (unpatched, CPU) ...", flush=True)
ref_agent = laya.load("convaiinnovations/laya", device="cpu", subfolder="multilingual")
print("loading our exported graph (CPU, so device numerics do not enter the comparison) ...", flush=True)
ov_agent = OpenVINOAgent(device="CPU", max_len=MAX_LEN)


def numbers(answers, question):
    """The comparable numeric content of one answer: its probabilities, or the noul scalar."""
    spec = questions[question]
    got = answers[question]
    if spec["type"] == "noul":
        return [float(got["noul"])]
    if spec["type"] == "choice":
        return [float(got["probabilities"][key]) for key in spec["criteria"]]
    return [float(got["probabilities"][str(i)]) for i in range(len(spec["criteria"]))]


def decision(answers, question):
    spec = questions[question]
    got = answers[question]
    if spec["type"] == "noul":
        return float(got["noul"]) >= 0.5
    if spec["type"] == "choice":
        return got["choice"]
    probabilities = got["probabilities"]
    best = int(max(probabilities, key=lambda k: probabilities[k]))
    return spec["criteria"][best].split(":")[0].strip()


diffs = {name: [] for name in questions}
flips = {name: 0 for name in questions}
for index, ticket in enumerate(tickets, start=1):
    reference = ref_agent.system_one(ticket["state"], questions,
                                     max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)["answers"]
    ours = ov_agent.system_one(ticket["state"], questions)["answers"]
    for name in questions:
        a, b = numbers(reference, name), numbers(ours, name)
        diffs[name].extend(abs(x - y) for x, y in zip(a, b))
        flips[name] += decision(reference, name) != decision(ours, name)
    if index % 25 == 0:
        print(f"  {index}/{len(tickets)} tickets ...", flush=True)

print(f"\n{'question':<15}{'values':>8}{'max |delta|':>13}{'mean |delta|':>14}{'decisions flipped':>20}")
overall = 0.0
total_flips = 0
for name in questions:
    values = diffs[name]
    worst = max(values)
    overall = max(overall, worst)
    total_flips += flips[name]
    print(f"{name:<15}{len(values):>8}{worst:>13.2e}{sum(values) / len(values):>14.2e}"
          f"{flips[name]:>20}")

print(f"\noverall max |delta| over {sum(len(v) for v in diffs.values())} compared values: {overall:.2e}")
print(f"decisions that changed between the two engines: {total_flips} of {len(tickets) * len(questions)}")
print("\nreading: max |delta| ~1e-5 or below with no flips means the mask rewrite is result-neutral.")
print("         max |delta| >> 1e-2 with flips means our numbers describe the export, not the")
print("         checkpoint, and every accuracy measured this session is suspect.")
