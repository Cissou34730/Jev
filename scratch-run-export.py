"""Run one exported checkpoint over the English 500-ticket set and report accuracy.

Parameterised by the export directory, so the english and multilingual checkpoints are measured by
exactly the same code path, on the same tickets, the same questions, the same device and max_len.
Nothing here touches the server: this loads the ONNX through the same OpenVINOAgent the server uses.

  uv run python scratch-run-export.py .laya-cache/checkpoints/laya-english GPU
  uv run python scratch-run-export.py .laya-cache/checkpoints/laya-master  GPU
"""
import collections
import json
import sys

sys.path.insert(0, ".")

from laya_openvino_agent import OpenVINOAgent  # noqa: E402

EXPORT = sys.argv[1]
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "GPU"
MAX_LEN = 256
TICKETS = "tickets-500-en.json"

data = json.load(open(TICKETS, encoding="utf-8"))
questions = data["questions"]
tickets = data["tickets"]

agent = OpenVINOAgent(onnx_path=EXPORT + "/laya.onnx", tokenizer_dir=EXPORT,
                      device=DEVICE, max_len=MAX_LEN)


def decision(answers, name):
    spec = questions[name]
    got = answers[name]
    if spec["type"] == "noul":
        return str(float(got["noul"]) >= 0.5)
    if spec["type"] == "choice":
        return str(got["choice"])
    probabilities = got["probabilities"]
    best = int(max(probabilities, key=lambda k: probabilities[k]))
    return spec["criteria"][best].split(":")[0].strip()


answers = [record["answers"] for record in
           agent.predict_batch([t["state"] for t in tickets], questions, batch_size=50)]

print(f"\nmodel   : {EXPORT}")
print(f"device  : {DEVICE}   max_len: {MAX_LEN}   head_max_len: {agent.head_max_len}"
      f"   tickets: {len(tickets)}")
print(f"\n{'question':<10}{'type':<8}{'correct':>10}{'accuracy':>11}   chosen")
for name in questions:
    got = [decision(a, name) for a in answers]
    expected = [str(t["expected"][name]) for t in tickets]
    hits = sum(1 for g, e in zip(got, expected) if g == e)
    chosen = dict(collections.Counter(got).most_common())
    print(f"{name:<10}{questions[name]['type']:<8}{hits:>7}/{len(tickets):<4}"
          f"{hits / len(tickets):>10.1%}   {chosen}")

print(f"\n{'question':<10}{'true labels in this set':>34}")
for name in questions:
    expected = collections.Counter(str(t["expected"][name]) for t in tickets)
    print(f"{name:<10}{str(dict(expected.most_common())):>34}")

confidences = [float(a[name].get("answer_confidence", 0)) for a in answers for name in questions]
print(f"\nmean answer_confidence over all answers: {sum(confidences) / len(confidences):.1%}")
