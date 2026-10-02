"""Local, no server. Same 300 tickets, same criteria wording, same instruction: only the question
TYPE changes (score vs choice). This is the direct test of "can urgency be made to work by
changing the primitive", and each variant is also run with the options permuted so a position
effect cannot be mistaken for accuracy.
"""
import collections
import json
import sys

sys.path.insert(0, ".")

from laya_openvino_agent import OpenVINOAgent  # noqa: E402

MAX_LEN = 256
PER_LEVEL = 100

data = json.load(open("tickets-500-en.json", encoding="utf-8"))
by_urgency = collections.defaultdict(list)
for t in data["tickets"]:
    by_urgency[t["expected"]["urgency"]].append(t)
sample = [t for lvl in ("Low", "Moderate", "High") for t in by_urgency[lvl][:PER_LEVEL]]
states = [t["state"] for t in sample]
truth = [t["expected"]["urgency"] for t in sample]
print(f"sample: {len(states)} tickets, {PER_LEVEL} per urgency level, chance 33.3%")

CRIT = {
    "Low": "they can wait, a few days or more",
    "Moderate": "a short delay, 48 hours or by the end of the week",
    "High": "today, they cannot wait",
}
INSTRUCTIONS = "What response delay does the person expect?"

agent = OpenVINOAgent(device="GPU", max_len=MAX_LEN)

VARIANTS = [
    ("score", ["Low", "Moderate", "High"]),
    ("score", ["High", "Low", "Moderate"]),
    ("choice", ["Low", "Moderate", "High"]),
    ("choice", ["High", "Low", "Moderate"]),
]

print(f"\n{'type':<8}{'order':<28}{'accuracy':>10}   chosen")
for qtype, order in VARIANTS:
    criteria = [f"{k}: {CRIT[k]}" for k in order]
    if qtype == "score":
        question = {"q": {"type": "score", "instructions": INSTRUCTIONS, "criteria": criteria}}
    else:
        question = {"q": {"type": "choice", "instructions": INSTRUCTIONS,
                          "criteria": {k: CRIT[k] for k in order}}}
    answers = [a["answers"]["q"] for a in
               agent.predict_batch(states, question, batch_size=PER_LEVEL)]
    if qtype == "score":
        chosen = [order[max(range(3), key=lambda i: a["probabilities"][str(i)])] for a in answers]
    else:
        chosen = [a["choice"] for a in answers]
    correct = sum(1 for c, t in zip(chosen, truth) if c == t)
    dist = dict(collections.Counter(chosen).most_common())
    print(f"{qtype:<8}{', '.join(order):<28}{correct / len(states):>9.1%}   {dist}")

print("\nreference: bag-of-words Naive Bayes on the full text scores 100.0% on this question")
