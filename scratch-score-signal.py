"""Local, no server. Two follow-ups on the score head.

1. Signal test: the continuous `score` field, averaged per TRUE urgency level. If it is ordered
   Low < Moderate < High the head does carry the signal and only the argmax readout is broken.
2. Position vs token: the same three words in three orders. If the same INDEX wins every time it
   is a position prior; if the same WORD wins it is a token prior.
"""
import collections
import json
import sys

sys.path.insert(0, ".")

from laya_openvino_agent import OpenVINOAgent  # noqa: E402

MAX_LEN = 256

data = json.load(open("tickets-500-en.json", encoding="utf-8"))
tickets = data["tickets"]

by_urgency = collections.defaultdict(list)
for t in tickets:
    by_urgency[t["expected"]["urgency"]].append(t)
sample = [t for lvl in ("Low", "Moderate", "High") for t in by_urgency[lvl][:30]]
states = [t["state"] for t in sample]
expected_urgency = [t["expected"]["urgency"] for t in sample]
print(f"sample: {len(states)} tickets, 30 per urgency level")

URGENCY_CRIT = {
    "Low": "they can wait, a few days or more",
    "Moderate": "a short delay, 48 hours or by the end of the week",
    "High": "today, they cannot wait",
}
INSTRUCTIONS = "What response delay does the person expect?"

agent = OpenVINOAgent(device="GPU", max_len=MAX_LEN)


def run(question):
    answers = agent.predict_batch(states, question, batch_size=len(states))
    return [a["answers"]["q"] for a in answers]


def top_index(answer):
    return max(range(3), key=lambda i: answer["probabilities"][f"{i}"])


print("\n" + "=" * 78)
print("1. does the continuous score field carry the signal?")
print("=" * 78)
ORDERED = ["Low", "Moderate", "High"]
for name, order in (("original order", ORDERED), ("reversed order", ORDERED[::-1])):
    criteria = [f"{k}: {URGENCY_CRIT[k]}" for k in order]
    answers = run({"q": {"type": "score", "instructions": INSTRUCTIONS, "criteria": criteria}})
    print(f"\n  {name}: {', '.join(order)}")
    for lvl in ("Low", "Moderate", "High"):
        rows = [a for a, exp in zip(answers, expected_urgency) if exp == lvl]
        mean_score = sum(r["score"] for r in rows) / len(rows)
        mean_probs = [sum(r["probabilities"][f"{i}"] for r in rows) / len(rows) for i in range(3)]
        chosen = collections.Counter(order[top_index(r)] for r in rows)
        print(f"    true {lvl:<9} n={len(rows):<3} mean score {mean_score:5.3f}   "
              f"mean probs {' '.join(f'{p:.2f}' for p in mean_probs)}   "
              f"chosen {dict(chosen.most_common())}")

print("\n" + "=" * 78)
print("2. same words, three orders: does the position or the word win?")
print("=" * 78)
TRIPLE = ["winter", "spring", "summer"]
for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
    words = [TRIPLE[i] for i in order]
    answers = run({"q": {"type": "score",
                         "instructions": "Which season does the customer mention?",
                         "criteria": words}})
    idx = [top_index(a) for a in answers]
    chosen = collections.Counter(words[i] for i in idx)
    print(f"  order {str(words):<32} index {dict(collections.Counter(idx).most_common())}   "
          f"word {dict(chosen.most_common())}")
