"""Local, no server, no HTTP. Two experiments on the OpenVINO agent.

A. service (choice, 3 options): the same tickets under 4 criteria orders. If the answer follows
   the meaning of the options it keeps its label; if it follows the prompt position it keeps its
   index and the label changes with the order.

B. urgency (score, 3 options): content replaced by things the ticket cannot answer. A head that
   reads the content should spread over the options; one keyed on position will not.
"""
import collections
import json
import sys

sys.path.insert(0, ".")

from laya_openvino_agent import OpenVINOAgent  # noqa: E402

MAX_LEN = 256
PER_CLASS = 30

data = json.load(open("tickets-500-en.json", encoding="utf-8"))
tickets = data["tickets"]

by_service = collections.defaultdict(list)
for t in tickets:
    by_service[t["expected"]["service"]].append(t)
sample = [t for cls in ("billing", "technical", "delivery") for t in by_service[cls][:PER_CLASS]]
states = [t["state"] for t in sample]
expected_service = [t["expected"]["service"] for t in sample]
expected_urgency = [t["expected"]["urgency"] for t in sample]
print(f"sample: {len(states)} tickets, {PER_CLASS} per service class")

SERVICE_CRIT = {
    "billing": "Payment, direct debit, invoice or subscription",
    "technical": "Software malfunction, access issue, error or unavailable feature",
    "delivery": "Shipping, tracking, receipt or parcel status",
}
SERVICE_INSTRUCTIONS = "Which department should handle this request?"

URGENCY_CRIT = {
    "Low": "they can wait, a few days or more",
    "Moderate": "a short delay, 48 hours or by the end of the week",
    "High": "today, they cannot wait",
}
URGENCY_INSTRUCTIONS = "What response delay does the person expect?"

agent = OpenVINOAgent(device="GPU", max_len=MAX_LEN)


def choice_question(order):
    return {
        "service": {
            "type": "choice",
            "instructions": SERVICE_INSTRUCTIONS,
            "criteria": {k: SERVICE_CRIT[k] for k in order},
        }
    }


def score_question(instructions, criteria):
    return {
        "q": {"type": "score", "instructions": instructions, "criteria": criteria}
    }


def top_index(answer):
    probs = answer["probabilities"]
    return max(range(len(probs)), key=lambda i: probs[f"{i}"])


def run(question, label):
    print(f"\n--- {label} ---", flush=True)
    answers = agent.predict_batch(states, question, batch_size=len(states))
    return [a["answers"][next(iter(question))] for a in answers]


print("\n" + "=" * 78)
print("A. choice: does the answer follow the meaning or the prompt position?")
print("=" * 78)

ORDERS = [
    ["billing", "technical", "delivery"],
    ["technical", "delivery", "billing"],
    ["delivery", "billing", "technical"],
    ["delivery", "technical", "billing"],
]
per_order_labels = []
per_order_positions = []
for order in ORDERS:
    answers = run(choice_question(order), f"order = {', '.join(order)}")
    labels = [a["choice"] for a in answers]
    positions = [order.index(lbl) if lbl in order else -1 for lbl in labels]
    per_order_labels.append(labels)
    per_order_positions.append(positions)
    correct = sum(1 for lbl, exp in zip(labels, expected_service) if lbl == exp)
    dist = collections.Counter(labels)
    pos_dist = collections.Counter(positions)
    print(f"  accuracy {correct / len(states):6.1%}   labels {dict(dist.most_common())}")
    print(f"                        positions {dict(pos_dist.most_common())}")

n = len(states)
label_stable = sum(1 for i in range(n) if len({per_order_labels[o][i] for o in range(len(ORDERS))}) == 1)
pos_stable = sum(1 for i in range(n) if len({per_order_positions[o][i] for o in range(len(ORDERS))}) == 1)
print(f"\n  same LABEL across all {len(ORDERS)} orders: {label_stable}/{n} ({label_stable / n:.1%})")
print(f"  same POSITION across all {len(ORDERS)} orders: {pos_stable}/{n} ({pos_stable / n:.1%})")

print("\n" + "=" * 78)
print("B. score: does the answer follow the content or the position?")
print("=" * 78)

CONTROLS = [
    ("urgency, original order",
     score_question(URGENCY_INSTRUCTIONS, [f"{k}: {URGENCY_CRIT[k]}" for k in ("Low", "Moderate", "High")]),
     ["Low", "Moderate", "High"]),
    ("urgency, reversed order",
     score_question(URGENCY_INSTRUCTIONS, [f"{k}: {URGENCY_CRIT[k]}" for k in ("High", "Moderate", "Low")]),
     ["High", "Moderate", "Low"]),
    ("nothing to do with the ticket: A/B/C",
     score_question("Answer with A, B or C.", ["A", "B", "C"]), None),
    ("nothing to do with the ticket: seasons",
     score_question("Which season does the customer mention?", ["winter", "spring", "summer"]), None),
    ("nothing to do with the ticket: colours",
     score_question("Which colour does the customer complain about?", ["red", "green", "blue"]), None),
]

for label, question, order in CONTROLS:
    answers = run(question, label)
    idx = [top_index(a) for a in answers]
    idx_dist = collections.Counter(idx)
    chosen = [order[i] for i in idx] if order else None
    line = f"  index 0:{idx_dist[0]:<4} 1:{idx_dist[1]:<4} 2:{idx_dist[2]:<4}"
    extra = ""
    if chosen:
        correct = sum(1 for c, exp in zip(chosen, expected_urgency) if c == exp)
        extra = f"   accuracy {correct / len(states):6.1%}   chosen {dict(collections.Counter(chosen).most_common())}"
    print(line + extra)
