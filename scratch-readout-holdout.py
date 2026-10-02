"""Held-out version of the readout audit.

scratch-readout-audit.py fitted the decision rule on the same 500 tickets it then scored on, so its
90.2% and 84.0% are upper bounds, not expectations. Here the rule is fitted on one half and scored
on the other, in both directions, which is what an unseen ticket would actually see.
"""
import json
from pathlib import Path

PATH = "resultats-500-en-anglais.json"
LABELS = ["Low", "Moderate", "High"]
records = json.loads(Path(PATH).read_text(encoding="utf-8"))
FOLDS = ([r for i, r in enumerate(records) if i % 2 == 0],
         [r for i, r in enumerate(records) if i % 2 == 1])


def refund_pairs(rows):
    return [(float(r["answers"]["refund"]["noul"]), bool(r["expected"]["refund"])) for r in rows]


def urgency_pairs(rows):
    return [(float(r["answers"]["urgency"]["score"]), r["expected"]["urgency"]) for r in rows]


def fit_refund(rows):
    best = (0.5, -1.0)
    for step in range(1, 200):
        threshold = step / 200
        accuracy = sum(1 for v, e in rows if (v >= threshold) == e) / len(rows)
        if accuracy > best[1]:
            best = (threshold, accuracy)
    return best[0]


def fit_urgency(rows):
    best = ((0.5, 1.5), -1.0)
    grid = [step / 20 for step in range(0, 61)]
    for low in grid:
        for high in grid:
            if high <= low:
                continue
            hits = 0
            for value, expected in rows:
                got = LABELS[0] if value <= low else (LABELS[2] if value > high else LABELS[1])
                hits += got == expected
            if hits / len(rows) > best[1]:
                best = ((low, high), hits / len(rows))
    return best[0]


def score_refund(rows, threshold):
    return sum(1 for v, e in rows if (v >= threshold) == e) / len(rows)


def score_urgency(rows, cuts):
    hits = 0
    for value, expected in rows:
        got = LABELS[0] if value <= cuts[0] else (LABELS[2] if value > cuts[1] else LABELS[1])
        hits += got == expected
    return hits / len(rows)


def score_urgency_argmax(rows):
    hits = 0
    for record in rows:
        probabilities = record["answers"]["urgency"]["probabilities"]
        best = int(max(probabilities, key=lambda k: probabilities[k]))
        hits += LABELS[best] == record["expected"]["urgency"]
    return hits / len(rows)


print(f"{PATH}: {len(records)} tickets\n")
print("held-out: fitted on one half, scored on the other, averaged over both directions\n")

print(f"  refund, rule used today (threshold 0.50)      {score_refund(refund_pairs(records), 0.50):>6.1%}  fitted on nothing")
fitted = []
for test, train in (FOLDS, FOLDS[::-1]):
    threshold = fit_refund(refund_pairs(train))
    held_out = score_refund(refund_pairs(test), threshold)
    fitted.append(held_out)
    print(f"  refund, fitted on the other half  t={threshold:.3f}   {held_out:>6.1%}  held out")
print(f"  refund, average of the two                     {sum(fitted) / 2:>6.1%}")

print()
print(f"  urgency, rule used today (argmax)              {score_urgency_argmax(records):>6.1%}  fitted on nothing")
fitted = []
for test, train in (FOLDS, FOLDS[::-1]):
    cuts = fit_urgency(urgency_pairs(train))
    held_out = score_urgency(urgency_pairs(test), cuts)
    fitted.append(held_out)
    print(f"  urgency, fitted on the other half {str(cuts):<13} {held_out:>6.1%}  held out")
print(f"  urgency, average of the two                    {sum(fitted) / 2:>6.1%}")
