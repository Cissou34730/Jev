"""Local, read-only: confusion matrix for the service question from an existing export.
Shows where the errors actually go, instead of guessing from the aggregate accuracy.
"""
import json
from collections import Counter

rows = json.loads(open("resultats.json", "r", encoding="utf-8").read())

labels = ["facturation", "technique", "livraison"]
pairs = Counter()
for r in rows:
    if r.get("error"):
        continue
    exp = (r.get("expected") or {}).get("service")
    ans = (r.get("answers") or {}).get("service") or {}
    got = ans.get("choice")
    if exp is None or got is None:
        continue
    pairs[(exp, got)] += 1

print("expected -> predicted   (rows: truth, cols: prediction)")
print(f"{'':<14}" + "".join(f"{lab:>14}" for lab in labels) + f"{'recall':>10}")
for exp in labels:
    total = sum(pairs[(exp, got)] for got in labels)
    line = f"{exp:<14}" + "".join(f"{pairs[(exp, got)]:>14}" for got in labels)
    recall = pairs[(exp, exp)] / total if total else 0
    line += f"{recall:>9.1%}"
    print(line)

print()
print(f"{'precision':<14}" + "".join(
    f"{(pairs[(lab, lab)] / max(1, sum(pairs[(exp, lab)] for exp in labels))):>13.1%}" for lab in labels))

scored = sum(pairs.values())
correct = sum(pairs[(lab, lab)] for lab in labels)
print()
print(f"scored={scored} correct={correct} accuracy={correct/scored:.1%}")
