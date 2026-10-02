"""Read the client's own export and test whether the two weak questions are limited by their
decision rule rather than by the model. No inference: everything comes from the saved answers.

refund  : the client calls True when noul >= 0.5. Sweep the threshold, and look at where the two
          classes actually sit, to see whether 0.5 is near the crossing point or far from it.
urgency : the client takes argmax over the levels. Compare that against rounding the continuous
          score field, and against the best pair of thresholds (an optimistic upper bound).
"""
import collections
import json
from pathlib import Path

PATH = "resultats-500-en-anglais.json"
LABELS = ["Low", "Moderate", "High"]

records = json.loads(Path(PATH).read_text(encoding="utf-8"))
print(f"{PATH}: {len(records)} records, errors {sum(1 for r in records if r.get('error'))}")

print("\n" + "=" * 72)
print("refund (noul): is the 0.5 threshold misplaced?")
print("=" * 72)
rows = [(float(r["answers"]["refund"]["noul"]), bool(r["expected"]["refund"])) for r in records]
print(f"  {sum(1 for _, e in rows if e)} true, {sum(1 for _, e in rows if not e)} false\n")
for label, want in (("True ", True), ("False", False)):
    values = sorted(v for v, e in rows if e is want)
    at = lambda f: values[int(f * (len(values) - 1))]
    print(f"  expected {label}: min {values[0]:.3f}  q25 {at(.25):.3f}  median {at(.5):.3f} "
          f" q75 {at(.75):.3f}  max {values[-1]:.3f}")

print(f"\n  {'threshold':>10}{'accuracy':>11}")
best = (0.0, -1.0)
for step in range(5, 100, 5):
    threshold = step / 100
    accuracy = sum(1 for v, e in rows if (v >= threshold) == e) / len(rows)
    if accuracy > best[1]:
        best = (threshold, accuracy)
    marker = "   <- the client" if step == 50 else ""
    print(f"  {threshold:>10.2f}{accuracy:>10.1%}{marker}")
print(f"\n  best: threshold {best[0]:.2f} gives {best[1]:.1%}, "
      f"the client's 0.50 gives "
      f"{sum(1 for v, e in rows if (v >= 0.5) == e) / len(rows):.1%}")

print("\n" + "=" * 72)
print("urgency (score): is argmax the right readout?")
print("=" * 72)


def by_argmax(answer):
    probabilities = answer["probabilities"]
    return int(max(probabilities, key=lambda k: probabilities[k]))


for name, readout in (("argmax of probabilities (the client's rule)", by_argmax),
                      ("round(score field)", lambda a: min(2, max(0, round(float(a["score"])))))):
    hits = sum(1 for r in records
               if LABELS[readout(r["answers"]["urgency"])] == r["expected"]["urgency"])
    print(f"  {name:<44}{hits:>4}/{len(records)}  {hits / len(records):>6.1%}")

scored = [(float(r["answers"]["urgency"]["score"]), r["expected"]["urgency"]) for r in records]
low = sorted(v for v, e in scored if e == "Low")
mid = sorted(v for v, e in scored if e == "Moderate")
high = sorted(v for v, e in scored if e == "High")
at = lambda values, f: values[int(f * (len(values) - 1))]
for name, values in (("Low", low), ("Moderate", mid), ("High", high)):
    print(f"  score | expected {name:<9} n={len(values):<4} median {at(values, .5):.3f}"
          f"   q25 {at(values, .25):.3f}  q75 {at(values, .75):.3f}")

grid = [step / 20 for step in range(0, 61)]
best = (None, -1.0)
for low_cut in grid:
    for high_cut in grid:
        if high_cut <= low_cut:
            continue
        hits = 0
        for value, expected in scored:
            got = LABELS[0] if value <= low_cut else (LABELS[2] if value > high_cut else LABELS[1])
            hits += got == expected
        if hits / len(scored) > best[1]:
            best = ((low_cut, high_cut), hits / len(scored))
print(f"\n  best thresholds fitted on the whole set: Low<={best[0][0]:.2f}, High>{best[0][1]:.2f}"
      f"  -> {best[1]:.1%}")
print("  that is an upper bound fitted on the answer key, not a held-out result.")
