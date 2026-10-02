"""Score the blind annotator, and Laya once its run exists, on the same 500 tickets.

Luna is scored from luna-answers.json joined through blind-mapping.json. Laya is scored from
resultats-500.json, produced by the client on tickets-500-sample.json, joined the same way. Both
then answered the same states under the same current question definitions, so the two accuracies
are directly comparable.

The dataset's own marginal is printed alongside: with three balanced classes a guessing annotator
also produces a balanced distribution, so a balanced output is not by itself evidence of accuracy.
"""
import collections
import json
from pathlib import Path

QUESTIONS = json.loads(Path("tickets-1000.json").read_text(encoding="utf-8"))["questions"]
LABELS = [criterion.split(":")[0].strip() for criterion in QUESTIONS["urgence"]["criteria"]]
mapping = json.loads(Path("blind-mapping.json").read_text(encoding="utf-8"))
by_original = {entry["id"]: key for key, entry in mapping.items()}


def accuracy(name, decisions):
    print(f"\n{name}: {len(decisions)} of {len(mapping)} tickets answered")
    for question in QUESTIONS:
        hits = total = 0
        for key, answers in decisions.items():
            expected = mapping[key]["expected"][question]
            if answers.get(question) is None:
                continue
            total += 1
            hits += str(answers[question]) == str(expected)
        print(f"  {question:<15} {hits:>4}/{total:<4} {hits / total:6.1%}")
    for question in QUESTIONS:
        dist = collections.Counter(str(mapping[k]["expected"][question]) for k in decisions)
        print(f"  {question:<15} dataset marginal {dict(sorted(dist.items()))}")


luna = {}
for entry in json.loads(Path("luna-answers.json").read_text(encoding="utf-8"))["answers"]:
    answers = entry["answers"]
    luna[entry["id"]] = {
        "service": answers["service"]["choice"],
        "remboursement": float(answers["remboursement"]["noul"]) >= 0.5,
        "urgence": LABELS[int(answers["urgence"]["score"])],
    }
accuracy("Luna (GPT-5.6, blind, no context)", luna)

laya = {}
results = Path("resultats-500.json")
if results.exists():
    for record in json.loads(results.read_text(encoding="utf-8")):
        key = by_original.get(record["id"])
        if key is None or record.get("error"):
            continue
        answers = record["answers"]
        probabilities = answers["urgence"]["probabilities"]
        index = max(probabilities, key=lambda k: probabilities[k])
        laya[key] = {
            "service": answers["service"]["choice"],
            "remboursement": float(answers["remboursement"]["noul"]) >= 0.5,
            "urgence": LABELS[int(index)],
        }
    accuracy("Laya (OpenVINO, current questions)", laya)
else:
    print(f"\nLaya: {results.name} not found, nothing to compare yet")

if laya:
    both = sorted(set(luna) & set(laya))
    print(f"\nagreement on the {len(both)} tickets both answered")
    for question in QUESTIONS:
        same = sum(1 for k in both if str(luna[k][question]) == str(laya[k][question]))
        print(f"  {question:<15} {same:>4}/{len(both):<4} {same / len(both):6.1%}")
