"""Validate the chunked blind annotations returned by the subagent and merge them into
luna-answers.json, in the same shape as the answers Laya produced.

Checks that every id in blind-mapping.json was answered exactly once, so a truncated or
duplicated chunk is caught before the comparison is run rather than after.
"""
import collections
import json
from pathlib import Path

entries = []
for path in sorted(Path(".").glob("luna-chunk-*.json")):
    chunk = json.loads(path.read_text(encoding="utf-8"))["answers"]
    print(f"{path.name:<22} {len(chunk):>4} entries   {chunk[0]['id']}..{chunk[-1]['id']}")
    entries.extend(chunk)

mapping = json.loads(Path("blind-mapping.json").read_text(encoding="utf-8"))
counts = collections.Counter(entry["id"] for entry in entries)
missing = sorted(set(mapping) - set(counts))
duplicated = sorted(i for i, n in counts.items() if n > 1)

print(f"\nentries {len(entries)}   unique ids {len(counts)}   expected {len(mapping)}")
print(f"missing {len(missing)}   duplicated {len(duplicated)}")
for name, values in (("missing", missing), ("duplicated", duplicated)):
    if values:
        print(f"  {name}: {values[:20]}")

if missing or duplicated or len(counts) != len(mapping):
    print("\nnot written: chunks are incomplete")
    raise SystemExit(1)

ordered = sorted(entries, key=lambda e: e["id"])
out = {"model": "GPT-5.6 Luna (copilot)", "answers": ordered}
Path("luna-answers.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

def decision(answer):
    # Score 0 is falsy, so `a or b` would fall through to the wrong branch: dispatch on the key.
    if "choice" in answer:
        return answer["choice"]
    if "noul" in answer:
        return str(float(answer["noul"]) >= 0.5)
    return str(answer["score"])


print("\nclass distribution of the annotations")
for question in next(iter(mapping.values()))["expected"]:
    dist = collections.Counter(decision(e["answers"][question]) for e in ordered)
    print(f"  {question:<15} {dict(sorted(dist.items()))}")
print("\nwritten luna-answers.json")
