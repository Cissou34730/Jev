"""Build a blind file for an independent annotator, mirroring exactly what Laya receives.

Laya is only ever sent `state`; it never sees `id` or `expected`. This file therefore keeps
`state` and `questions` byte-for-byte and drops everything else, so an annotator reading it gets
the same input Laya got.

The ids are replaced because they are not neutral: in tickets-1000.json the prefix encodes the
service class perfectly (FAC->facturation 334/334, LIV->livraison 333/333, TEC->technique
333/333). Unknown ids would leave a solver able to read the answer off the key.

The mapping back to the original id and its expected labels goes to blind-mapping.json, which the
annotator is told not to read.
"""
import json
import random

SOURCE = "tickets-1000.json"
BLIND = "tickets-500-blind.json"
MAPPING = "blind-mapping.json"
LAYA_SIDE = "tickets-500-sample.json"
SAMPLE = 500

data = json.load(open(SOURCE, encoding="utf-8"))
picked = random.Random(0).sample(data["tickets"], SAMPLE)

blind, mapping = [], {}
for index, ticket in enumerate(picked, start=1):
    key = "B-%04d" % index
    blind.append({"id": key, "state": ticket["state"]})
    mapping[key] = {"id": ticket["id"], "expected": ticket["expected"]}

question_types = {name: spec["type"] for name, spec in data["questions"].items()}
json.dump({"questions": data["questions"], "tickets": blind},
          open(BLIND, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(mapping, open(MAPPING, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# The same 500 tickets with their original ids and labels, so Laya answers exactly these inputs.
# resultats.json cannot be reused for the comparison: it was produced with the pre-shortening
# urgence criteria and answered level 2 for all 1000, the signature of the max_len=128 run where
# the question's own text ate the whole budget and no ticket text reached the model.
laya_side = [{"id": t["id"], "state": t["state"], "expected": t["expected"]} for t in picked]
json.dump({"questions": data["questions"], "tickets": laya_side},
          open(LAYA_SIDE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

leaks = [t for t in blind if set(t) != {"id", "state"}]
assert not leaks, f"{len(leaks)} entries carry extra fields"
assert len({t["id"] for t in blind}) == SAMPLE
assert all("expected" not in json.dumps(t) for t in blind)
assert all(t["state"] in json.dumps(data["tickets"], ensure_ascii=False) for t in blind[:5])

print(f"{BLIND}: {len(blind)} tickets, questions {question_types}")
print(f"{LAYA_SIDE}: the same {len(laya_side)} tickets, original ids and labels, for the Laya run")
print(f"{MAPPING}: mapping for {len(mapping)} ids (annotator must not read this)")
print(f"fields per ticket: {sorted(blind[0])}  -- no 'expected' anywhere")
print(f"original id prefixes present: {sorted({t['id'].split('-')[0] for t in blind})}")
