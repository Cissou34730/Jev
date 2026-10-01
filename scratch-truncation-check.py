"""Local, read-only: does max_len=128 cut off the end of the ticket (where the urgency
clause sits) before the model ever sees it? Reuses the production tokenizer and sequence
builder so the measurement matches exactly what the compiled model receives. No GPU compile,
no server, no request: constructing OpenVINOAgent only loads the tokenizer and config.
"""
import json
import sys

sys.path.insert(0, ".")
from laya.common import encode_text, serialize_state  # noqa: E402

from laya_openvino_agent import OpenVINOAgent  # noqa: E402

agent = OpenVINOAgent()
data = json.loads(open("tickets-1000.json", "r", encoding="utf-8").read())
questions = {"urgence": data["questions"]["urgence"]}
ids_list, internal = agent._prepare(questions)

# Base overhead (question + options + specials) is state-independent: measure it once with an
# empty state, then room = max_len - overhead - 1 is exactly what's left for the ticket text.
empty_items = agent._encode_state("", internal, ids_list)
overhead = len(empty_items[0]["ids"])  # CLS + head + options + SEP, no state, no trailing SEP
room = max(0, agent.max_len - overhead)

tickets = data["tickets"]
full_lengths = []
for t in tickets:
    state_ids = encode_text(agent.tokenizer, serialize_state(t["state"]), add_special_tokens=False)["input_ids"]
    full_lengths.append(len(state_ids))

truncated = [n for n in full_lengths if n > room]
print(f"tickets: {len(tickets)}")
print(f"max_len={agent.max_len}  head/options overhead={overhead}  room left for state={room}")
print(f"ticket token length: min={min(full_lengths)} mean={sum(full_lengths)/len(full_lengths):.1f} max={max(full_lengths)}")
print(f"tickets whose full state exceeds room (would lose their tail): {len(truncated)} / {len(tickets)}")
if truncated:
    cut = [n - room for n in truncated]
    print(f"tokens lost off the end, among truncated: min={min(cut)} mean={sum(cut)/len(cut):.1f} max={max(cut)}")
