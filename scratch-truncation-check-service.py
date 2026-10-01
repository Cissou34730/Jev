"""Same measurement as scratch-truncation-check.py, for the 'service' question instead of
'urgence': does its (much shorter) criteria text leave enough room for the ticket, now that
the server runs at max_len=256? Local, read-only, no GPU compile, no server, no request.
"""
import json
import sys

sys.path.insert(0, ".")
from laya.common import encode_text, serialize_state  # noqa: E402

from laya_openvino_agent import OpenVINOAgent  # noqa: E402

agent = OpenVINOAgent(max_len=256)
data = json.loads(open("tickets-1000.json", "r", encoding="utf-8").read())
questions = {"service": data["questions"]["service"]}
ids_list, internal = agent._prepare(questions)

empty_items = agent._encode_state("", internal, ids_list)
overhead = len(empty_items[0]["ids"])
room = max(0, agent.max_len - overhead)

tickets = data["tickets"]
full_lengths = [
    len(encode_text(agent.tokenizer, serialize_state(t["state"]), add_special_tokens=False)["input_ids"])
    for t in tickets
]
truncated = [n for n in full_lengths if n > room]
print(f"max_len={agent.max_len}  head/options overhead={overhead}  room left for state={room}")
print(f"ticket token length: min={min(full_lengths)} mean={sum(full_lengths)/len(full_lengths):.1f} max={max(full_lengths)}")
print(f"tickets whose full state exceeds room: {len(truncated)} / {len(tickets)}")
