"""Local, read-only: for each question on its own, how much of max_len the question and its
options take, and how much is left for the ticket. Answers whether sending one question per
request could help: it cannot, because that head cost is per question and every row already
carries exactly one question's head. No GPU compile, no server, no request.
"""
import json
import sys

sys.path.insert(0, ".")
from laya.common import encode_text, serialize_state  # noqa: E402

from laya_openvino_agent import OpenVINOAgent  # noqa: E402

MAX_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 256

agent = OpenVINOAgent(max_len=MAX_LEN)
data = json.loads(open("tickets-1000.json", "r", encoding="utf-8").read())
questions = data["questions"]

state_lengths = []
for t in data["tickets"]:
    ids = encode_text(agent.tokenizer, serialize_state(t["state"]), add_special_tokens=False)["input_ids"]
    state_lengths.append(len(ids))

print(f"max_len={MAX_LEN}   tickets={len(state_lengths)}   "
      f"ticket tokens: min={min(state_lengths)} mean={sum(state_lengths)/len(state_lengths):.1f} "
      f"max={max(state_lengths)}")
print()
print(f"{'question':<14}{'type':<8}{'head+opts':>10}{'room for state':>16}{'truncated':>11}")
for name, spec in questions.items():
    empty_row = agent._encode_state("", agent._prepare({name: spec})[1], [name])[0]["ids"]
    head = len(empty_row)
    room = max(0, MAX_LEN - head)
    truncated = sum(1 for n in state_lengths if n > room)
    print(f"{name:<14}{spec['type']:<8}{head:>10}{room:>16}{truncated:>7} / {len(state_lengths)}")
