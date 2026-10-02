"""Token budget for the english export, which differs from the multilingual one in two ways that
make the old measurements invalid: a 50k vocabulary (4x smaller than the multilingual 256k, so more
tokens per word) and head_max_len=192 instead of 256.

Measures, per question, the head cost (instructions + options) and the room left for the ticket, and
counts how many tickets would be truncated.
"""
import json
import sys

sys.path.insert(0, ".")

from laya.common import encode_text, serialize_state  # noqa: E402

from laya_openvino_agent import OpenVINOAgent  # noqa: E402

EXPORT = ".laya-cache/checkpoints/laya-english"
TICKETS = "tickets-500-en.json"
MAX_LEN = 256

data = json.load(open(TICKETS, encoding="utf-8"))
questions = data["questions"]
agent = OpenVINOAgent(onnx_path=EXPORT + "/laya.onnx", tokenizer_dir=EXPORT, device="CPU",
                      max_len=MAX_LEN)
ids, internal = agent._prepare(questions)
print(f"head_max_len from the export's own config: {agent.head_max_len}")
print(f"max_markers: {agent.num_markers}   max_len used here: {MAX_LEN}")


def state_tokens(state):
    text = serialize_state(state).replace(agent.tokenizer.mask_token, " ")
    return len(encode_text(agent.tokenizer, text, add_special_tokens=False)["input_ids"])


head = {}
for qid in ids:
    head[qid] = len(agent._encode_state("", internal, [qid])[0]["ids"]) - 1

def option_count(q):
    # noul has no criteria: it always renders exactly two options (false, true).
    return 2 if q["t"] == "noul" else len(q["crit"])


print(f"\n{'question':<16}{'type':<8}{'opts':>5}{'head tok':>10}{'room':>7}{'head fits 192':>15}")
for qid in ids:
    room = MAX_LEN - head[qid] - 1
    fits = "yes" if head[qid] <= agent.head_max_len else "NO"
    print(f"{qid:<16}{internal[qid]['t']:<8}{option_count(internal[qid]):>5}"
          f"{head[qid]:>10}{room:>7}{fits:>15}")

states = [state_tokens(t["state"]) for t in data["tickets"]]
states_sorted = sorted(states)
print(f"\nticket text alone: min {states_sorted[0]}  mean {sum(states)/len(states):.1f}  "
      f"max {states_sorted[-1]} tokens")

print(f"\n{'question':<16}{'head+longest':>14}{'max_len':>9}{'truncated':>11}")
for qid in ids:
    over = sum(1 for s in states if head[qid] + s + 1 > MAX_LEN)
    print(f"{qid:<16}{head[qid] + states_sorted[-1] + 1:>14}{MAX_LEN:>9}{over:>11}")
print("\ntruncated = tickets whose text would be cut for that question at this max_len")
