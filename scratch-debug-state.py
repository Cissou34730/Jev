import json

data = json.load(open("tickets-1000.json", "r", encoding="utf-8"))
for t in data["tickets"][:3]:
    print(repr(t["state"]))
    print("---")
