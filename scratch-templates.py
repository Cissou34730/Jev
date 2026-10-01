import json
import re

data = json.load(open("tickets-1000.json", "r", encoding="utf-8"))
tickets = data["tickets"][:500]

GREETINGS = [
    "Madame, Monsieur,", "Bonjour,", "Bonjour au support,", "Bonjour l'équipe,",
    "Je vous contacte car",
]
CLOSINGS = [
    "Une intervention immédiate est nécessaire car toute mon équipe est bloquée.",
    "Merci d'agir aujourd'hui, les conséquences sont importantes.",
    "C'est urgent : cette situation bloque mon activité aujourd'hui.",
    "Merci de traiter cette demande dans les 48 heures.",
    "Le problème devient gênant et nécessite un retour rapide.",
    "J'aimerais une solution avant la fin de la semaine.",
    "Je peux attendre quelques jours pour votre retour.",
    "Ce n'est pas urgent, une réponse cette semaine me convient.",
    "Merci de regarder ce point lorsque vous aurez le temps.",
]

def strip_greeting(text):
    for g in GREETINGS:
        if text.startswith(g):
            return g, text[len(g):].strip()
    return None, text

def strip_closing(text):
    # "Référence" trails every closing clause before the "TKT-" cut, e.g. "...bloquée. Référence ".
    text = re.sub(r"Référence\s*$", "", text).strip()
    for c in CLOSINGS:
        if text.rstrip().endswith(c):
            return text[: -len(c)].strip(), c
    return text, None

templates = {}
unmatched = []
for t in tickets:
    idx = t["state"].find("TKT-")
    before = t["state"][:idx] if idx >= 0 else t["state"]
    greet, rest = strip_greeting(before)
    body, closing = strip_closing(rest)
    if greet is None or closing is None:
        unmatched.append(t["id"])
        continue
    masked = re.sub(r"\d+", "#", body)
    templates.setdefault(masked, []).append(t["id"])

print(f"tickets analysed: {len(tickets)}  unmatched: {len(unmatched)}")
print(f"distinct body templates (numbers masked): {len(templates)}")
print()
for masked, ids in sorted(templates.items(), key=lambda kv: -len(kv[1])):
    print(f"{len(ids):>4}  {masked}")
