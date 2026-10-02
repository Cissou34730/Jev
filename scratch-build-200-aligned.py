"""Build tickets-200-aligned.json: a test set designed around what this checkpoint measurably can do.

Every answer is carried by its own explicit sentence in the ticket, and no option description
shares vocabulary with a ticket of another class, so there is nothing for the model to confuse.
The set is a full factorial (3 requests x 2 deadlines x 3 tones = 18 combinations, 200 tickets),
so the three questions are independent and each can be read on its own.

Design decisions, each from a measurement made on this checkpoint:

  * the time-pressure question is `noul`, not `score`. noul is the only primitive that measurably
    works here (99.0% on a literally stated refund request), and Laya's own customer-triage preset
    (laya/presets.py, `triage_questions`) asks urgency as noul too.
  * score criteria carry no "Low:"/"High:" label. `render_options` already prefixes every score
    option with "level N:", so our previous criteria rendered as "level 0: Low: they can wait ...".
    The preset uses bare, short, parallel phrases; this file follows the preset.
  * option descriptions are kept within a few characters of each other in length. The service
    question's were 47/77/43 characters and the model picked the longest one 67% of the time.
"""
import collections
import itertools
import json
import random

GREETINGS = ["Madame, Monsieur,", "Bonjour,", "Bonjour au support,", "Bonjour,"]

# choice: what the customer asks for. Each paraphrase states it outright.
REQUESTS = {
    "refund": [
        "Pouvez-vous me rembourser la somme prélevée en double ?",
        "Je demande le remboursement de ce montant.",
        "Pouvez-vous me rembourser ce prélèvement ?",
    ],
    "cancel": [
        "Je souhaite résilier mon abonnement.",
        "Je veux mettre fin à mon abonnement.",
        "Pouvez-vous résilier mon abonnement dès que possible ?",
    ],
    "document": [
        "Pouvez-vous m'envoyer une copie de la facture ?",
        "Je souhaite obtenir le justificatif de ce paiement.",
        "J'aimerais recevoir le document correspondant.",
    ],
}

# noul: is a deadline stated, yes or no.
DEADLINES = {
    True: [
        "Il me faut une réponse avant vendredi.",
        "J'ai besoin d'une solution aujourd'hui.",
        "Cette demande est à traiter avant la fin de la semaine.",
    ],
    False: [
        "Rien ne presse, prenez votre temps.",
        "Prenez le temps qu'il faut, je ne suis pas pressé.",
        "Vous pouvez me répondre la semaine prochaine.",
    ],
}

# score: the customer's tone, stated in the surface text.
TONES = {
    "calme et neutre": [
        "Merci pour votre aide.",
        "Bonne journée à vous.",
        "Je vous remercie par avance.",
    ],
    "agacé": [
        "Cette situation commence à être pénible.",
        "Je trouve cela vraiment agaçant.",
        "Cela devient lassant à la longue.",
    ],
    "en colère": [
        "C'est inadmissible, je suis furieux.",
        "Je suis en colère, ce traitement est scandaleux.",
        "C'est une honte, je suis exaspéré.",
    ],
}

QUESTIONS = {
    "request": {
        "type": "choice",
        "instructions": "Que demande le client ?",
        "criteria": {
            "refund": "un remboursement ou un prélèvement annulé",
            "cancel": "mettre fin à l'abonnement du client",
            "document": "recevoir une facture ou un justificatif",
        },
    },
    "deadline": {
        "type": "noul",
        "instructions": "Le message indique-t-il une échéance ou une pression de temps ?",
    },
    "tone": {
        "type": "score",
        "instructions": "Sur quel ton le client s'exprime-t-il ?",
        "criteria": ["calme et neutre", "agacé", "en colère"],
    },
}

# Words that should appear only in tickets of their own class, for the contamination check below.
KEYWORDS = {
    "request=refund": ["rembours"],
    "request=cancel": ["résili", "mettre fin"],
    "request=document": ["facture", "justificatif", "document"],
    "deadline=True": ["avant vendredi", "aujourd'hui", "avant la fin"],
    "deadline=False": ["rien ne presse", "pas pressé", "semaine prochaine"],
    "tone=calme et neutre": ["merci", "remercie"],
    "tone=agacé": ["pénible", "agaçant", "lassant"],
    "tone=en colère": ["inadmissible", "furieux", "colère", "scandaleux", "honte", "exaspéré"],
}

TOTAL = 200

combos = list(itertools.product(REQUESTS, (True, False), TONES))
counts = {combo: (12 if index < 2 else 11) for index, combo in enumerate(combos)}

rows = []
for combo, count in counts.items():
    request, deadline, tone = combo
    for k in range(count):
        state = " ".join([
            GREETINGS[k % len(GREETINGS)],
            REQUESTS[request][k % len(REQUESTS[request])],
            DEADLINES[deadline][k % len(DEADLINES[deadline])],
            TONES[tone][k % len(TONES[tone])],
        ])
        rows.append({
            "state": state + " Ref TKT-%04d." % (len(rows) + 1),
            "expected": {"request": request, "deadline": deadline, "tone": tone},
        })

random.Random(0).shuffle(rows)
for index, row in enumerate(rows, start=1):
    row["id"] = "T-%04d" % index
tickets = [{"id": r["id"], "state": r["state"], "expected": r["expected"]} for r in rows]

with open("tickets-200-aligned.json", "w", encoding="utf-8") as handle:
    json.dump({"questions": QUESTIONS, "tickets": tickets}, handle, ensure_ascii=False, indent=2)

print(f"tickets: {len(tickets)}  combinations: {len(counts)} ({min(counts.values())}-{max(counts.values())} each)")
for name in QUESTIONS:
    dist = collections.Counter(str(t["expected"][name]) for t in tickets)
    print(f"  {name:<9} {dict(sorted(dist.items()))}")

print("\ncontamination check, one matrix per question: a class must not contain another class's words")
for question, classes in (("request", REQUESTS), ("deadline", DEADLINES), ("tone", TONES)):
    values = list(classes)
    print(f"\n  {question}\t({len(values)} {QUESTIONS[question]['type']} classes)")
    for row_value in values:
        cells = []
        for col_value in values:
            words = KEYWORDS[f"{question}={col_value}"]
            hits = sum(1 for t in tickets
                       if str(t["expected"][question]) == str(row_value)
                       and any(word in t["state"].lower() for word in words))
            cells.append(f"{hits:>8}")
        print(f"    {str(row_value):<18}" + "".join(cells))
    print(f"    {'':<18}" + "".join(f"{str(v)[:7]:>8}" for v in values))
print("\n    rows = true class, columns = whose words the ticket contains.")
print("    Values belong on the diagonal only; any off-diagonal hit is a confound.")
