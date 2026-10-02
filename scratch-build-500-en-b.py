"""Build tickets-500-en-b.json: a second, independent English test set for the same three questions.

Purpose: check whether decision rules fitted on tickets-500-en.json (refund threshold 0.80) hold on
tickets the fit never saw. For that the questions are copied byte-for-byte from the first file, and
every sentence (greeting, problem, refund request, urgency closing) is new wording. The script
refuses to write the file if any sentence, digits masked, also appears in the first set.

Labels are stated in the text and independent of each other (full factorial over service x refund x
urgency, balanced), so each question can be read on its own.
"""
import collections
import itertools
import json
import random
import re

OLD = "tickets-500-en.json"
OUT = "tickets-500-en-b.json"
TOTAL = 500
rng = random.Random(20261002)

GREETINGS = ["Hi there,", "Good morning,", "Hello customer care,", "Dear support,",
             "To whom it may concern,", "Hi,"]

PROBLEMS = {
    "billing": [
        "My bank statement shows two charges of {amt} euros for the same subscription.",
        "Invoice INV-{n} lists a plan I never signed up for.",
        "The direct debit for {amt} euros went through on the wrong date this month.",
        "I was billed at the old price although my new rate has already started.",
        "My payment of {amt} euros was declined, yet the money left my account.",
        "I cannot find the receipt for my last payment in the billing portal.",
        "The VAT number on invoice INV-{n} is wrong.",
        "My subscription renewed automatically although I had switched to the monthly plan.",
        "There is an unexplained fee of {amt} euros on my latest invoice.",
        "I need to change the card used for my subscription.",
        "The billing portal shows an overdue balance, but I paid on time.",
        "The company name on invoice INV-{n} is misspelled.",
        "I would like to know why my monthly payment went up by {amt} euros.",
        "A promotional code was not applied when I paid for my subscription.",
    ],
    "technical": [
        "The mobile app crashes as soon as I tap the dashboard icon.",
        "I keep getting a blank page after logging in on the web portal.",
        "Two-factor codes never reach my phone, so I cannot sign in.",
        "The report builder freezes when I select more than {rows} rows.",
        "Dark mode resets itself every time I reopen the application.",
        "The search bar returns no results even for items I know exist.",
        "File uploads fail with error code {code} halfway through.",
        "My shared folders disappeared from the sidebar after the last release.",
        "The desktop client keeps asking me to sign in again every few minutes.",
        "Exported spreadsheets open with scrambled characters.",
        "The API times out whenever I send more than a handful of requests.",
        "Calendar sync stopped working between the app and my phone.",
        "Autosave does not seem to work, I have lost my last changes twice.",
        "The installer stops at {pct} percent on my laptop.",
    ],
    "delivery": [
        "My parcel has been marked as out for delivery for {days} days now.",
        "The courier left the package at a neighbour's address without telling me.",
        "The tracking page has not changed since the parcel reached the depot.",
        "I received only two boxes out of the three I ordered.",
        "The delivery slot I booked for Tuesday was missed with no notice.",
        "The box arrived with its seal broken and the contents scattered.",
        "I need to redirect order {n} to my office instead of my home.",
        "The pickup point says it has no trace of my parcel.",
        "My package was delivered to the wrong building number.",
        "The courier asked for a signature but nobody came to the door.",
        "The wrong colour of chair was delivered with order {n}.",
        "My parcel has been stuck at the customs office for {days} days.",
        "I would like to change the delivery date for order {n}.",
        "The driver could not find my address and took the parcel back to the warehouse.",
    ],
}

REFUNDS = {
    "billing": [
        "I would like a refund of the overcharged amount.",
        "Please refund the duplicate payment to my card.",
        "I am asking for my money back for this charge.",
        "Could you reimburse me for the difference?",
        "Please issue a refund for this month.",
        "I'm requesting a full refund of that payment.",
    ],
    "technical": [
        "Please refund this month's subscription since the tool is unusable.",
        "I'd like my money back for the days I could not work.",
        "I am requesting a refund for the faulty release.",
        "Could you reimburse me for the period without access?",
        "Kindly refund the fee for this period.",
    ],
    "delivery": [
        "I would like a refund for the parcel that never arrived.",
        "Please refund the shipping costs.",
        "I want my money back for the damaged goods.",
        "Could you reimburse me for the missing items?",
        "I'm asking for a full refund of order {n}.",
    ],
}

CLOSINGS = {
    "Low": [
        "No rush at all on my side.",
        "Feel free to get back to me next week.",
        "It can wait until you have a free moment.",
        "I'm in no hurry for an answer.",
        "Take your time, nothing depends on this right now.",
        "This isn't pressing, any day this month is fine.",
    ],
    "Moderate": [
        "I'd appreciate an answer in the next couple of days.",
        "Could you come back to me by Thursday?",
        "A reply within two days would be great.",
        "Please get back to me in the next day or two.",
        "It would help to have this sorted by the end of the week.",
        "I'd like this resolved within the next 48 hours.",
    ],
    "High": [
        "I need this resolved today, my customers are waiting.",
        "This can't wait, please respond within the hour.",
        "We are completely stuck until this is fixed, so please act right now.",
        "Urgent: I need an answer before the end of the day.",
        "Please treat this as top priority, it is blocking everything today.",
        "I have a deadline this afternoon and need this fixed immediately.",
    ],
}

REFUND_WORDS = re.compile(r"refund|reimburs|money back", re.I)


def fill(template):
    return template.format(
        n=rng.randint(10000, 99999), amt=rng.randint(9, 240), code=rng.randint(100, 599),
        pct=rng.randint(12, 94), days=rng.randint(3, 9), rows=rng.randint(200, 900))


combos = list(itertools.product(("billing", "technical", "delivery"), (True, False),
                                ("Low", "Moderate", "High")))
rng.shuffle(combos)
base, extra = divmod(TOTAL, len(combos))

rows = []
for index, (service, refund, level) in enumerate(combos):
    for _ in range(base + (1 if index < extra else 0)):
        tail = [fill(rng.choice(CLOSINGS[level]))]
        if refund:
            tail.insert(0, fill(rng.choice(REFUNDS[service])))
            if rng.random() < 0.5:
                tail.reverse()
        text = " ".join([rng.choice(GREETINGS), fill(rng.choice(PROBLEMS[service]))] + tail)
        rows.append({"state": text, "expected": {"service": service, "refund": refund, "urgency": level}})

rng.shuffle(rows)
tickets = []
for number, row in enumerate(rows, start=1):
    tickets.append({"id": "N-%04d" % number,
                    "state": "%s Ref TKT-%04d." % (row["state"], number),
                    "expected": row["expected"]})

for ticket in tickets:
    has_refund_words = bool(REFUND_WORDS.search(ticket["state"]))
    assert has_refund_words == ticket["expected"]["refund"], ticket["id"]
assert len({t["state"] for t in tickets}) == TOTAL

old = json.load(open(OLD, encoding="utf-8"))
OLD_GREETINGS = ["Dear Sir or Madam,", "Hello support team,", "Hello team,", "Hello,", "I am writing because"]


def sentences(state, greetings):
    for greeting in greetings:
        if state.startswith(greeting):
            state = state[len(greeting):].strip()
            break
    state = re.sub(r"Ref TKT-\d+\.", "", state)
    return {re.sub(r"\d+", "#", s.strip()) for s in re.split(r"(?<=[.?!])\s+", state) if s.strip()}


old_sentences = set().union(*(sentences(t["state"], OLD_GREETINGS) for t in old["tickets"]))
new_sentences = set().union(*(sentences(t["state"], GREETINGS) for t in tickets))
shared = old_sentences & new_sentences
print(f"distinct sentences  first set: {len(old_sentences)}   this set: {len(new_sentences)}   shared: {len(shared)}")
for sentence in sorted(shared):
    print("  SHARED:", sentence)
if shared:
    raise SystemExit("not written: sentences are reused from the first set")

json.dump({"questions": old["questions"], "tickets": tickets},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print(f"\nwritten {OUT}: {len(tickets)} tickets, questions copied from {OLD}")
for name in ("service", "refund", "urgency"):
    counts = collections.Counter(str(t["expected"][name]) for t in tickets)
    print(f"  {name:<8} {dict(sorted(counts.items()))}")
