"""Audit the test sample itself, not the model.

1. Class balance.
2. Cross-validated bag-of-words Naive Bayes: how much of the label is recoverable
   from the text by a trivial learner?  ~95% => sample is fine, Laya is the problem.
3. Per-template label determinism: does one problem sentence always get one label?
4. urgency: tokens of the body alone, per level -> do bodies carry urgency signal at all?
"""
import json
import math
import random
import re
from collections import Counter, defaultdict

DATA = "tickets-500-en.json"

GREETINGS = [
    "Dear Sir or Madam,", "Hello,", "Hello support team,", "Hello team,",
    "I am writing because",
]
CLOSINGS = [
    "Immediate action is required because my whole team is blocked.",
    "Please act today, the consequences are serious.",
    "This is urgent: it is blocking my work today.",
    "Please handle this within 48 hours.",
    "This is becoming annoying and needs a quick turnaround.",
    "I'd like a solution before the end of the week.",
    "I can wait a few days for your reply.",
    "This isn't urgent, a response this week works for me.",
    "Please look into this whenever you have time.",
]
URGENT_WORDS = {
    "blocked", "urgent", "immediately", "immediate", "brocken", "broken", "error",
    "unusable", "crash", "crashing", "stuck", "cannot", "can't", "never", "missing",
    "damaged", "declined", "refused", "wrong", "unavailable", "lost",
}


def tokens(s):
    return re.findall(r"[a-z0-9#]+", s.lower())


def strip_parts(state):
    """Return the problem sentence alone, digits masked."""
    body = state
    for g in GREETINGS:
        if body.startswith(g):
            body = body[len(g):]
            break
    for c in CLOSINGS:
        if c in body:
            body = body.replace(c, " ")
            break
    body = re.sub(r"Ref\s+TKT-\d+\.?", "", body)
    body = re.sub(r"\d+", "#", body)
    return " ".join(body.split())


def cv_naive_bayes(texts, labels, k=5, seed=0):
    idx = list(range(len(texts)))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    correct = 0
    for f in folds:
        fset = set(f)
        train = [i for i in idx if i not in fset]
        word_counts = defaultdict(Counter)
        totals = Counter()
        prior = Counter()
        vocab = set()
        for i in train:
            y = labels[i]
            prior[y] += 1
            for w in tokens(texts[i]):
                word_counts[y][w] += 1
                vocab.add(w)
        for y in word_counts:
            totals[y] = sum(word_counts[y].values())
        V = len(vocab)
        classes = list(prior)
        for i in f:
            ws = tokens(texts[i])
            best, best_ll = None, None
            for y in classes:
                ll = math.log(prior[y] / len(train))
                denom = totals[y] + V
                for w in ws:
                    ll += math.log((word_counts[y][w] + 1) / denom)
                if best_ll is None or ll > best_ll:
                    best, best_ll = y, ll
            correct += best == labels[i]
    return correct / len(texts)


data = json.load(open(DATA, encoding="utf-8"))
tickets = data["tickets"]
print(f"dataset: {DATA}  tickets: {len(tickets)}  questions: {list(data['questions'])}")

print("\n=== 1. class balance ===")
for q in data["questions"]:
    c = Counter(str(t["expected"][q]) for t in tickets)
    total = sum(c.values())
    parts = "  ".join(f"{k}:{v} ({v/total:.0%})" for k, v in c.most_common())
    print(f"{q:<9} {parts}")

print("\n=== 2. bag-of-words Naive Bayes, 5-fold CV ===")
print("    (trivial learner: if this scores low, the labels are not in the text)")
for q in data["questions"]:
    labels = [str(t["expected"][q]) for t in tickets]
    full = [t["state"] for t in tickets]
    body = [strip_parts(t["state"]) for t in tickets]
    acc_full = cv_naive_bayes(full, labels)
    acc_body = cv_naive_bayes(body, labels)
    print(f"{q:<9} full text {acc_full:6.1%}   problem sentence only {acc_body:6.1%}")

print("\n=== 3. service: one problem sentence = one label? ===")
by_template = defaultdict(Counter)
for t in tickets:
    by_template[strip_parts(t["state"])][t["expected"]["service"]] += 1
conflicted = {k: v for k, v in by_template.items() if len(v) > 1}
print(f"distinct problem sentences: {len(by_template)}   with more than one label: {len(conflicted)}")
for k, v in conflicted.items():
    print(f"  AMBIGUOUS {dict(v)}  <- {k}")
print("  label per problem sentence:")
for k, v in sorted(by_template.items(), key=lambda kv: -sum(kv[1].values())):
    label = v.most_common(1)[0][0]
    print(f"    {label:<10} n={sum(v.values()):<4} {k}")

print("\n=== 4. urgency: what the problem sentence looks like, per level ===")
for lvl in ("Low", "Moderate", "High"):
    bodies = [strip_parts(t["state"]) for t in tickets if t["expected"]["urgency"] == lvl]
    words = Counter(w for b in bodies for w in set(tokens(b)))
    urgent = sum(1 for b in bodies if URGENT_WORDS & set(tokens(b)))
    top = "  ".join(f"{w}:{n}" for w, n in words.most_common(12))
    print(f"{lvl:<9} n={len(bodies):<4} containing alarming words: {urgent/len(bodies):5.0%}")
    print(f"          body tokens: {top}")
