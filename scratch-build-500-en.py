"""Build tickets-500-en.json from the first 500 tickets of tickets-1000.json: same structure
(greeting + problem + urgency closing + reference), same expected labels (translated), only the
language changes. All 43 distinct problem templates found in that slice are translated by hand
below, preserving the count and left-to-right order of their numeric placeholders so the real
amounts/order numbers/day counts carry over unchanged.
"""
import json
import re

GREETINGS = [
    "Madame, Monsieur,", "Bonjour,", "Bonjour au support,", "Bonjour l'équipe,",
    "Je vous contacte car",
]
GREETINGS_EN = {
    "Madame, Monsieur,": "Dear Sir or Madam,",
    "Bonjour,": "Hello,",
    "Bonjour au support,": "Hello support team,",
    "Bonjour l'équipe,": "Hello team,",
    "Je vous contacte car": "I am writing because",
}

# (french closing, english closing, urgency level) - same 9 clauses used in tickets-50-en.json
CLOSINGS = [
    ("Une intervention immédiate est nécessaire car toute mon équipe est bloquée.",
     "Immediate action is required because my whole team is blocked.", "High"),
    ("Merci d'agir aujourd'hui, les conséquences sont importantes.",
     "Please act today, the consequences are serious.", "High"),
    ("C'est urgent : cette situation bloque mon activité aujourd'hui.",
     "This is urgent: it is blocking my work today.", "High"),
    ("Merci de traiter cette demande dans les 48 heures.",
     "Please handle this within 48 hours.", "Moderate"),
    ("Le problème devient gênant et nécessite un retour rapide.",
     "This is becoming annoying and needs a quick turnaround.", "Moderate"),
    ("J'aimerais une solution avant la fin de la semaine.",
     "I'd like a solution before the end of the week.", "Moderate"),
    ("Je peux attendre quelques jours pour votre retour.",
     "I can wait a few days for your reply.", "Low"),
    ("Ce n'est pas urgent, une réponse cette semaine me convient.",
     "This isn't urgent, a response this week works for me.", "Low"),
    ("Merci de regarder ce point lorsque vous aurez le temps.",
     "Please look into this whenever you have time.", "Low"),
]
CLOSING_EN = {fr: (en, lvl) for fr, en, lvl in CLOSINGS}

SERVICE_EN = {"facturation": "billing", "technique": "technical", "livraison": "delivery"}
URGENCE_EN = {"Faible": "Low", "Modérée": "Moderate", "Élevée": "High"}

# Masked (digits -> '#') French body -> English body with the same placeholder count and order.
BODY_EN = {
    "La mise à jour a supprimé mes paramètres et l'application reste inutilisable. Je souhaite être remboursé.":
        "The latest update deleted my settings and the application is now unusable. I would like a refund.",
    "La remise promise n'a pas été appliquée sur la facture F-#. Je souhaite le remboursement de la différence.":
        "The promised discount was not applied to invoice F-#. I would like a refund of the difference.",
    "J'ai reçu un article qui ne correspond pas à ma commande CMD-#. Je demande son remboursement.":
        "I received an item that does not match my order CMD-#. I am requesting a refund.",
    "J'ai reçu le modèle bleu au lieu du modèle rouge commandé. Pouvez-vous organiser le retour ?":
        "I received the blue model instead of the red one I ordered. Can you arrange a return?",
    "Le courriel de réinitialisation du mot de passe n'arrive jamais.":
        "The password reset email never arrives.",
    "Les notifications ont disparu de l'application mobile alors qu'elles sont visibles sur le web.":
        "Notifications have disappeared from the mobile app even though they still show on the website.",
    "Pouvez-vous modifier l'adresse de facturation du compte CLI-# ?":
        "Can you update the billing address on account CLI-#?",
    "Le renouvellement de mon abonnement apparaît deux fois dans mon espace de facturation.":
        "My subscription renewal appears twice in my billing area.",
    "Le transporteur a retourné le colis à l'expéditeur sans me contacter. Veuillez me rembourser.":
        "The carrier returned the parcel to the sender without contacting me. Please refund me.",
    "Je souhaite comprendre le prélèvement de # euros visible sur mon relevé.":
        "I would like to understand the # euro charge showing on my statement.",
    "Le produit est arrivé cassé dans un carton écrasé. Je souhaite être remboursé.":
        "The product arrived broken inside a crushed box. I would like a refund.",
    "J'ai payé une option annuelle par erreur au lieu de l'option mensuelle. Je souhaite être remboursé.":
        "I paid for the annual option by mistake instead of the monthly one. I would like a refund.",
    "Le logiciel plante à chaque ouverture depuis # jours. Je demande le remboursement de la période inutilisable.":
        "The software has been crashing on every launch for # days. I am requesting a refund for the unusable period.",
    "La fonctionnalité d'export incluse dans mon offre ne fonctionne plus. Je souhaite le remboursement du mois courant.":
        "The export feature included in my plan no longer works. I would like a refund for the current month.",
    "La commande CMD-# a été annulée mais son paiement est toujours encaissé. Veuillez me rembourser.":
        "Order CMD-# was cancelled but the payment was still collected. Please refund me.",
    "Je ne peux plus accéder à mon compte CLI-# malgré plusieurs réinitialisations. Merci de rembourser mon abonnement.":
        "I can no longer access account CLI-# despite several password resets. Please refund my subscription.",
    "Je souhaite corriger l'adresse de livraison avant l'expédition de la commande CMD-#.":
        "I would like to correct the delivery address before order CMD-# ships.",
    "Le bouton Enregistrer reste désactivé après la saisie du formulaire.":
        "The Save button stays disabled after filling out the form.",
    "Le montant indiqué sur la facture F-# ne correspond pas au tarif de mon contrat.":
        "The amount shown on invoice F-# does not match my contract's rate.",
    "La visioconférence coupe après quelques secondes sur tous nos postes. Remboursez la licence de ce mois.":
        "Video calls drop after a few seconds on every workstation. Please refund this month's license.",
    "Mon abonnement CLI-# a été prélevé après sa résiliation. Merci de me rembourser.":
        "My subscription CLI-# was charged after I cancelled it. Please refund me.",
    "Un paiement de # euros que je n'ai pas autorisé apparaît sur mon compte. Je demande son remboursement.":
        "A payment of # euros that I did not authorize appears on my account. I am requesting a refund.",
    "La date de livraison annoncée pour la commande CMD-# vient de changer sans explication.":
        "The announced delivery date for order CMD-# just changed without explanation.",
    "Le transporteur indique livré, mais je n'ai pas reçu le colis CMD-#.":
        "The carrier says delivered, but I never received parcel CMD-#.",
    "Il manque la moitié des articles dans le colis CMD-#. Je souhaite un remboursement des produits absents.":
        "Half the items are missing from parcel CMD-#. I would like a refund for the missing products.",
    "Le paiement de # euros est refusé alors que ma banque confirme que la carte fonctionne.":
        "The # euro payment is declined even though my bank confirms the card works.",
    "La synchronisation reste bloquée à # % sur tous mes documents.":
        "Sync stays stuck at #% on all of my documents.",
    "Ma carte a été débitée deux fois de # euros pour la commande CMD-#. Je demande le remboursement du doublon.":
        "My card was charged twice for # euros for order CMD-#. I am requesting a refund for the duplicate.",
    "Le suivi indique livré, mais le colis CMD-# est introuvable. Merci de me rembourser.":
        "Tracking says delivered, but parcel CMD-# cannot be found. Please refund me.",
    "Le suivi de la commande CMD-# n'a pas été actualisé depuis # jours.":
        "Tracking for order CMD-# hasn't updated in # days.",
    "Le colis CMD-# est en retard de # jours. Je demande un remboursement complet.":
        "Parcel CMD-# is # days late. I am requesting a full refund.",
    "Le carton est arrivé endommagé et je voudrais organiser un échange.":
        "The box arrived damaged and I would like to arrange an exchange.",
    "L'application affiche l'erreur E# lorsque j'ouvre le dossier CLI-#.":
        "The application shows error E# when I open file CLI-#.",
    "Depuis la dernière mise à jour, l'export CSV produit un fichier vide.":
        "Since the last update, the CSV export produces an empty file.",
    "L'erreur E# empêche toute utilisation du service. Je demande le remboursement des jours perdus.":
        "Error E# prevents any use of the service. I am requesting a refund for the lost days.",
    "Je n'ai pas reçu la facture de mai pour mon abonnement CLI-#.":
        "I never received the May invoice for subscription CLI-#.",
    "L'erreur SYNC-# empêche toute utilisation du service. Je demande le remboursement des jours perdus.":
        "Error SYNC-# prevents any use of the service. I am requesting a refund for the lost days.",
    "L'erreur APP-# empêche toute utilisation du service. Je demande le remboursement des jours perdus.":
        "Error APP-# prevents any use of the service. I am requesting a refund for the lost days.",
    "L'application affiche l'erreur SYNC-# lorsque j'ouvre le dossier CLI-#.":
        "The application shows error SYNC-# when I open file CLI-#.",
    "Je n'ai pas reçu la facture de juin pour mon abonnement CLI-#.":
        "I never received the June invoice for subscription CLI-#.",
    "L'application affiche l'erreur APP-# lorsque j'ouvre le dossier CLI-#.":
        "The application shows error APP-# when I open file CLI-#.",
    "Je n'ai pas reçu la facture de avril pour mon abonnement CLI-#.":
        "I never received the April invoice for subscription CLI-#.",
    "Je n'ai pas reçu la facture de mars pour mon abonnement CLI-#.":
        "I never received the March invoice for subscription CLI-#.",
}


def strip_greeting(text):
    for g in GREETINGS:
        if text.startswith(g):
            return g, text[len(g):].strip()
    return None, text


def strip_closing(text):
    text = re.sub(r"Référence\s*$", "", text).strip()
    for fr, (en, lvl) in CLOSING_EN.items():
        if text.endswith(fr):
            return text[: -len(fr)].strip(), fr
    return text, None


def refill(template, numbers):
    it = iter(numbers)
    return re.sub(r"#", lambda m: next(it), template)


data = json.load(open("tickets-1000.json", "r", encoding="utf-8"))
tickets = data["tickets"][:500]

out_tickets = []
problems = []
for i, t in enumerate(tickets, start=1):
    idx = t["state"].find("TKT-")
    before = t["state"][:idx] if idx >= 0 else t["state"]
    greet, rest = strip_greeting(before)
    body, closing = strip_closing(rest)
    if greet is None or closing is None:
        problems.append(t["id"])
        continue
    masked = re.sub(r"\d+", "#", body)
    if masked not in BODY_EN:
        problems.append(t["id"])
        continue
    numbers = re.findall(r"\d+", body)
    body_en = refill(BODY_EN[masked], numbers)
    closing_en, level = CLOSING_EN[closing]
    greet_en = GREETINGS_EN[greet]
    ref = f"T-{i:04d}"
    state_en = f"{greet_en} {body_en} {closing_en} Ref TKT-{i:04d}."
    out_tickets.append({
        "id": ref,
        "state": state_en,
        "expected": {
            "service": SERVICE_EN[t["expected"]["service"]],
            "refund": t["expected"]["remboursement"],
            "urgency": URGENCE_EN[t["expected"]["urgence"]],
        },
    })

questions = {
    "service": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "Payment, direct debit, invoice or subscription",
            "technical": "Software malfunction, access issue, error or unavailable feature",
            "delivery": "Shipping, tracking, receipt or parcel status",
        },
    },
    "refund": {
        "type": "noul",
        "instructions": "Does the person explicitly ask for a refund?",
    },
    "urgency": {
        "type": "score",
        "instructions": "What response delay does the person expect?",
        "criteria": [
            "Low: they can wait, a few days or more",
            "Moderate: a short delay, 48 hours or by the end of the week",
            "High: today, they cannot wait",
        ],
    },
}

print(f"tickets in: {len(tickets)}  generated: {len(out_tickets)}  skipped: {len(problems)}")
if problems:
    print("skipped ids:", problems[:20])

json.dump({"questions": questions, "tickets": out_tickets},
          open("tickets-500-en.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("written: tickets-500-en.json")
