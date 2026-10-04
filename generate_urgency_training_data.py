"""Generate balanced urgency-only training examples without touching the benchmark tickets."""
from __future__ import annotations

import json
from pathlib import Path

BASE_PROBLEMS = [
    "Le paiement de ma dernière facture semble incorrect.",
    "Je ne reconnais pas le prélèvement affiché sur mon compte.",
    "Mon abonnement a été renouvelé au mauvais tarif.",
    "L'application affiche une erreur lors de la connexion.",
    "Je ne parviens plus à accéder à mon espace client.",
    "Le logiciel se ferme dès son ouverture.",
    "Mon colis n'est toujours pas arrivé.",
    "Le suivi de livraison n'a pas évolué depuis plusieurs jours.",
    "Le produit reçu ne correspond pas à ma commande.",
    "Je souhaite comprendre la procédure applicable à cette demande.",
    "Le problème persiste malgré plusieurs nouvelles tentatives.",
    "Je vous contacte afin d'obtenir une assistance sur mon dossier.",
]

PHRASES = {
    "Faible": [
        "Cette demande n'est pas urgente et peut être traitée dans les prochains jours.",
        "Vous pouvez revenir vers moi lorsque vous aurez un créneau disponible.",
        "Je n'ai pas de contrainte de délai particulière.",
        "Une réponse au cours de la semaine prochaine conviendra parfaitement.",
        "Le traitement peut attendre plusieurs jours sans conséquence.",
        "Prenez le temps nécessaire, aucune action immédiate n'est attendue.",
        "Il n'est pas nécessaire de traiter ce dossier rapidement.",
        "Un retour ultérieur me convient, la situation n'est pas bloquante.",
        "Je peux patienter jusqu'à la semaine prochaine.",
        "Cette demande peut être placée dans la file normale de traitement.",
        "Je souhaite simplement une réponse lorsque cela sera possible.",
        "Aucun délai court n'est requis pour résoudre ce point.",
        "Le sujet est mineur et peut attendre sans difficulté.",
        "Je consulterai votre réponse dans quelques jours.",
        "Il n'y a aucun impact immédiat sur mon activité.",
        "Cette demande est informative et ne nécessite pas de priorité.",
        "Vous pouvez traiter ce point après les dossiers urgents.",
        "Un retour à votre convenance sera suffisant.",
    ],
    "Modérée": [
        "Merci de me répondre dans les deux prochains jours.",
        "J'aurais besoin d'un retour sous quarante-huit heures.",
        "Une réponse avant la fin de cette semaine serait souhaitable.",
        "Merci de traiter ce point prochainement, sans nécessité d'intervenir aujourd'hui.",
        "Je souhaite une résolution dans un délai de deux à trois jours.",
        "Pourriez-vous revenir vers moi avant vendredi ?",
        "La demande devient gênante et mérite un traitement rapide cette semaine.",
        "Un retour dans les prochaines quarante-huit heures me conviendrait.",
        "Merci de ne pas laisser ce dossier attendre plus de quelques jours.",
        "J'ai besoin d'une réponse rapide, mais la situation n'est pas immédiatement bloquante.",
        "Le sujet doit être réglé cette semaine si possible.",
        "Une prise en charge sous deux jours serait adaptée.",
        "La situation est gênante et j'attends un retour rapide, mais elle n'est pas bloquante.",
        "Le problème mérite une réponse rapide sans intervention immédiate.",
        "Merci de revenir rapidement vers moi, une résolution aujourd'hui n'est toutefois pas nécessaire.",
        "Ce point devient contraignant et doit être examiné prochainement.",
        "Un traitement prioritaire dans les prochains jours serait apprécié.",
        "La demande presse un peu, mais elle peut attendre jusqu'à demain.",
    ],
    "Élevée": [
        "La situation est bloquante et doit être résolue immédiatement.",
        "J'ai besoin d'une intervention aujourd'hui.",
        "Merci de prendre en charge cette demande sans attendre.",
        "Toute mon activité est actuellement arrêtée, une action immédiate est indispensable.",
        "Ce problème empêche de travailler et nécessite une résolution dans la journée.",
        "La demande est critique et ne peut pas attendre demain.",
        "Veuillez intervenir au plus vite, le service est complètement bloqué.",
        "Une réponse immédiate est nécessaire pour reprendre l'activité.",
        "Ce dossier doit être traité en priorité absolue aujourd'hui.",
        "Je ne peux pas poursuivre tant que ce problème n'est pas corrigé.",
        "L'incident est en cours et exige une prise en charge urgente.",
        "Merci d'escalader immédiatement cette demande critique.",
        "La production est arrêtée et une correction immédiate est requise.",
        "Je suis totalement bloqué et j'attends une action dans l'heure.",
        "L'incident critique doit être pris en charge maintenant.",
        "Aucun travail ne peut continuer avant votre intervention aujourd'hui.",
        "La panne touche tous les utilisateurs et exige une réponse immédiate.",
        "Cette urgence absolue doit passer avant les autres demandes.",
    ],
}


def main() -> None:
    tickets = []
    counter = 1
    for label, phrases in PHRASES.items():
        for phrase_index, phrase in enumerate(phrases):
            for base_index, problem in enumerate(BASE_PROBLEMS):
                tickets.append({
                    "id": f"AUG-URG-{counter:04d}",
                    "state": f"Bonjour, {problem} {phrase}",
                    "expected": {"urgence": label},
                    "phrase_group": f"{label}-{phrase_index:02d}",
                    "base_group": base_index,
                })
                counter += 1
    output = {
        "purpose": "training-only urgency paraphrases; never use as benchmark data",
        "tickets": tickets,
    }
    Path("tickets-urgence-training-augmentation.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(tickets)} balanced training-only tickets")


if __name__ == "__main__":
    main()
