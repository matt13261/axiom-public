# =============================================================================
# AXIOM — engine/hand_evaluator.py
# Évaluation des mains de poker via la librairie treys.
# Détermine qui gagne entre plusieurs mains, et calcule les équités
# (probabilité de gagner) par simulation Monte Carlo.
# =============================================================================

import random
from treys import Evaluator, Card, Deck
from engine.card import DeckAXIOM


# -----------------------------------------------------------------------------
# ÉVALUATEUR SINGLETON
# (on crée une seule instance réutilisée partout — plus rapide)
# -----------------------------------------------------------------------------

_evaluateur = Evaluator()


# -----------------------------------------------------------------------------
# FONCTIONS D'ÉVALUATION
# -----------------------------------------------------------------------------

def score_main(cartes_joueur: list, board: list) -> int:
    """
    Calcule le score d'une main (plus le score est BAS, meilleure est la main).
    treys utilise cette convention : 1 = quinte flush royale (meilleur), 7462 = pire main.

    cartes_joueur : liste de 2 cartes (format treys int)
    board         : liste de 3, 4 ou 5 cartes communes
    """
    return _evaluateur.evaluate(board, cartes_joueur)


def classe_main(score: int) -> str:
    """
    Retourne le nom de la combinaison à partir du score treys.
    Exemple : 'Flush', 'Straight', 'Two Pair', etc.
    """
    return _evaluateur.class_to_string(_evaluateur.get_rank_class(score))


def determiner_gagnants(joueurs_actifs: list, board: list) -> list:
    """
    Détermine le(s) gagnant(s) parmi les joueurs encore en jeu.

    joueurs_actifs : liste d'objets Player (voir player.py) encore dans la main
    board          : liste de 5 cartes communes (format treys int)

    Retourne une liste des joueurs gagnants (peut être plusieurs en cas d'égalité).
    """
    meilleur_score = None
    gagnants = []

    for joueur in joueurs_actifs:
        score = score_main(joueur.cartes, board)
        if meilleur_score is None or score < meilleur_score:
            meilleur_score = score
            gagnants = [joueur]
        elif score == meilleur_score:
            gagnants.append(joueur)   # égalité → partage du pot

    return gagnants


def calculer_equite(cartes_joueur: list, cartes_adversaires: list, board: list, nb_simulations: int = 1000, nb_adversaires: int = None) -> float:
    """
    Calcule l'équité d'une main par simulation Monte Carlo.
    L'équité = probabilité de gagner ou faire égalité.

    cartes_joueur     : 2 cartes du joueur dont on calcule l'équité
    cartes_adversaires: liste des cartes des adversaires (peut être vide = inconnu)
    board             : cartes communes déjà visibles (0 à 5 cartes)
    nb_simulations    : nombre de simulations aléatoires
    nb_adversaires    : nombre d'adversaires à simuler (utilisé si cartes_adversaires
                        est vide). Si None, fallback sur 1 adversaire (heads-up).
                        Pour 3-max, passer nb_adversaires=2.

    Retourne un float entre 0.0 et 1.0.
    """
    # Cartes déjà utilisées = les nôtres + board visible + adversaires connus
    cartes_connues = set(cartes_joueur + board + cartes_adversaires)

    # Pool de cartes disponibles pour compléter le board et les mains adverses.
    # GetFullDeck() retourne un ordre fixe (pas de shuffle) — nécessaire pour
    # que random.seed() dans bucket_postflop rende le résultat déterministe.
    toutes_cartes = Deck.GetFullDeck()
    cartes_disponibles = [c for c in toutes_cartes if c not in cartes_connues]

    nb_victoires = 0
    # Si des cartes adverses sont fournies, le nombre d'adversaires en découle.
    # Sinon, on utilise nb_adversaires explicite (par défaut : 1 = heads-up).
    if cartes_adversaires:
        nb_adv = max(1, len(cartes_adversaires) // 2)
    else:
        nb_adv = nb_adversaires if nb_adversaires is not None else 1

    for _ in range(nb_simulations):
        tirage = random.sample(cartes_disponibles, (5 - len(board)) + nb_adv * 2 - len(cartes_adversaires))

        # Compléter le board jusqu'à 5 cartes
        board_complet = board + tirage[:5 - len(board)]
        idx = 5 - len(board)

        # Compléter les mains adverses inconnues
        mains_adv = []
        if cartes_adversaires:
            # On regroupe les cartes adverses connues par paires
            for i in range(0, len(cartes_adversaires), 2):
                mains_adv.append(cartes_adversaires[i:i+2])
        else:
            # Adversaires inconnus → on tire leurs cartes aléatoirement
            for i in range(nb_adv):
                mains_adv.append(tirage[idx:idx+2])
                idx += 2

        # Évaluer notre main
        notre_score = score_main(cartes_joueur, board_complet)

        # Comparer avec chaque adversaire
        on_gagne = all(notre_score <= score_main(adv, board_complet) for adv in mains_adv)
        if on_gagne:
            nb_victoires += 1

    return nb_victoires / nb_simulations
