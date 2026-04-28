# =============================================================================
# AXIOM — engine/card.py
# Représentation d'une carte et génération du deck complet.
# On utilise la librairie treys pour encoder les cartes en entiers 32 bits
# (format très rapide pour l'évaluation des mains).
# =============================================================================

import random
from treys import Card, Deck


# -----------------------------------------------------------------------------
# CONSTANTES
# -----------------------------------------------------------------------------

# Les 4 couleurs (suits) du jeu
COULEURS = ['s', 'h', 'd', 'c']   # spades, hearts, diamonds, clubs

# Les 13 valeurs (ranks)
VALEURS  = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


# -----------------------------------------------------------------------------
# FONCTIONS UTILITAIRES
# -----------------------------------------------------------------------------

def creer_carte(valeur: str, couleur: str) -> int:
    """
    Crée une carte au format treys (entier 32 bits).
    Exemple : creer_carte('A', 's') → As de pique
    """
    return Card.new(valeur + couleur)


def carte_en_texte(carte: int) -> str:
    """
    Convertit une carte treys en texte lisible.
    Exemple : carte_en_texte(carte) → 'As' (As de pique)
    """
    return Card.int_to_pretty_str(carte)


def cartes_en_texte(cartes: list) -> str:
    """
    Convertit une liste de cartes en texte lisible.
    Exemple : ['As', 'Kh', 'Qd']
    """
    return ' '.join(carte_en_texte(c) for c in cartes)


# -----------------------------------------------------------------------------
# CLASSE DECK
# -----------------------------------------------------------------------------

class DeckAXIOM:
    """
    Un deck de 52 cartes mélangées.
    Fournit les méthodes pour distribuer les cartes aux joueurs et au board.
    """

    def __init__(self):
        self.deck = Deck()          # crée et mélange automatiquement les 52 cartes
        self.cartes_restantes = list(self.deck.cards)

    def melanger(self):
        """Remet toutes les cartes dans le deck et mélange."""
        self.deck = Deck()
        self.cartes_restantes = list(self.deck.cards)

    def distribuer(self, nb: int) -> list:
        """
        Tire 'nb' cartes du dessus du deck et les retire du deck.
        Retourne une liste d'entiers (format treys).
        """
        if nb > len(self.cartes_restantes):
            raise ValueError(f"Plus assez de cartes dans le deck (demandé: {nb}, restant: {len(self.cartes_restantes)})")
        tirees = self.cartes_restantes[:nb]
        self.cartes_restantes = self.cartes_restantes[nb:]
        return tirees

    def nb_restantes(self) -> int:
        """Retourne le nombre de cartes restantes dans le deck."""
        return len(self.cartes_restantes)

    def __repr__(self):
        return f"DeckAXIOM({self.nb_restantes()} cartes restantes)"
