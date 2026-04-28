# =============================================================================
# AXIOM — engine/player.py
# Représentation d'un joueur à la table.
# Gère le stack, les cartes, la position et le statut dans la main.
# =============================================================================

from enum import Enum, auto


# -----------------------------------------------------------------------------
# STATUTS D'UN JOUEUR DANS UNE MAIN
# -----------------------------------------------------------------------------

class StatutJoueur(Enum):
    ACTIF     = auto()   # encore dans la main, peut agir
    FOLD      = auto()   # a couché ses cartes cette main
    ALL_IN    = auto()   # a misé tout son stack, ne peut plus agir
    ELIMINE   = auto()   # n'a plus de jetons, éliminé du tournoi


# -----------------------------------------------------------------------------
# TYPES DE JOUEUR
# -----------------------------------------------------------------------------

class TypeJoueur(Enum):
    HUMAIN = auto()   # contrôlé par l'interface graphique
    AXIOM  = auto()   # contrôlé par l'IA


# -----------------------------------------------------------------------------
# CLASSE JOUEUR
# -----------------------------------------------------------------------------

class Joueur:
    """
    Représente un joueur à la table AXIOM.

    Attributs :
        nom         : nom affiché à l'écran
        type        : HUMAIN ou AXIOM
        stack       : jetons restants
        cartes      : liste de 2 cartes (format treys int), vide hors main
        statut      : StatutJoueur (ACTIF, FOLD, ALL_IN, ELIMINE)
        mise_tour   : montant misé dans le tour en cours (reset à chaque street)
        position    : index de position à la table (0, 1, 2)
    """

    def __init__(self, nom: str, type_joueur: TypeJoueur, stack: int, position: int):
        self.nom               = nom
        self.type              = type_joueur
        self.stack             = stack
        self.cartes            = []
        self.statut            = StatutJoueur.ACTIF
        self.mise_tour         = 0   # montant misé dans le tour en cours (reset chaque street)
        self.contribution_main = 0   # total investi depuis le début de la main courante
        self.position          = position

    # ------------------------------------------------------------------
    # ACTIONS DE BASE SUR LE STACK
    # ------------------------------------------------------------------

    def miser(self, montant: int) -> int:
        """
        Retire 'montant' du stack du joueur.
        Si montant >= stack → all-in automatique.
        Retourne le montant réellement misé.
        """
        montant_reel = min(montant, self.stack)
        self.stack             -= montant_reel
        self.mise_tour         += montant_reel
        self.contribution_main += montant_reel

        if self.stack == 0:
            self.statut = StatutJoueur.ALL_IN

        return montant_reel

    def recevoir(self, montant: int):
        """Ajoute des jetons au stack (gain de pot)."""
        self.stack += montant

    def recevoir_cartes(self, cartes: list):
        """Distribue 2 cartes au joueur."""
        self.cartes = cartes

    # ------------------------------------------------------------------
    # RÉINITIALISATION ENTRE LES MAINS
    # ------------------------------------------------------------------

    def reinitialiser_main(self):
        """
        Remet le joueur à zéro entre deux mains.
        Conserve le stack et la position, efface cartes et mises.
        """
        self.cartes            = []
        self.mise_tour         = 0
        self.contribution_main = 0
        if self.stack > 0:
            self.statut = StatutJoueur.ACTIF
        else:
            self.statut = StatutJoueur.ELIMINE

    def reinitialiser_tour(self):
        """Remet la mise du tour à 0 (entre deux streets : preflop→flop→etc.)"""
        self.mise_tour = 0

    # ------------------------------------------------------------------
    # PROPRIÉTÉS UTILES
    # ------------------------------------------------------------------

    @property
    def est_actif(self) -> bool:
        return self.statut == StatutJoueur.ACTIF

    @property
    def est_elimine(self) -> bool:
        return self.statut == StatutJoueur.ELIMINE

    @property
    def peut_agir(self) -> bool:
        """Un joueur peut agir s'il est ACTIF (pas fold, pas all-in, pas éliminé)."""
        return self.statut == StatutJoueur.ACTIF

    def __repr__(self):
        return (f"Joueur({self.nom} | stack={self.stack} | "
                f"statut={self.statut.name} | pos={self.position})")
