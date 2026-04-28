# =============================================================================
# AXIOM — engine/actions.py
# Définition des actions possibles au poker et calcul des actions légales
# pour un joueur donné dans un état de jeu donné.
# =============================================================================

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


# -----------------------------------------------------------------------------
# TYPES D'ACTIONS
# -----------------------------------------------------------------------------

class TypeAction(Enum):
    FOLD   = auto()   # se coucher
    CHECK  = auto()   # passer sans miser (seulement si pas de mise à suivre)
    CALL   = auto()   # suivre la mise en cours
    RAISE  = auto()   # relancer (inclut le bet initial si personne n'a misé)
    ALL_IN = auto()   # tapis complet


# -----------------------------------------------------------------------------
# DATACLASS ACTION
# Une action = un type + un montant (0 pour fold/check/call)
# -----------------------------------------------------------------------------

@dataclass
class Action:
    type    : TypeAction
    montant : int = 0    # montant total de la mise (pas juste le supplément)

    def __repr__(self):
        if self.type in (TypeAction.FOLD, TypeAction.CHECK):
            return self.type.name
        return f"{self.type.name}({self.montant})"


# -----------------------------------------------------------------------------
# CALCUL DES ACTIONS LÉGALES
# -----------------------------------------------------------------------------

def actions_legales(joueur, mise_a_suivre: int, pot: int, mise_min_raise: int) -> list:
    """
    Retourne la liste des Actions légales pour un joueur donné.

    joueur         : objet Joueur (voir player.py)
    mise_a_suivre  : montant total que le joueur doit atteindre pour suivre
                     (= la plus grosse mise du tour en cours)
    pot            : taille du pot actuel (pour calculer les raises possibles)
    mise_min_raise : montant minimum d'une relance (= dernier raise * 2)

    Règles Texas Hold'em No Limit :
    - On peut toujours fold
    - Check possible seulement si mise_a_suivre == joueur.mise_tour
    - Call possible si mise_a_suivre > joueur.mise_tour ET joueur a assez de jetons
    - Raise possible si joueur a assez pour dépasser la mise_a_suivre + min_raise
    - All-in toujours possible si joueur a des jetons
    """
    actions = []
    a_payer = mise_a_suivre - joueur.mise_tour   # ce qu'il reste à payer pour suivre

    # FOLD : toujours possible si on doit payer quelque chose
    if a_payer > 0:
        actions.append(Action(TypeAction.FOLD))

    # CHECK : possible seulement si rien à payer
    if a_payer == 0:
        actions.append(Action(TypeAction.CHECK))

    # CALL : possible si on doit payer et qu'on a assez pour payer sans all-in
    if 0 < a_payer < joueur.stack:
        actions.append(Action(TypeAction.CALL, montant=mise_a_suivre))

    # ALL-IN : toujours possible si on a des jetons
    if joueur.stack > 0:
        montant_allin = joueur.mise_tour + joueur.stack
        # on n'ajoute all-in que si c'est différent du call
        if montant_allin != mise_a_suivre:
            actions.append(Action(TypeAction.ALL_IN, montant=montant_allin))
        elif a_payer >= joueur.stack:
            # le call EST un all-in
            actions.append(Action(TypeAction.ALL_IN, montant=montant_allin))

    # RAISE : possible si le joueur peut miser plus que la mise min
    montant_raise_min = mise_a_suivre + mise_min_raise
    if joueur.stack > a_payer and (joueur.mise_tour + joueur.stack) > montant_raise_min:
        actions.append(Action(TypeAction.RAISE, montant=montant_raise_min))

    return actions
