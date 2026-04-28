# =============================================================================
# AXIOM — abstraction/action_abstraction.py
# Abstraction des actions de mise.
#
# Pourquoi ? En No Limit Hold'em, un joueur peut théoriquement miser n'importe
# quel montant entre le min-raise et son stack. C'est un espace continu et
# infini d'actions possibles — impossible à traiter pour MCCFR.
#
# Solution : on réduit toutes les mises à 6 tailles exprimées en fraction du pot
# (ex: 33%, 50%, 75%, 100%, 150%, 200%) + all-in.
# Ce sont les mêmes fractions que dans config/settings.py (TAILLES_MISE).
#
# Ce module est utilisé par MCCFR (Phase 3) pour construire l'arbre de jeu.
# =============================================================================

from config.settings import TAILLES_MISE, ALL_IN
from engine.actions import Action, TypeAction


# =============================================================================
# PSEUDO-HARMONIC ACTION TRANSLATION  (Ganzfried & Sandholm, 2013)
#
# Problème : si l'adversaire mise une fraction x qui ne correspond à aucune
# taille abstraite, comment l'interpréter ? L'approche naïve consiste à
# arrondir à la taille la plus proche ; c'est ce qu'on appelle "action
# mapping" (Libratus l'a explicitement abandonné).
#
# Pseudo-harmonic mapping : on répartit la proba entre les deux tailles
# abstraites A et B qui encadrent x (A ≤ x ≤ B) selon :
#
#        (B - x) × (1 + A)
#   P(A) = ─────────────────
#          (B - A) × (1 + x)
#
#   P(B) = 1 - P(A)
#
# Cette formule est celle utilisée dans Libratus et Pluribus. Elle empêche
# un adversaire d'exploiter les "trous" de notre abstraction en misant
# systématiquement entre nos buckets.
# =============================================================================

def pseudo_harmonic_mapping(x: float, A: float, B: float) -> tuple:
    """
    Retourne (proba_A, proba_B) pour une fraction observée x entre A et B.

    x : fraction de mise observée (montant_raise / pot) — positive.
    A : fraction abstraite inférieure la plus proche.
    B : fraction abstraite supérieure la plus proche.
    A < B, et normalement A ≤ x ≤ B (sinon clamp).

    Retourne (p_A, p_B) avec p_A + p_B = 1.0.
    """
    if B <= A:
        return (1.0, 0.0)
    # Clamp : si x est hors [A, B], on renvoie tout à la borne la plus proche
    if x <= A:
        return (1.0, 0.0)
    if x >= B:
        return (0.0, 1.0)
    p_A = ((B - x) * (1.0 + A)) / ((B - A) * (1.0 + x))
    p_A = max(0.0, min(1.0, p_A))
    return (p_A, 1.0 - p_A)


def traduire_fraction(fraction: float, tailles: list) -> list:
    """
    Traduit une fraction de mise observée vers une distribution pseudo-harmonic
    sur les indices des `tailles` abstraites.

    fraction : fraction observée (montant_raise / pot).
    tailles  : liste ordonnée croissante de fractions abstraites
               (ex : [0.20, 0.35, 0.65, 1.0, 1.5]).

    Retourne une liste [(idx_taille, probabilite)] non-nulle :
      - 1 couple si fraction coïncide ou est hors bornes
      - 2 couples (les deux voisins) sinon
    """
    if not tailles:
        return []

    # Hors bornes : attribution à la taille extrême la plus proche
    if fraction <= tailles[0]:
        return [(0, 1.0)]
    if fraction >= tailles[-1]:
        return [(len(tailles) - 1, 1.0)]

    # Trouver les deux voisins qui encadrent la fraction
    for i in range(len(tailles) - 1):
        A = tailles[i]
        B = tailles[i + 1]
        if A <= fraction <= B:
            p_A, p_B = pseudo_harmonic_mapping(fraction, A, B)
            out = []
            if p_A > 0.0:
                out.append((i,     p_A))
            if p_B > 0.0:
                out.append((i + 1, p_B))
            return out

    # Filet de sécurité (ne devrait jamais arriver)
    return [(len(tailles) - 1, 1.0)]


# -----------------------------------------------------------------------------
# CLASSE PRINCIPALE
# -----------------------------------------------------------------------------

class AbstractionAction:
    """
    Calcule l'ensemble réduit d'actions disponibles pour un joueur donné.

    Au lieu de toutes les tailles de raise possibles, on ne garde que :
    - FOLD (si mise à suivre > 0)
    - CHECK (si mise à suivre == 0)
    - CALL (si mise à suivre > 0 et joueur a assez)
    - RAISE à X% du pot pour chaque fraction dans TAILLES_MISE
    - ALL_IN (toujours, si le joueur a des jetons)

    C'est exactement ce que fait actions_legales_abstraites() dans engine/actions.py,
    mais encapsulé ici de façon plus propre pour être utilisé par MCCFR.
    """

    def __init__(self, tailles_mise: list = None):
        """
        tailles_mise : liste de fractions du pot (ex: [0.33, 0.5, 0.75, 1.0, 1.5, 2.0])
                       Si None, utilise TAILLES_MISE depuis settings.py
        """
        self.tailles_mise = tailles_mise if tailles_mise is not None else TAILLES_MISE

    def actions_abstraites(self, joueur, mise_a_suivre: int, pot: int,
                           mise_min_raise: int) -> list:
        """
        Retourne la liste réduite des Actions légales pour MCCFR.

        joueur         : objet Joueur (engine/player.py)
        mise_a_suivre  : montant total à suivre (plus grosse mise du tour)
        pot            : taille du pot actuel
        mise_min_raise : montant minimum d'un raise

        Retourne une liste d'objets Action (engine/actions.py).
        """
        actions = []
        a_payer = mise_a_suivre - joueur.mise_tour

        # --- FOLD : possible seulement si on doit payer quelque chose ---
        if a_payer > 0:
            actions.append(Action(TypeAction.FOLD))

        # --- CHECK : possible seulement si rien à payer ---
        if a_payer == 0:
            actions.append(Action(TypeAction.CHECK))

        # --- CALL : possible si on doit payer et qu'on n'est pas à court ---
        if 0 < a_payer < joueur.stack:
            actions.append(Action(TypeAction.CALL, montant=mise_a_suivre))

        # --- RAISES ABSTRAITS : une action par fraction du pot ---
        for fraction in self.tailles_mise:
            # Montant total visé = mise actuelle + fraction * pot
            montant_raise = mise_a_suivre + int(pot * fraction)

            # Vérifier que ce raise est supérieur au minimum légal
            if montant_raise < mise_a_suivre + mise_min_raise:
                montant_raise = mise_a_suivre + mise_min_raise

            # Vérifier que le joueur a assez de jetons pour ce raise
            montant_total_joueur = joueur.mise_tour + joueur.stack
            if montant_raise < montant_total_joueur:
                actions.append(Action(TypeAction.RAISE, montant=montant_raise))

        # --- ALL-IN : toujours possible si le joueur a des jetons ---
        if ALL_IN and joueur.stack > 0:
            montant_allin = joueur.mise_tour + joueur.stack
            # Éviter le doublon si l'all-in est identique à un raise déjà ajouté
            deja_present = any(
                a.type == TypeAction.RAISE and a.montant == montant_allin
                for a in actions
            )
            if not deja_present:
                actions.append(Action(TypeAction.ALL_IN, montant=montant_allin))

        # --- DÉDOUBLONNAGE : supprimer les raises avec montant identique ---
        vus = set()
        actions_uniques = []
        for a in actions:
            cle = (a.type, a.montant)
            if cle not in vus:
                vus.add(cle)
                actions_uniques.append(a)

        return actions_uniques

    def nb_actions_max(self) -> int:
        """
        Retourne le nombre maximum d'actions abstraites possibles.
        Utile pour dimensionner les vecteurs dans MCCFR et Deep CFR.

        = FOLD + CHECK + CALL + len(tailles_mise) raises + ALL_IN
        = 1    + 1     + 1    + len(tailles_mise)         + 1
        """
        return 3 + len(self.tailles_mise) + 1

    def index_action(self, action: Action,
                     pot: int = 1, mise_courante: int = 0) -> int:
        """
        Convertit une Action en index entier (pour les tableaux MCCFR).

        Index fixes :
          0 → FOLD
          1 → CHECK
          2 → CALL
          3..3+len(tailles_mise)-1 → RAISE (par ordre croissant de fraction)
          3+len(tailles_mise) → ALL_IN

        pot, mise_courante : nécessaires pour identifier la taille du RAISE.
            Sans eux, on ne peut pas calculer la fraction raise/pot.
            Valeurs par défaut (pot=1, mise_courante=0) → fraction = montant,
            cohérent avec des montants déjà exprimés en fraction du pot.

        Retourne -1 si l'action n'est pas reconnue.
        """
        if action.type == TypeAction.FOLD:
            return 0
        if action.type == TypeAction.CHECK:
            return 1
        if action.type == TypeAction.CALL:
            return 2
        if action.type == TypeAction.RAISE:
            fraction = (action.montant - mise_courante) / max(pot, 1)
            idx_meilleur = 0
            dist_min = float('inf')
            for k, frac in enumerate(self.tailles_mise):
                dist = abs(frac - fraction)
                if dist < dist_min:
                    dist_min = dist
                    idx_meilleur = k
            return 3 + idx_meilleur
        if action.type == TypeAction.ALL_IN:
            return 3 + len(self.tailles_mise)
        return -1

    def __repr__(self):
        fractions = [f"{int(f*100)}%" for f in self.tailles_mise]
        return f"AbstractionAction(tailles={fractions}, all_in={ALL_IN})"


# -----------------------------------------------------------------------------
# INSTANCE GLOBALE (singleton réutilisé partout dans MCCFR)
# -----------------------------------------------------------------------------

abstraction_action = AbstractionAction()