# =============================================================================
# AXIOM — abstraction/info_set.py
# Information Set — ce que l'IA sait à un instant T.
#
# La clé est construite dans le MÊME format que mccfr.py (_cle_infoset),
# ce qui garantit que le blueprint est bien consulté pendant le jeu.
#
# Format :
#   PHASE|pos=X|bucket=Y|pot=Z|stacks=(a,b,c)|hist=ABC
#
# Où hist = actions de la PHASE COURANTE UNIQUEMENT (pas les blindes,
# pas les actions des phases précédentes) — compatible avec mccfr.py.
#
# ─────────────────────────────────────────────────────────────────────────────
# PALIERS DE NORMALISATION (mis à jour Phase 10)
# ─────────────────────────────────────────────────────────────────────────────
#
# Les paliers contrôlent la granularité de la clé d'infoset.
# Des paliers trop larges fusionnent des situations trop différentes
# (ex : pot=21BB et pot=49BB dans le même palier → même décision).
#
# Anciens paliers (trop grossiers) :
#   POT   : [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]
#   STACK : [0, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
#
# Nouveaux paliers (plus fins) :
#   POT   : résolution doublée entre 2 et 20 BB (zone critique du jeu)
#   STACK : résolution améliorée entre 10 et 50 BB (stacks courts de tournoi)
#
# Impact : +30 à +50% d'infosets distincts → meilleure couverture postflop.
# Contrepartie : le blueprint doit être ré-entraîné pour en bénéficier.
# =============================================================================

from engine.game_state import EtatJeu, Phase, _PHASE_IDX
from engine.player import Joueur
from abstraction.card_abstraction import abstraction_cartes


# ─────────────────────────────────────────────────────────────────────────────
# DISCRÉTISATION DE LA FRACTION DE RAISE
# ─────────────────────────────────────────────────────────────────────────────

def _abstraire_sizing(idx_taille: int) -> str:
    pass


def _discretiser_raise_frac(frac: float) -> int:
    """
    Convertit la fraction raise/pot en bucket discret (0–4).

      0 → pas de raise (frac == 0)
      1 → micro raise  (0 < frac ≤ 0.33)
      2 → small raise  (0.33 < frac ≤ 0.75)
      3 → pot raise    (0.75 < frac ≤ 1.25)
      4 → over-raise   (frac > 1.25)

    Utilisé à la fois dans la clé d'infoset (info_set.py) et dans
    l'encodage réseau (network.py) pour rester cohérent.
    """
    if frac <= 0.0:
        return 0
    if frac <= 0.33:
        return 1
    if frac <= 0.75:
        return 2
    if frac <= 1.25:
        return 3
    return 4


# ─────────────────────────────────────────────────────────────────────────────
# PSEUDO-HARMONIC BLENDING SUR LES BUCKETS DE RAISE
# ─────────────────────────────────────────────────────────────────────────────
#
# Problème : _discretiser_raise_frac écrase 60-80% d'info quand la fraction
# tombe près d'une frontière (ex : frac=0.34 → bucket 2, frac=0.32 → bucket 1).
# Solution Ganzfried-Sandholm : répartir la "masse" d'observation entre les
# deux buckets voisins via pseudo-harmonic mapping, puis faire un double
# lookup blueprint et blender les stratégies.
#
# Les CENTRES de buckets correspondent au milieu de chaque plage :
#   bucket 1 : (0, 0.33]   → centre 0.165
#   bucket 2 : (0.33, 0.75] → centre 0.54
#   bucket 3 : (0.75, 1.25] → centre 1.0
#   bucket 4 : (> 1.25)     → centre 1.5 (représentatif)
#
# On NE BLEND JAMAIS avec bucket 0 (pas de raise) : bucket 0 est
# catégoriquement distinct (CHECK/CALL vs RAISE).
# ─────────────────────────────────────────────────────────────────────────────

_CENTRES_RAISE_BUCKET = [0.165, 0.54, 1.0, 1.5]   # buckets 1..4
_BUCKETS_CENTRES      = [1, 2, 3, 4]


def buckets_pseudo_harmonic(frac: float) -> list:
    """
    Retourne une distribution [(bucket_idx, poids)] sur les buckets de raise
    via pseudo-harmonic mapping entre centres de buckets voisins.

    frac : fraction de raise observée (mise_courante / pot).

    Retourne :
      - [(0, 1.0)] si pas de raise (frac == 0)
      - [(k, 1.0)] si frac coïncide avec un centre / est hors bornes
      - [(k, p_k), (k+1, p_{k+1})] sinon (les deux voisins pseudo-harmoniques)

    Les poids somment à 1.0. À utiliser pour double-lookup + blending.
    """
    # Import local pour éviter import circulaire
    from abstraction.action_abstraction import pseudo_harmonic_mapping

    if frac <= 0.0:
        return [(0, 1.0)]

    # En-deçà du plus petit centre → tout sur bucket 1
    if frac <= _CENTRES_RAISE_BUCKET[0]:
        return [(_BUCKETS_CENTRES[0], 1.0)]

    # Au-delà du plus grand centre → tout sur bucket 4
    if frac >= _CENTRES_RAISE_BUCKET[-1]:
        return [(_BUCKETS_CENTRES[-1], 1.0)]

    # Encadrement entre deux centres
    for k in range(len(_CENTRES_RAISE_BUCKET) - 1):
        A = _CENTRES_RAISE_BUCKET[k]
        B = _CENTRES_RAISE_BUCKET[k + 1]
        if A <= frac <= B:
            p_A, p_B = pseudo_harmonic_mapping(frac, A, B)
            out = []
            if p_A > 0.0:
                out.append((_BUCKETS_CENTRES[k],     p_A))
            if p_B > 0.0:
                out.append((_BUCKETS_CENTRES[k + 1], p_B))
            return out

    return [(_BUCKETS_CENTRES[-1], 1.0)]


# ─────────────────────────────────────────────────────────────────────────────
# PALIERS DE NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

# Pot normalisé par la grande blinde
# Résolution fine dans la zone 1-20 BB (la plus fréquente en tournoi 500 chips)
# Résolution large au-delà (situations rares)
PALIERS_POT = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100
]

# Stack normalisé par la grande blinde
# Résolution fine entre 5 et 50 BB (tournoi avec stack départ 500 chips / 10BB)
# Points clés : 10BB (push/fold), 20BB (open-shove), 50BB (profond)
PALIERS_STACK = [
    0, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 65, 75, 100, 150, 200
]


def _normaliser(valeur: float, paliers: list) -> int:
    """
    Retourne le plus grand palier inférieur ou égal à la valeur.
    Utilisé pour discrétiser pot et stacks dans la clé d'infoset.
    """
    for i in range(len(paliers) - 1, -1, -1):
        if valeur >= paliers[i]:
            return paliers[i]
    return 0


class InfoSet:
    """
    Information set d'un joueur à un instant T.
    Clé compatible avec mccfr.py — garantit que le blueprint est consulté.
    """

    def __init__(self, etat: EtatJeu, joueur: Joueur):
        self.phase    = etat.phase
        self.position = joueur.position

        # Bucket des cartes
        self.bucket = abstraction_cartes.bucket(joueur.cartes, etat.board)

        # Pot et stacks normalisés par la grande blinde
        gb = max(etat.grande_blinde, 1)
        self.pot_norm   = _normaliser(etat.pot / gb, PALIERS_POT)
        stacks_norm     = []
        for j in etat.joueurs:
            stacks_norm.append(_normaliser(j.stack / gb, PALIERS_STACK))
        self.stacks_norm = tuple(stacks_norm)

        # Historique de la PHASE COURANTE uniquement (compatible mccfr.py)
        # On utilise etat.historique_phases qui est maintenu par enregistrer_action()
        phase_idx       = _PHASE_IDX.get(etat.phase, 0)
        self.historique = etat.historique_phases[phase_idx]

        # Fraction de raise discrétisée — mise_courante / pot
        # Capture la taille de la mise face à laquelle on doit agir.
        self.raise_bucket = _discretiser_raise_frac(
            etat.mise_courante / max(etat.pot, 1)
        )

        self.cle = self._construire_cle()

    def _construire_cle(self) -> str:
        stacks_str = ','.join(str(s) for s in self.stacks_norm)
        return (
            f"{self.phase.name}"
            f"|pos={self.position}"
            f"|bucket={self.bucket}"
            f"|pot={self.pot_norm}"
            f"|stacks=({stacks_str})"
            f"|hist={self.historique}"
            f"|raise={self.raise_bucket}"
        )

    def __hash__(self)        : return hash(self.cle)
    def __eq__(self, autre)   :
        if isinstance(autre, InfoSet): return self.cle == autre.cle
        if isinstance(autre, str)    : return self.cle == autre
        return False
    def __repr__(self)        : return f"InfoSet({self.cle})"


def construire_cle_infoset(etat: EtatJeu, joueur: Joueur) -> str:
    """
    Raccourci : retourne la clé string de l'infoset.
    Compatible avec les clés générées par mccfr.py pendant l'entraînement.
    """
    return InfoSet(etat, joueur).cle
