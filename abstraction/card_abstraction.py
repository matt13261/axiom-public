# =============================================================================
# AXIOM — abstraction/card_abstraction.py
# Abstraction des cartes par buckets d'équité.
#
# Pourquoi ? Il existe des millions de combinaisons de mains possibles en
# Texas Hold'em. MCCFR ne peut pas traiter chaque combinaison séparément —
# le temps de calcul serait astronomique.
#
# Solution : on regroupe les mains en N "buckets" selon leur équité estimée.
# Deux mains avec une équité similaire sont traitées comme identiques par MCCFR.
# Ex avec 8 buckets : bucket 0 = mains très faibles (0-12%), bucket 7 = très fortes (88-100%)
#
# Phase 2 : implémentation par découpage uniforme de l'équité (sans sklearn).
# Phase 4 : on pourra affiner avec un vrai k-means (sklearn) si besoin.
#
# Références :
# - Preflop : 169 mains distinctes réduites à NB_BUCKETS_PREFLOP buckets
# - Postflop (flop/turn/river) : équité calculée par Monte Carlo → bucket
# =============================================================================

import bisect
import random

from engine.hand_evaluator import calculer_equite
from engine.card import VALEURS, COULEURS
from treys import Card


# -----------------------------------------------------------------------------
# PARAMÈTRES
# -----------------------------------------------------------------------------

NB_BUCKETS_PREFLOP  = 8    # nombre de buckets au preflop
NB_BUCKETS_POSTFLOP = 8    # nombre de buckets au flop/turn/river
NB_SIMULATIONS      = 100  # simulations Monte Carlo par calcul d'équité
                            # (100 = compromis vitesse/précision, cohérent avec
                            #  toutes les instances : agent, mccfr, solver, infoset)
NB_ADVERSAIRES      = 2    # on joue à 3 joueurs → 2 adversaires

# -----------------------------------------------------------------------------
# SEUILS DE DÉCOUPAGE NON-UNIFORMES (8 buckets → 7 frontières)
#
# Calibrés pour 3-max (vs 2 adversaires aléatoires) via le script
# recalibrer_3max.py. Les seuils sont les percentiles [12.5, 25, 37.5, 50,
# 62.5, 75, 87.5] de la distribution réelle → buckets équilibrés.
#
# PREFLOP — 169 mains, équités calculées vs 2 adversaires.
#   Plage : [0.21, 0.74] — bien plus tassée qu'en HU.
#
# POSTFLOP — distribution 3-way échantillonnée (flop/turn/river mélangés).
#   moyenne ≈ 0.35, médiane ≈ 0.28 (vs ~0.50 en HU).
#
# ⚠️  Modifier ces seuils invalide tout blueprint existant → ré-entraîner.
# -----------------------------------------------------------------------------

SEUILS_BUCKETS_PREFLOP  = [0.273, 0.304, 0.325, 0.351, 0.379, 0.411, 0.453]
#   bucket 0 : équité < 0.273 (trash : 72o, 32o, 82o…)
#   bucket 1 : 0.273 – 0.304  (très faible offsuit bas)
#   bucket 2 : 0.304 – 0.325  (faible offsuit / petits suited)
#   bucket 3 : 0.325 – 0.351  (marginal : petites paires, Qx bas suited)
#   bucket 4 : 0.351 – 0.379  (autour moyenne : broadway offsuit, Ax bas)
#   bucket 5 : 0.379 – 0.411  (au-dessus : suited connectors, Ax mid)
#   bucket 6 : 0.411 – 0.453  (fort : KQ, Ax haut, paires moyennes)
#   bucket 7 : équité ≥ 0.453 (premium : AA-JJ, AK, AQs)

SEUILS_BUCKETS_POSTFLOP = [0.086, 0.154, 0.213, 0.284, 0.378, 0.508, 0.698]
#   bucket 0 : équité < 0.086 (mort : overcards sans rien)
#   bucket 1 : 0.086 – 0.154  (très faible : runners, backdoor)
#   bucket 2 : 0.154 – 0.213  (faible : bottom pair, gutshot)
#   bucket 3 : 0.213 – 0.284  (marginal : middle pair, OESD)
#   bucket 4 : 0.284 – 0.378  (autour moyenne : top pair faible, draws)
#   bucket 5 : 0.378 – 0.508  (solide : top pair bon kicker, two pair)
#   bucket 6 : 0.508 – 0.698  (fort : overpair, set, straight, flush)
#   bucket 7 : équité ≥ 0.698 (nuts / quasi-nuts contre 2 adversaires)


# -----------------------------------------------------------------------------
# TABLE DE LOOKUP PREFLOP (précalculée)
# Associe chaque main preflop à un bucket 0..NB_BUCKETS_PREFLOP-1
# On stocke sous forme de dict {(carte1_treys, carte2_treys) -> bucket}
# mais comme les cartes changent à chaque partie, on utilise une clé abstraite.
# -----------------------------------------------------------------------------

def _cle_preflop_abstraite(cartes: list) -> str:
    """
    Génère une clé abstraite pour une main preflop (indépendante de la couleur).

    En poker, AKs (suited) et AKo (offsuit) sont différentes,
    mais AsKs et AhKh sont identiques (même structure).

    Retourne une string du type :
      "AKs" (suited — même couleur)
      "AKo" (offsuit — couleurs différentes)
      "AA"  (paire)

    Les valeurs sont toujours triées de la plus haute à la plus basse.
    """
    if len(cartes) != 2:
        return "??"

    # Extraire rang et couleur via treys
    rang1  = Card.get_rank_int(cartes[0])   # 0=2 ... 12=As
    rang2  = Card.get_rank_int(cartes[1])
    suit1  = Card.get_suit_int(cartes[0])   # 1=spades, 2=hearts, 4=diamonds, 8=clubs
    suit2  = Card.get_suit_int(cartes[1])

    # Trier du plus grand au plus petit
    if rang1 < rang2:
        rang1, rang2 = rang2, rang1
        suit1, suit2 = suit2, suit1

    # Noms des rangs (treys : 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A)
    noms_rangs = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    r1 = noms_rangs[rang1]
    r2 = noms_rangs[rang2]

    if rang1 == rang2:
        return f"{r1}{r2}"          # paire : "AA", "KK", etc.
    elif suit1 == suit2:
        return f"{r1}{r2}s"         # suited : "AKs", "T9s", etc.
    else:
        return f"{r1}{r2}o"         # offsuit : "AKo", "T9o", etc.


# Table de ranking preflop des 169 mains distinctes
# Source : consensus de la théorie GTO du poker
# Rank 1 = meilleure main (AA), rank 169 = pire main (72o)
# On ne liste pas les 169 ici — on utilise une équité approximative connue.

_EQUITE_PREFLOP_APPROX = {
    # Table complète des 169 mains distinctes, équités vs 2 adversaires
    # aléatoires (3-max). Calculée par MC 3000 simulations via
    # recalibrer_3max.py (seed=42).
    # ── Paires ──
    "AA": 0.74, "KK": 0.70, "QQ": 0.65, "JJ": 0.62, "TT": 0.59,
    "99": 0.54, "88": 0.50, "77": 0.47, "66": 0.43, "55": 0.40,
    "44": 0.37, "33": 0.34, "22": 0.31,
    # ── Suited aces ──
    "AKs": 0.50, "AQs": 0.49, "AJs": 0.48, "ATs": 0.50,
    "A9s": 0.45, "A8s": 0.46, "A7s": 0.44, "A6s": 0.44,
    "A5s": 0.42, "A4s": 0.41, "A3s": 0.41, "A2s": 0.40,
    # ── Suited kings ──
    "KQs": 0.48, "KJs": 0.46, "KTs": 0.47, "K9s": 0.45,
    "K8s": 0.41, "K7s": 0.41, "K6s": 0.40, "K5s": 0.39,
    "K4s": 0.40, "K3s": 0.37, "K2s": 0.37,
    # ── Suited queens ──
    "QJs": 0.44, "QTs": 0.45, "Q9s": 0.43, "Q8s": 0.41,
    "Q7s": 0.38, "Q6s": 0.38, "Q5s": 0.37, "Q4s": 0.37,
    "Q3s": 0.35, "Q2s": 0.34,
    # ── Suited jacks ──
    "JTs": 0.43, "J9s": 0.40, "J8s": 0.41, "J7s": 0.38,
    "J6s": 0.35, "J5s": 0.36, "J4s": 0.33, "J3s": 0.33, "J2s": 0.33,
    # ── Suited tens + lower ──
    "T9s": 0.41, "T8s": 0.39, "T7s": 0.36, "T6s": 0.33,
    "T5s": 0.31, "T4s": 0.31, "T3s": 0.31, "T2s": 0.31,
    "98s": 0.37, "97s": 0.37, "96s": 0.34, "95s": 0.31,
    "94s": 0.31, "93s": 0.31, "92s": 0.28,
    "87s": 0.34, "86s": 0.33, "85s": 0.32, "84s": 0.31,
    "83s": 0.28, "82s": 0.29,
    "76s": 0.36, "75s": 0.33, "74s": 0.30, "73s": 0.28, "72s": 0.26,
    "65s": 0.33, "64s": 0.31, "63s": 0.29, "62s": 0.27,
    "54s": 0.32, "53s": 0.28, "52s": 0.27,
    "43s": 0.29, "42s": 0.26, "32s": 0.27,
    # ── Offsuit aces ──
    "AKo": 0.50, "AQo": 0.48, "AJo": 0.48, "ATo": 0.46,
    "A9o": 0.43, "A8o": 0.43, "A7o": 0.41, "A6o": 0.40,
    "A5o": 0.41, "A4o": 0.40, "A3o": 0.38, "A2o": 0.37,
    # ── Offsuit kings ──
    "KQo": 0.45, "KJo": 0.44, "KTo": 0.43, "K9o": 0.40,
    "K8o": 0.40, "K7o": 0.38, "K6o": 0.37, "K5o": 0.37,
    "K4o": 0.34, "K3o": 0.36, "K2o": 0.33,
    # ── Offsuit queens ──
    "QJo": 0.42, "QTo": 0.41, "Q9o": 0.39, "Q8o": 0.36,
    "Q7o": 0.35, "Q6o": 0.35, "Q5o": 0.34, "Q4o": 0.32,
    "Q3o": 0.31, "Q2o": 0.30,
    # ── Offsuit jacks ──
    "JTo": 0.41, "J9o": 0.38, "J8o": 0.38, "J7o": 0.34,
    "J6o": 0.32, "J5o": 0.31, "J4o": 0.31, "J3o": 0.30, "J2o": 0.29,
    # ── Offsuit tens + lower ──
    "T9o": 0.37, "T8o": 0.34, "T7o": 0.35, "T6o": 0.31,
    "T5o": 0.29, "T4o": 0.29, "T3o": 0.28, "T2o": 0.26,
    "98o": 0.34, "97o": 0.31, "96o": 0.33, "95o": 0.27,
    "94o": 0.28, "93o": 0.26, "92o": 0.25,
    "87o": 0.33, "86o": 0.31, "85o": 0.29, "84o": 0.26,
    "83o": 0.25, "82o": 0.24,
    "76o": 0.30, "75o": 0.27, "74o": 0.27, "73o": 0.24, "72o": 0.21,
    "65o": 0.28, "64o": 0.27, "63o": 0.25, "62o": 0.23,
    "54o": 0.29, "53o": 0.25, "52o": 0.23,
    "43o": 0.25, "42o": 0.23, "32o": 0.22,
}

# Valeur de secours si une clé n'est pas dans la table (ne devrait pas arriver
# puisque les 169 mains sont listées). 0.22 = équité moyenne d'une main trash
# 3-way (32o, 82o).
_EQUITE_PREFLOP_DEFAUT = 0.22


# -----------------------------------------------------------------------------
# CONSTANTES HEADS-UP (mode 2 joueurs, endgame de tournoi)
#
# Calibrées via recalibrer_hu.py (nb_adversaires=1).
# Utilisées par train_hu.py et par l'agent en mode HU.
# -----------------------------------------------------------------------------

SEUILS_BUCKETS_PREFLOP_HU  = [0.42, 0.459, 0.493, 0.526, 0.556, 0.588, 0.631]
SEUILS_BUCKETS_POSTFLOP_HU = [0.24, 0.334, 0.414, 0.504, 0.592, 0.708, 0.836]

_EQUITE_PREFLOP_APPROX_HU = {
    # Paires
    "AA": 0.84, "KK": 0.83, "QQ": 0.81, "JJ": 0.77, "TT": 0.75,
    "99": 0.72, "88": 0.68, "77": 0.67, "66": 0.64, "55": 0.61,
    "44": 0.57, "33": 0.54, "22": 0.51,
    # Suited aces
    "AKs": 0.68, "AQs": 0.67, "AJs": 0.68, "ATs": 0.65,
    "A9s": 0.65, "A8s": 0.62, "A7s": 0.64, "A6s": 0.62,
    "A5s": 0.63, "A4s": 0.59, "A3s": 0.61, "A2s": 0.58,
    # Suited kings
    "KQs": 0.64, "KJs": 0.64, "KTs": 0.62, "K9s": 0.60,
    "K8s": 0.60, "K7s": 0.57, "K6s": 0.60, "K5s": 0.58,
    "K4s": 0.59, "K3s": 0.57, "K2s": 0.56,
    # Suited queens
    "QJs": 0.61, "QTs": 0.61, "Q9s": 0.59, "Q8s": 0.57,
    "Q7s": 0.56, "Q6s": 0.55, "Q5s": 0.54, "Q4s": 0.53,
    "Q3s": 0.53, "Q2s": 0.54,
    # Suited jacks
    "JTs": 0.58, "J9s": 0.57, "J8s": 0.54, "J7s": 0.54,
    "J6s": 0.51, "J5s": 0.52, "J4s": 0.52, "J3s": 0.49, "J2s": 0.49,
    # Suited tens + lower
    "T9s": 0.56, "T8s": 0.55, "T7s": 0.53, "T6s": 0.51,
    "T5s": 0.50, "T4s": 0.50, "T3s": 0.48, "T2s": 0.49,
    "98s": 0.54, "97s": 0.51, "96s": 0.49, "95s": 0.48,
    "94s": 0.47, "93s": 0.46, "92s": 0.45,
    "87s": 0.52, "86s": 0.49, "85s": 0.46, "84s": 0.45,
    "83s": 0.44, "82s": 0.43,
    "76s": 0.48, "75s": 0.48, "74s": 0.45, "73s": 0.45, "72s": 0.41,
    "65s": 0.46, "64s": 0.44, "63s": 0.44, "62s": 0.39,
    "54s": 0.44, "53s": 0.42, "52s": 0.40,
    "43s": 0.42, "42s": 0.40, "32s": 0.39,
    # Offsuit aces
    "AKo": 0.66, "AQo": 0.65, "AJo": 0.66, "ATo": 0.64,
    "A9o": 0.61, "A8o": 0.63, "A7o": 0.60, "A6o": 0.60,
    "A5o": 0.61, "A4o": 0.58, "A3o": 0.58, "A2o": 0.56,
    # Offsuit kings
    "KQo": 0.62, "KJo": 0.61, "KTo": 0.61, "K9o": 0.58,
    "K8o": 0.57, "K7o": 0.57, "K6o": 0.55, "K5o": 0.56,
    "K4o": 0.55, "K3o": 0.55, "K2o": 0.54,
    # Offsuit queens
    "QJo": 0.61, "QTo": 0.58, "Q9o": 0.56, "Q8o": 0.55,
    "Q7o": 0.53, "Q6o": 0.52, "Q5o": 0.53, "Q4o": 0.50,
    "Q3o": 0.51, "Q2o": 0.50,
    # Offsuit jacks
    "JTo": 0.56, "J9o": 0.54, "J8o": 0.53, "J7o": 0.52,
    "J6o": 0.51, "J5o": 0.49, "J4o": 0.49, "J3o": 0.47, "J2o": 0.47,
    # Offsuit tens + lower
    "T9o": 0.52, "T8o": 0.53, "T7o": 0.51, "T6o": 0.49,
    "T5o": 0.47, "T4o": 0.45, "T3o": 0.46, "T2o": 0.45,
    "98o": 0.51, "97o": 0.48, "96o": 0.47, "95o": 0.46,
    "94o": 0.43, "93o": 0.39, "92o": 0.42,
    "87o": 0.47, "86o": 0.46, "85o": 0.43, "84o": 0.42,
    "83o": 0.39, "82o": 0.41,
    "76o": 0.46, "75o": 0.43, "74o": 0.42, "73o": 0.39, "72o": 0.38,
    "65o": 0.44, "64o": 0.41, "63o": 0.39, "62o": 0.37,
    "54o": 0.40, "53o": 0.38, "52o": 0.37,
    "43o": 0.39, "42o": 0.36, "32o": 0.35,
}

# Équité de secours HU (≈ 0.35 = pire main 32o).
_EQUITE_PREFLOP_DEFAUT_HU = 0.35


# -----------------------------------------------------------------------------
# CLASSE PRINCIPALE
# -----------------------------------------------------------------------------

class AbstractionCartes:
    """
    Regroupe les mains de poker en buckets d'équité pour MCCFR.

    Utilisation :
        abstraction = AbstractionCartes()
        bucket = abstraction.bucket_preflop(cartes_joueur)
        bucket = abstraction.bucket_postflop(cartes_joueur, board)
    """

    def __init__(self,
                 nb_buckets_preflop:  int  = NB_BUCKETS_PREFLOP,
                 nb_buckets_postflop: int  = NB_BUCKETS_POSTFLOP,
                 nb_simulations:      int  = NB_SIMULATIONS,
                 seuils_preflop:      list = None,
                 seuils_postflop:     list = None,
                 nb_adversaires:      int  = None,
                 mode:                str  = '3max'):
        """
        mode : '3max' (par défaut, 2 adversaires) ou 'hu' (1 adversaire).
               Sélectionne automatiquement les seuils, la table preflop et le
               nombre d'adversaires adaptés. Les paramètres explicites
               (seuils_*, nb_adversaires) surchargent le mode.
        """
        if mode not in ('3max', 'hu'):
            raise ValueError(f"mode doit être '3max' ou 'hu', reçu {mode!r}")

        self.mode                = mode
        self.nb_buckets_preflop  = nb_buckets_preflop
        self.nb_buckets_postflop = nb_buckets_postflop
        self.nb_simulations      = nb_simulations

        # Sélection des constantes selon le mode
        if mode == 'hu':
            defaut_seuils_pre  = SEUILS_BUCKETS_PREFLOP_HU
            defaut_seuils_post = SEUILS_BUCKETS_POSTFLOP_HU
            self.table_preflop = _EQUITE_PREFLOP_APPROX_HU
            self.equite_defaut = _EQUITE_PREFLOP_DEFAUT_HU
            defaut_nb_adv      = 1
        else:  # '3max'
            defaut_seuils_pre  = SEUILS_BUCKETS_PREFLOP
            defaut_seuils_post = SEUILS_BUCKETS_POSTFLOP
            self.table_preflop = _EQUITE_PREFLOP_APPROX
            self.equite_defaut = _EQUITE_PREFLOP_DEFAUT
            defaut_nb_adv      = NB_ADVERSAIRES

        # Surcharges explicites (None → valeur par défaut du mode)
        self.nb_adversaires  = nb_adversaires if nb_adversaires is not None \
                               else defaut_nb_adv
        self.seuils_preflop  = seuils_preflop  if seuils_preflop  is not None \
                               else defaut_seuils_pre
        self.seuils_postflop = seuils_postflop if seuils_postflop is not None \
                               else defaut_seuils_post

    # ------------------------------------------------------------------
    # BUCKET PREFLOP
    # ------------------------------------------------------------------

    def bucket_preflop(self, cartes: list) -> int:
        """
        Retourne le bucket preflop d'une main (0 = très faible, max = très forte).

        cartes : liste de 2 cartes au format treys (int)
        """
        cle    = _cle_preflop_abstraite(cartes)
        equite = self.table_preflop.get(cle, self.equite_defaut)
        return self._equite_vers_bucket(equite, self.seuils_preflop)

    # ------------------------------------------------------------------
    # BUCKET POSTFLOP (flop, turn, river)
    # ------------------------------------------------------------------

    def bucket_postflop(self, cartes: list, board: list) -> int:
        """
        Retourne le bucket postflop d'une main en fonction du board visible.
        Utilise une simulation Monte Carlo pour estimer l'équité.

        cartes : liste de 2 cartes au format treys
        board  : liste de 3, 4 ou 5 cartes communes au format treys
        """
        random.seed(42)
        equite = calculer_equite(
            cartes_joueur      = cartes,
            cartes_adversaires = [],          # adversaires inconnus
            board              = board,
            nb_simulations     = self.nb_simulations,
            nb_adversaires     = self.nb_adversaires,
        )
        return self._equite_vers_bucket(equite, self.seuils_postflop)

    # ------------------------------------------------------------------
    # MÉTHODE UNIFIÉE : choisit automatiquement preflop ou postflop
    # ------------------------------------------------------------------

    def bucket(self, cartes: list, board: list) -> int:
        """
        Retourne le bucket approprié selon la phase de jeu.

        cartes : 2 cartes du joueur (format treys)
        board  : 0 carte = preflop, 3-5 cartes = postflop
        """
        if len(board) == 0:
            return self.bucket_preflop(cartes)
        else:
            return self.bucket_postflop(cartes, board)

    def bucket_et_equite(self, cartes: list, board: list) -> tuple:
        """
        Retourne (bucket, equite_brute) en une seule passe Monte Carlo.

        Évite de calculer deux fois l'équité (une pour le bucket, une pour
        le vecteur d'encodage réseau). Le calcul MC est fait une seule fois.

        cartes : 2 cartes du joueur (format treys)
        board  : 0 carte = preflop, 3-5 cartes = postflop

        Retourne
        --------
        (bucket: int, equite: float 0-1)
        """
        if len(board) == 0:
            cle    = _cle_preflop_abstraite(cartes)
            equite = self.table_preflop.get(cle, self.equite_defaut)
            bucket = self._equite_vers_bucket(equite, self.seuils_preflop)
            return bucket, equite
        else:
            random.seed(42)
            equite = calculer_equite(
                cartes_joueur      = cartes,
                cartes_adversaires = [],
                board              = board,
                nb_simulations     = self.nb_simulations,
                nb_adversaires     = self.nb_adversaires,
            )
            bucket = self._equite_vers_bucket(equite, self.seuils_postflop)
            return bucket, equite

    # ------------------------------------------------------------------
    # UTILITAIRE INTERNE
    # ------------------------------------------------------------------

    def _equite_vers_bucket(self, equite: float, seuils: list) -> int:
        """
        Convertit une équité (float 0.0-1.0) en numéro de bucket (int).

        Découpage non-uniforme via bisect sur une liste de seuils triés.
        Produit len(seuils)+1 buckets, numérotés 0..len(seuils).

        Exemple avec seuils=[0.35, 0.45, 0.52, 0.58, 0.64, 0.70, 0.78] :
          équité < 0.35  → bucket 0
          0.35 ≤ eq < 0.45 → bucket 1
          ...
          équité ≥ 0.78  → bucket 7
        """
        return bisect.bisect_right(seuils, equite)

    def __repr__(self):
        return (f"AbstractionCartes("
                f"mode={self.mode}, "
                f"preflop={self.nb_buckets_preflop} buckets, "
                f"postflop={self.nb_buckets_postflop} buckets, "
                f"nb_adv={self.nb_adversaires}, "
                f"simulations={self.nb_simulations})")


# -----------------------------------------------------------------------------
# INSTANCES GLOBALES
#   - abstraction_cartes     : mode 3-max (par défaut partout dans MCCFR/agent)
#   - abstraction_cartes_hu  : mode heads-up (endgame tournoi + train_hu.py)
# -----------------------------------------------------------------------------

abstraction_cartes    = AbstractionCartes(nb_simulations=100, mode='3max')
abstraction_cartes_hu = AbstractionCartes(nb_simulations=100, mode='hu')


# -----------------------------------------------------------------------------
# PHASE 2 — AbstractionCartesV2
# Implémentation progressive — Étape D de la migration P6.
# Ref : docs/investigations/P6-abstraction/spec.md section 2.1
# -----------------------------------------------------------------------------

class AbstractionCartesV2:
    """
    Abstraction cartes Phase 2 — 50 buckets postflop par street via K-means 3D.

    Features : (E[HS], E[HS²], Potentiel) calculées par MC avec treys.
    Centroïdes entraînés hors ligne et stockés dans centroides_v2.npz.
    V1 (préflop 8 buckets) préservée sans modification.
    """
    NB_BUCKETS_FLOP         = 50
    NB_BUCKETS_TURN         = 50
    NB_BUCKETS_RIVER        = 50
    NB_BUCKETS_PREFLOP      = 8
    NB_MC_SIMULATIONS       = 100
    DEFAULT_CENTROIDES_PATH = 'data/abstraction/centroides_v2.npz'

    def __init__(self, centroides_path=None, centroides=None):
        """
        Paramètres
        ----------
        centroides_path : str|None  — chemin vers un .npz
        centroides      : dict|None — {'flop': np.ndarray, 'turn':..., 'river':...}
                           passé directement (utile pour tests / calibration)
        """
        self.centroides = None
        self._cache     = {}           # LRU dict — clé : (tuple(cartes), tuple(board), street)

        if centroides_path is not None:
            import os, numpy as np
            if not os.path.exists(centroides_path):
                raise FileNotFoundError(
                    f"AbstractionCartesV2 : centroïdes introuvables : {centroides_path}"
                )
            data = np.load(centroides_path)
            self.centroides = {k: data[k] for k in data.files}
        elif centroides is not None:
            self.centroides = centroides
        else:
            import logging, os, numpy as np
            _default = self.DEFAULT_CENTROIDES_PATH
            if os.path.exists(_default):
                data = np.load(_default)
                self.centroides = {k: data[k] for k in data.files}
                logging.getLogger(__name__).info(
                    f"AbstractionCartesV2 : centroïdes auto-chargés depuis "
                    f"{_default} (pour personnaliser : centroides_path=...)"
                )
            else:
                logging.getLogger(__name__).warning(
                    f"AbstractionCartesV2 : aucun centroïde chargé — "
                    f"bucket_postflop() lèvera RuntimeError. "
                    f"Path par défaut testé : {_default}"
                )

    def bucket_preflop(self, cartes):
        """Délègue à V1 — le préflop n'est pas refondu en Phase 2."""
        return AbstractionCartes().bucket_preflop(cartes)

    def bucket_postflop(self, cartes, board, street='flop'):
        """
        Retourne le bucket postflop (0–49) via K-means.
        Résultats mis en cache par (cartes, board, street).

        Raises
        ------
        RuntimeError si les centroïdes ne sont pas chargés.
        """
        if self.centroides is None:
            raise RuntimeError(
                "AbstractionCartesV2 : centroïdes non chargés. "
                "Fournir centroides= ou centroides_path= à l'instanciation, "
                "ou lancer recalibrer_3max_v2.py pour générer centroides_v2.npz."
            )

        key = (tuple(cartes), tuple(board), street)
        if key in self._cache:
            return self._cache[key]

        from abstraction.card_clustering import compute_features, predict_bucket
        features = compute_features(
            cartes, board, street=street,
            n_sim=self.NB_MC_SIMULATIONS, seed=42
        )
        result = predict_bucket(features, self.centroides[street])
        self._cache[key] = result
        return result

    def bucket(self, cartes, board):
        """Dispatch préflop/postflop — interface compatible avec V1."""
        if len(board) == 0:
            return self.bucket_preflop(cartes)
        street = {3: 'flop', 4: 'turn', 5: 'river'}.get(len(board), 'flop')
        return self.bucket_postflop(cartes, board, street=street)

    def bucket_et_equite(self, cartes, board):
        """Retourne (bucket, equite) — equite = E[HS] MC, compatible interface V1."""
        if len(board) == 0:
            return AbstractionCartes().bucket_et_equite(cartes, board)
        if self.centroides is None:
            raise RuntimeError(
                "AbstractionCartesV2 : centroïdes non chargés. "
                "Fournir centroides= ou centroides_path= à l'instanciation."
            )
        street = {3: 'flop', 4: 'turn', 5: 'river'}.get(len(board), 'flop')
        from abstraction.card_clustering import compute_features, predict_bucket
        features = compute_features(cartes, board, street=street,
                                    n_sim=self.NB_MC_SIMULATIONS, seed=42)
        e_hs = features[0]
        key = (tuple(cartes), tuple(board), street)
        if key in self._cache:
            bucket = self._cache[key]
        else:
            bucket = predict_bucket(features, self.centroides[street])
            self._cache[key] = bucket
        return bucket, float(e_hs)