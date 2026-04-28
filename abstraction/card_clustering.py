"""
card_clustering.py — Abstraction cartes Phase 2 (V2).

Fournit les outils pour :
- compute_features : calcul Monte Carlo de (E[HS], E[HS²], Potentiel)
- fit_centroids    : K-means sklearn sur les features 3D
- predict_bucket   : argmin distance L2 vers les centroïdes
"""

import random as _random
import numpy as _np
from treys import Card as _Card, Evaluator as _Evaluator

# Deck complet : 52 cartes treys pré-calculées
_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
_SUITS = ['s', 'h', 'd', 'c']
_FULL_DECK = [_Card.new(r + s) for r in _RANKS for s in _SUITS]

_EVALUATOR = _Evaluator()


def _hs_score(score_hero, score_opp):
    """Convertit deux scores treys (lower=better) en {0.0, 0.5, 1.0}."""
    if score_hero < score_opp:
        return 1.0
    elif score_hero == score_opp:
        return 0.5
    return 0.0


def compute_features(cartes_hero, board, street, n_sim=100, seed=None):
    """
    Calcule (E[HS], E[HS²], Potentiel) par Monte Carlo.

    Paramètres
    ----------
    cartes_hero : list[int]   — 2 cartes treys du héros
    board       : list[int]   — 3 (flop), 4 (turn) ou 5 (river) cartes
    street      : str         — 'flop' | 'turn' | 'river'
    n_sim       : int         — nombre de simulations MC
    seed        : int|None    — graine pour la reproductibilité

    Retourne
    --------
    (float, float, float) — E[HS], E[HS²], Potentiel
    """
    rng     = _random.Random(seed)
    used    = set(cartes_hero) | set(board)
    pool    = [c for c in _FULL_DECK if c not in used]
    n_extra = 5 - len(board)

    arr_c = _np.empty(n_sim)
    arr_r = _np.empty(n_sim)

    for i in range(n_sim):
        rng.shuffle(pool)
        opp         = pool[:2]
        river_board = board + pool[2:2 + n_extra]

        sc_h = _EVALUATOR.evaluate(board, cartes_hero)
        sc_o = _EVALUATOR.evaluate(board, opp)
        arr_c[i] = _hs_score(sc_h, sc_o)

        if street != 'river':
            sr_h = _EVALUATOR.evaluate(river_board, cartes_hero)
            sr_o = _EVALUATOR.evaluate(river_board, opp)
            arr_r[i] = _hs_score(sr_h, sr_o)

    e_hs  = float(arr_c.mean())
    e_hs2 = float(_np.square(arr_c).mean())

    if street == 'river':
        potentiel = 0.0
    else:
        # Potentiel peut être négatif : overpair perd de la valeur (→ clustering distinct)
        potentiel = float(float(arr_r.mean()) - e_hs)

    return (e_hs, e_hs2, potentiel)


def fit_centroids(features, n_clusters=50, seed=42):
    """
    Lance K-means sur un tableau de features (N, 3) et retourne les centroïdes.

    Paramètres
    ----------
    features   : np.ndarray shape (N, 3) — E[HS], E[HS²], Potentiel
    n_clusters : int — nombre de buckets (défaut 50)
    seed       : int — graine sklearn pour déterminisme

    Retourne
    --------
    np.ndarray shape (n_clusters, 3), dtype float32
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(features)
    return km.cluster_centers_.astype(_np.float32)


def predict_bucket(features_vec, centroids):
    """
    Retourne l'index du centroïde le plus proche (distance L2 euclidienne).

    Paramètres
    ----------
    features_vec : array-like shape (3,)
    centroids    : np.ndarray shape (n_clusters, 3)

    Retourne
    --------
    int — index dans [0, n_clusters - 1]
    """
    vec  = _np.asarray(features_vec, dtype=_np.float32)
    dist = _np.linalg.norm(centroids - vec, axis=1)
    return int(_np.argmin(dist))
