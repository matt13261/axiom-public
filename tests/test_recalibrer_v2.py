"""
Tests TDD pour recalibrer_3max_v2.py — Étape E calibration centroïdes.
"""

import numpy as np
import pytest


# =============================================================================
# TEST E.1 — recalibrer_3max_v2 : génère un fichier centroides_v2.npz valide
# =============================================================================

def test_recalibrer_genere_centroides_valides(tmp_path):
    """main(n_spots=10) doit générer un .npz avec 3 streets (50, 3)."""
    import recalibrer_3max_v2 as cal

    output = str(tmp_path / "centroides_v2.npz")
    cal.main(n_spots=60, output_path=output, n_workers=1, seed=42)

    assert (tmp_path / "centroides_v2.npz").exists(), "Fichier non généré"

    c = np.load(output)
    assert set(c.files) == {'flop', 'turn', 'river'}, (
        f"Clés attendues flop/turn/river, obtenu {c.files}")

    for street in ('flop', 'turn', 'river'):
        arr = c[street]
        assert arr.shape == (50, 3), (
            f"{street} : shape attendu (50, 3), obtenu {arr.shape}")
        # E[HS] et E[HS²] (colonnes 0 et 1) dans [0, 1]
        assert arr[:, 0].min() >= -0.01 and arr[:, 0].max() <= 1.01, (
            f"{street} E[HS] hors [0,1] : [{arr[:,0].min():.3f}, {arr[:,0].max():.3f}]")
        assert arr[:, 1].min() >= -0.01 and arr[:, 1].max() <= 1.01, (
            f"{street} E[HS²] hors [0,1] : [{arr[:,1].min():.3f}, {arr[:,1].max():.3f}]")
        # Potentiel (colonne 2) dans [-1, 1]
        assert arr[:, 2].min() >= -1.01 and arr[:, 2].max() <= 1.01, (
            f"{street} potentiel hors [-1,1] : [{arr[:,2].min():.3f}, {arr[:,2].max():.3f}]")


# =============================================================================
# TEST E.1b — recalibrer_3max_v2 : centroïdes diversifiés (pas tous identiques)
# =============================================================================

def test_recalibrer_genere_centroides_diversifies(tmp_path):
    """Les centroïdes générés doivent être distincts (vrai K-means, pas zeros)."""
    import recalibrer_3max_v2 as cal

    output = str(tmp_path / "centroides_v2.npz")
    cal.main(n_spots=60, output_path=output, n_workers=1, seed=42)

    c = np.load(output)
    for street in ('flop', 'turn', 'river'):
        arr = c[street]
        # Au moins 5 centroïdes avec des valeurs E[HS] distinctes
        unique_hs = np.unique(arr[:, 0].round(4))
        assert len(unique_hs) >= 5, (
            f"{street} : seulement {len(unique_hs)} valeurs E[HS] distinctes "
            f"(attendu ≥5) — centroïdes pas calibrés ?")
