"""
Tests TDD pour abstraction/card_clustering.py — Phase 2 abstraction V2.
Ajoutés un par un (TDD atomique, Étape B de la migration P6).
"""

from treys import Card


# =============================================================================
# TEST A.1 — compute_features : signature et types de retour
# =============================================================================

def test_compute_features_returns_tuple_of_three_floats():
    """compute_features doit retourner (E[HS], E[HS²], potentiel) — 3 floats."""
    from abstraction.card_clustering import compute_features

    cartes_hero = [Card.new('As'), Card.new('Ks')]
    board = [Card.new('Qh'), Card.new('8c'), Card.new('3d')]

    result = compute_features(cartes_hero, board, street='flop',
                              n_sim=10, seed=42)

    assert isinstance(result, tuple), f"Attendu tuple, obtenu {type(result)}"
    assert len(result) == 3, f"Attendu 3 éléments, obtenu {len(result)}"
    assert all(isinstance(x, float) for x in result), (
        f"Tous les éléments doivent être des floats : {result}")
    assert all(0.0 <= x <= 1.0 for x in result[:2]), (
        f"E[HS] et E[HS²] doivent être dans [0, 1] : {result[:2]}")
    assert -1.0 <= result[2] <= 1.0, (
        f"Potentiel doit être dans [-1, 1] : {result[2]}")
