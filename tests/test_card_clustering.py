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


# =============================================================================
# TEST B.1 — compute_features : déterminisme (même seed → même résultat)
# =============================================================================

def test_compute_features_deterministic():
    """Deux appels identiques avec seed=42 doivent retourner les mêmes features."""
    from abstraction.card_clustering import compute_features

    cartes = [Card.new('9c'), Card.new('8c')]
    board  = [Card.new('Th'), Card.new('7d'), Card.new('3s')]

    r1 = compute_features(cartes, board, street='flop', n_sim=10, seed=42)
    r2 = compute_features(cartes, board, street='flop', n_sim=10, seed=42)

    assert r1 == r2, f"Résultats non déterministes : {r1} != {r2}"


# =============================================================================
# TEST B.2 — compute_features : potentiel ≈ 0 à la river
# =============================================================================

def test_potentiel_zero_river():
    """À la river (board complet 5 cartes), le potentiel doit être quasi-nul."""
    from abstraction.card_clustering import compute_features

    cartes = [Card.new('Ah'), Card.new('Kh')]
    board  = [Card.new('Qd'), Card.new('Jc'), Card.new('Ts'),
              Card.new('2h'), Card.new('7d')]  # river complète

    _, _, potentiel = compute_features(cartes, board, street='river',
                                       n_sim=10, seed=42)

    assert abs(potentiel) < 0.01, (
        f"Potentiel à la river doit être ~0, obtenu {potentiel}")


# =============================================================================
# TEST B.3 — compute_features : potentiel > 0 au flop avec drawing hand
# =============================================================================

def test_potentiel_positive_flop_drawing_hand():
    """Au flop avec un flush draw + straight draw, le potentiel doit être > 0.05."""
    from abstraction.card_clustering import compute_features

    # J♠T♠ sur Q♥8♣3♦ : OESD (J-T-9-8 ou T-9-8-7) + backdoor FD
    cartes = [Card.new('Js'), Card.new('Ts')]
    board  = [Card.new('Qh'), Card.new('8c'), Card.new('3d')]

    _, _, potentiel = compute_features(cartes, board, street='flop',
                                       n_sim=10, seed=42)

    assert potentiel > 0.05, (
        f"Drawing hand au flop doit avoir potentiel > 0.05, obtenu {potentiel}")


# =============================================================================
# TEST B.4 — compute_features : E[HS²] ∈ [0,1] et ≥ E[HS]² (Jensen)
# =============================================================================

def test_e_hs_squared_within_bounds():
    """E[HS²] doit être dans [0,1] et satisfaire Jensen : E[HS²] >= E[HS]²."""
    from abstraction.card_clustering import compute_features

    cartes = [Card.new('9c'), Card.new('8c')]
    board  = [Card.new('Th'), Card.new('7d'), Card.new('3s')]

    e_hs, e_hs2, _ = compute_features(cartes, board, street='flop',
                                       n_sim=10, seed=42)

    assert 0.0 <= e_hs2 <= 1.0, f"E[HS²] hors [0,1] : {e_hs2}"
    assert e_hs2 >= e_hs ** 2 - 1e-6, (
        f"Jensen violé : E[HS²]={e_hs2:.4f} < E[HS]²={e_hs**2:.4f}")


# =============================================================================
# TEST B.5 — fit_centroids : shape de retour (n_clusters, 3)
# =============================================================================

def test_fit_centroids_returns_correct_shape():
    """fit_centroids doit retourner un array de forme (n_clusters, 3)."""
    import numpy as np
    from abstraction.card_clustering import fit_centroids

    rng      = np.random.RandomState(42)
    features = rng.rand(100, 3).astype(np.float32)

    centroids = fit_centroids(features, n_clusters=50, seed=42)

    assert centroids.shape == (50, 3), (
        f"Shape attendu (50, 3), obtenu {centroids.shape}")


# =============================================================================
# TEST B.6 — fit_centroids : déterminisme (même seed → mêmes centroïdes)
# =============================================================================

def test_fit_centroids_deterministic():
    """Deux appels avec les mêmes features et seed=42 donnent les mêmes centroïdes."""
    import numpy as np
    from abstraction.card_clustering import fit_centroids

    rng      = np.random.RandomState(0)
    features = rng.rand(200, 3).astype(np.float32)

    c1 = fit_centroids(features, n_clusters=50, seed=42)
    c2 = fit_centroids(features, n_clusters=50, seed=42)

    np.testing.assert_array_equal(c1, c2,
        err_msg="fit_centroids non déterministe avec même seed")


# =============================================================================
# TEST B.7 — predict_bucket : résultat dans [0, n_clusters-1]
# =============================================================================

def test_predict_bucket_in_range():
    """predict_bucket doit retourner un entier dans [0, 49]."""
    import numpy as np
    from abstraction.card_clustering import predict_bucket

    rng       = np.random.RandomState(42)
    centroids = rng.rand(50, 3).astype(np.float32)

    for _ in range(20):
        features_vec = rng.rand(3).astype(np.float32)
        b = predict_bucket(features_vec, centroids)
        assert isinstance(b, (int, np.integer)), f"Attendu int, obtenu {type(b)}"
        assert 0 <= b < 50, f"Bucket hors plage : {b}"


# =============================================================================
# TEST B.13 — compute_features : E[HS] dans une plage réaliste (main premium)
# =============================================================================

def test_compute_features_e_hs_in_realistic_range():
    """A♠K♠ sur board sec 2♥7♣Q♦ au flop : E[HS] doit être dans (0.55, 0.75)."""
    from abstraction.card_clustering import compute_features

    cartes = [Card.new('As'), Card.new('Ks')]
    board  = [Card.new('2h'), Card.new('7c'), Card.new('Qd')]

    e_hs, _, _ = compute_features(cartes, board, street='flop',
                                   n_sim=10, seed=42)

    assert 0.55 < e_hs < 0.75, (
        f"E[HS] pour A♠K♠ sur board sec doit être dans (0.55, 0.75), "
        f"obtenu {e_hs:.4f}")


# =============================================================================
# TEST B.14 — fit_centroids : centroïdes diversifiés (pas tous nuls)
# =============================================================================

def test_fit_centroids_produces_diverse_centroids():
    """fit_centroids doit produire des centroïdes distincts (pas tous identiques)."""
    import numpy as np
    from abstraction.card_clustering import fit_centroids

    rng      = np.random.RandomState(42)
    features = rng.rand(200, 3).astype(np.float32)

    centroids = fit_centroids(features, n_clusters=10, seed=42)

    # Vérifier qu'au moins 5 centroïdes distincts existent (norm > 0.01)
    norms = [np.linalg.norm(centroids[i] - centroids[j])
             for i in range(len(centroids))
             for j in range(i + 1, len(centroids))]

    n_distinct_pairs = sum(1 for n in norms if n > 0.01)
    assert n_distinct_pairs >= 5, (
        f"Centroïdes pas assez diversifiés : seulement {n_distinct_pairs} "
        f"paires à distance > 0.01 (attendu ≥ 5)")
