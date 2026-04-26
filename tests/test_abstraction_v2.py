"""
Tests TDD pour AbstractionCartesV2 — Phase 2 abstraction V2.
Ajoutés un par un (TDD atomique, Étape B de la migration P6).
"""


# =============================================================================
# TEST A.2 — AbstractionCartesV2 : existence et constantes
# =============================================================================

def test_v2_class_exists_with_expected_constants():
    """AbstractionCartesV2 doit exister avec les bonnes constantes de classe."""
    from abstraction.card_abstraction import AbstractionCartesV2

    assert hasattr(AbstractionCartesV2, 'NB_BUCKETS_FLOP')
    assert hasattr(AbstractionCartesV2, 'NB_BUCKETS_TURN')
    assert hasattr(AbstractionCartesV2, 'NB_BUCKETS_RIVER')
    assert hasattr(AbstractionCartesV2, 'NB_BUCKETS_PREFLOP')
    assert AbstractionCartesV2.NB_BUCKETS_FLOP   == 50
    assert AbstractionCartesV2.NB_BUCKETS_TURN   == 50
    assert AbstractionCartesV2.NB_BUCKETS_RIVER  == 50
    assert AbstractionCartesV2.NB_BUCKETS_PREFLOP == 8


# =============================================================================
# TEST B.8 — AbstractionCartesV2 : bucket_postflop déterministe
# =============================================================================

def test_v2_bucket_postflop_deterministic():
    """Deux appels identiques à bucket_postflop retournent le même bucket."""
    from treys import Card
    from abstraction.card_abstraction import AbstractionCartesV2

    v2     = AbstractionCartesV2()
    cartes = [Card.new('8c'), Card.new('7c')]
    board  = [Card.new('9h'), Card.new('6d'), Card.new('2s')]

    b1 = v2.bucket_postflop(cartes, board)
    b2 = v2.bucket_postflop(cartes, board)

    assert b1 == b2, f"bucket_postflop non déterministe : {b1} != {b2}"


# =============================================================================
# TEST B.9 — AbstractionCartesV2 : distingue draw et paire faible
# =============================================================================

def test_v2_distinguishes_draw_from_pair():
    """J♠T♠ (OESD) et 6♥6♦ (paire faible) sur même board → buckets différents."""
    from treys import Card
    from abstraction.card_abstraction import AbstractionCartesV2

    v2    = AbstractionCartesV2()
    board = [Card.new('Qh'), Card.new('8c'), Card.new('3d')]

    b_draw = v2.bucket_postflop([Card.new('Js'), Card.new('Ts')], board)
    b_pair = v2.bucket_postflop([Card.new('6h'), Card.new('6d')], board)

    assert b_draw != b_pair, (
        f"Draw (J♠T♠) et paire faible (6♥6♦) doivent avoir des buckets distincts, "
        f"obtenu b_draw={b_draw}, b_pair={b_pair}")


# =============================================================================
# TEST B.10 — AbstractionCartesV2 : chargement centroïdes depuis disque
# =============================================================================

def test_v2_load_centroids_from_disk(tmp_path):
    """V2 charge correctement les centroïdes depuis un fichier .npz."""
    import numpy as np
    from abstraction.card_abstraction import AbstractionCartesV2

    flop  = np.random.RandomState(0).rand(50, 3).astype(np.float32)
    turn  = np.random.RandomState(1).rand(50, 3).astype(np.float32)
    river = np.random.RandomState(2).rand(50, 3).astype(np.float32)
    path  = str(tmp_path / "centroides_v2.npz")
    np.savez(path, flop=flop, turn=turn, river=river)

    v2 = AbstractionCartesV2(centroides_path=path)

    assert v2.centroides is not None, "centroides doit être chargé depuis le fichier"
    assert v2.centroides['flop'].shape  == (50, 3)
    assert v2.centroides['turn'].shape  == (50, 3)
    assert v2.centroides['river'].shape == (50, 3)


# =============================================================================
# TEST B.11 — AbstractionCartesV2 : API compatible avec V1
# =============================================================================

def test_v2_api_compatible_v1():
    """V2 doit exposer les mêmes méthodes publiques que V1 (bucket_postflop, bucket_preflop)."""
    from abstraction.card_abstraction import AbstractionCartesV2

    required_methods = ['bucket_postflop', 'bucket_preflop']
    for method in required_methods:
        assert hasattr(AbstractionCartesV2, method), (
            f"AbstractionCartesV2 manque la méthode publique : {method!r}")


# =============================================================================
# TEST B.12 — Régression : OFT non affecté par initialisation de V2
# =============================================================================

def test_oft_unaffected_by_v2_initialization():
    """Créer une instance V2 ne perturbe pas le fonctionnement d'OpponentTracker."""
    import numpy as np
    from abstraction.card_abstraction import AbstractionCartesV2
    from ai.opponent_tracker import OpponentTracker
    from ai.exploit_mixer   import ExploitMixer

    # Initialiser V2 (ne doit pas avoir d'effets de bord sur OFT)
    _v2 = AbstractionCartesV2()

    # OFT doit continuer à fonctionner normalement
    tracker = OpponentTracker()
    for _ in range(30):
        tracker.observer_action(seat_index=1, action=2,
                                contexte={'phase': 'preflop'})

    mixer  = ExploitMixer(tracker)
    bp     = np.array([0.1, 0.0, 0.8, 0.1, 0., 0., 0., 0., 0.], dtype=np.float32)
    result = mixer.ajuster(bp, seat_index=1, game_type='3max')

    assert abs(result.sum() - 1.0) < 1e-5, (
        f"Distribution OFT invalide après init V2 : sum={result.sum()}")


# =============================================================================
# TEST D.1 — AbstractionCartesV2 : init avec dict centroides direct
# =============================================================================

def test_v2_init_with_centroides_dict():
    """V2 accepte un dict centroides passé directement (sans fichier .npz)."""
    import numpy as np
    from abstraction.card_abstraction import AbstractionCartesV2

    centroides = {
        'flop':  np.zeros((50, 3), dtype=np.float32),
        'turn':  np.zeros((50, 3), dtype=np.float32),
        'river': np.zeros((50, 3), dtype=np.float32),
    }

    v2 = AbstractionCartesV2(centroides=centroides)

    assert v2.centroides is not None, "centroides doit être stocké"
    assert set(v2.centroides.keys()) == {'flop', 'turn', 'river'}
    assert v2.centroides['flop'].shape  == (50, 3)
    assert v2.centroides['turn'].shape  == (50, 3)
    assert v2.centroides['river'].shape == (50, 3)


# =============================================================================
# TEST D.2 — AbstractionCartesV2 : bucket_postflop utilise les centroïdes réels
# =============================================================================

def test_v2_bucket_postflop_uses_centroids():
    """bucket_postflop retourne l'index du centroïde le plus proche."""
    import numpy as np
    from treys import Card
    from abstraction.card_abstraction import AbstractionCartesV2

    # 50 centroïdes flop : un seul proche des features attendues pour AKs sur Q83
    # On place le centroïde 7 à [0.70, 0.60, 0.05] — proche de AKs equity flop
    # Tous les autres à [0.0, 0.0, 0.0] (très éloignés)
    flop_cents = np.zeros((50, 3), dtype=np.float32)
    flop_cents[7] = [0.70, 0.60, 0.05]

    centroides = {
        'flop':  flop_cents,
        'turn':  np.zeros((50, 3), dtype=np.float32),
        'river': np.zeros((50, 3), dtype=np.float32),
    }

    v2     = AbstractionCartesV2(centroides=centroides)
    cartes = [Card.new('As'), Card.new('Ks')]
    board  = [Card.new('Qh'), Card.new('8c'), Card.new('3d')]

    bucket = v2.bucket_postflop(cartes, board, street='flop')

    assert bucket == 7, (
        f"bucket_postflop doit retourner 7 (centroïde proche de AKs equity), "
        f"obtenu {bucket}")


# =============================================================================
# TEST D.3 — AbstractionCartesV2 : bucket_preflop retourne int dans [0, 7]
# =============================================================================

def test_v2_bucket_preflop_returns_int_in_range():
    """bucket_preflop délègue à V1 et retourne un entier dans [0, 7]."""
    from treys import Card
    from abstraction.card_abstraction import AbstractionCartesV2

    v2    = AbstractionCartesV2()
    mains = [
        [Card.new('As'), Card.new('Ks')],
        [Card.new('2h'), Card.new('7d')],
        [Card.new('Jc'), Card.new('Jd')],
        [Card.new('8s'), Card.new('5h')],
    ]

    for cartes in mains:
        b = v2.bucket_preflop(cartes)
        assert isinstance(b, int), f"bucket_preflop doit retourner int, obtenu {type(b)}"
        assert 0 <= b < 8, f"bucket_preflop hors [0,7] : {b} pour {cartes}"


# =============================================================================
# TEST D.4 — AbstractionCartesV2 : LRU cache accélère les appels répétés
# =============================================================================

def test_v2_caches_repeated_calls():
    """100 appels identiques doivent être bien plus rapides qu'1 appel × 100."""
    import time
    import numpy as np
    from treys import Card
    from abstraction.card_abstraction import AbstractionCartesV2

    rng   = np.random.RandomState(0)
    cents = {
        'flop':  rng.rand(50, 3).astype(np.float32),
        'turn':  rng.rand(50, 3).astype(np.float32),
        'river': rng.rand(50, 3).astype(np.float32),
    }
    v2     = AbstractionCartesV2(centroides=cents)
    cartes = [Card.new('As'), Card.new('Ks')]
    board  = [Card.new('Qh'), Card.new('8c'), Card.new('3d')]

    # Premier appel (cache miss)
    t0    = time.perf_counter()
    b_ref = v2.bucket_postflop(cartes, board, street='flop')
    t_miss = time.perf_counter() - t0

    # 99 appels répétés (cache hits)
    t0 = time.perf_counter()
    for _ in range(99):
        b = v2.bucket_postflop(cartes, board, street='flop')
    t_hits = time.perf_counter() - t0

    assert b == b_ref, "Le cache doit retourner le même bucket"
    # Les 99 hits doivent prendre moins que 5x le coût d'un seul miss
    assert t_hits < t_miss * 5, (
        f"Cache inefficace : 99 hits={t_hits*1000:.1f}ms, "
        f"1 miss={t_miss*1000:.1f}ms (ratio={t_hits/t_miss:.1f}x)")


# =============================================================================
# TEST D.5 — Régression : OFT + V2 réelle sans interaction parasite
# =============================================================================

def test_oft_works_with_v2_real_buckets():
    """V2 avec centroïdes réels et OFT fonctionnent indépendamment."""
    import numpy as np
    from treys import Card
    from abstraction.card_abstraction import AbstractionCartesV2
    from ai.opponent_tracker import OpponentTracker
    from ai.exploit_mixer   import ExploitMixer

    # V2 avec centroïdes réalistes (random scaled)
    rng   = np.random.RandomState(7)
    cents = {
        'flop':  rng.rand(50, 3).astype(np.float32),
        'turn':  rng.rand(50, 3).astype(np.float32),
        'river': rng.rand(50, 3).astype(np.float32),
    }
    v2 = AbstractionCartesV2(centroides=cents)

    # V2 produit des buckets normaux sur quelques mains
    cartes = [Card.new('As'), Card.new('Ks')]
    board  = [Card.new('Qh'), Card.new('8c'), Card.new('3d')]
    b = v2.bucket_postflop(cartes, board, street='flop')
    assert 0 <= b < 50, f"bucket_postflop hors [0,49] : {b}"

    # OFT fonctionne normalement en parallèle
    tracker = OpponentTracker()
    for _ in range(30):
        tracker.observer_action(seat_index=1, action=2,
                                contexte={'phase': 'preflop'})

    mixer  = ExploitMixer(tracker)
    bp     = np.array([0.1, 0.0, 0.8, 0.1, 0., 0., 0., 0., 0.], dtype=np.float32)
    result = mixer.ajuster(bp, seat_index=1, game_type='3max')

    assert abs(result.sum() - 1.0) < 1e-5, (
        f"Distribution OFT invalide avec V2 réelle : sum={result.sum()}")
