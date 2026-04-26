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
