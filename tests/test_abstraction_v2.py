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
