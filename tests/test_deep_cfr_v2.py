"""
Tests TDD pour ai/deep_cfr.py — migration V1 → V2 (Étape F.3).
"""


# =============================================================================
# TEST F.3 — DeepCFRSolver utilise AbstractionCartesV2
# =============================================================================

def test_deep_cfr_uses_v2_abstraction():
    """DeepCFRSolver doit utiliser AbstractionCartesV2 après migration."""
    from ai.deep_cfr import DeepCFR
    from abstraction.card_abstraction import AbstractionCartesV2

    solver = DeepCFR()

    assert isinstance(solver._abs_cartes, AbstractionCartesV2), (
        f"_abs_cartes doit être AbstractionCartesV2, obtenu {type(solver._abs_cartes)}")
