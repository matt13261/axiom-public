"""
Tests TDD pour ai/mccfr.py — migration V1 → V2 (Étape F.5).
"""


# =============================================================================
# TEST F.5 — MCCFRHoldEm utilise AbstractionCartesV2
# =============================================================================

def test_mccfr_uses_v2_abstraction():
    """MCCFRHoldEm doit utiliser AbstractionCartesV2 après migration."""
    from ai.mccfr import MCCFRHoldEm
    from abstraction.card_abstraction import AbstractionCartesV2

    solver = MCCFRHoldEm()

    assert isinstance(solver._abs_cartes, AbstractionCartesV2), (
        f"_abs_cartes doit être AbstractionCartesV2, obtenu {type(solver._abs_cartes)}")
