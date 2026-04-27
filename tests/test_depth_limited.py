"""
Tests TDD pour solver/depth_limited.py — migration V1 → V2 (Étape F.2).
"""


# =============================================================================
# TEST F.2 — SolveurProfondeurLimitee utilise AbstractionCartesV2
# =============================================================================

def test_depth_limited_uses_v2_abstraction():
    """SolveurProfondeurLimitee doit utiliser AbstractionCartesV2 après migration."""
    from solver.depth_limited import SolveurProfondeurLimitee
    from abstraction.card_abstraction import AbstractionCartesV2

    solveur = SolveurProfondeurLimitee()

    assert isinstance(solveur._abs_cartes, AbstractionCartesV2), (
        f"_abs_cartes doit être AbstractionCartesV2, obtenu {type(solveur._abs_cartes)}")
