"""
Tests TDD pour solver/subgame_solver.py — migration V1 → V2 (Étape F.1).
"""

import pytest


# =============================================================================
# TEST F.1 — SolveurSousJeu utilise AbstractionCartesV2 par défaut
# =============================================================================

def test_subgame_solver_uses_v2_abstraction():
    """SolveurSousJeu doit utiliser AbstractionCartesV2 après migration."""
    from solver.subgame_solver import SolveurSousJeu
    from abstraction.card_abstraction import AbstractionCartesV2

    solveur = SolveurSousJeu()

    assert isinstance(solveur._abs_cartes, AbstractionCartesV2), (
        f"_abs_cartes doit être AbstractionCartesV2, obtenu {type(solveur._abs_cartes)}")


# =============================================================================
# TEST F.1b — SolveurSousJeu._abs_cartes est l'instance utilisée (pas le singleton V1)
# =============================================================================

def test_subgame_solver_abs_cartes_is_not_v1_singleton():
    """_abs_cartes ne doit pas être l'instance globale V1 abstraction_cartes."""
    from solver.subgame_solver import SolveurSousJeu
    from abstraction.card_abstraction import abstraction_cartes as v1_global

    solveur = SolveurSousJeu()

    assert solveur._abs_cartes is not v1_global, (
        "_abs_cartes ne doit pas pointer vers le singleton V1")
