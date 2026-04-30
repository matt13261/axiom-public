"""
Test TA1 — creer_agent() doit charger les continuation strategies (Pluribus k=4)
et activer le solveur real-time, en plus du blueprint baseline + Deep CFR.

Voir docs/investigations/P7-hist-stacks-abstraction/architecture_audit.md
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_creer_agent_charge_continuations_et_solveur():
    """creer_agent() doit produire un agent avec continuations k=4 et solveurs."""
    from ai.agent import creer_agent

    agent = creer_agent(
        chemin_blueprint='data/strategy/blueprint_v1.pkl',
        verbose=False,
    )

    # Continuations Pluribus k=4 chargées (au moins baseline + 1 variante biaisée)
    assert hasattr(agent, '_blueprints_k')
    assert len(agent._blueprints_k) >= 2, (
        f"creer_agent doit charger les continuations Pluribus si dispo "
        f"(au moins baseline + 1 variante). _blueprints_k = {sorted(agent._blueprints_k)}"
    )

    # Solveur depth-limited et subgame activés
    assert agent._solveur is not None, (
        "creer_agent doit activer SolveurProfondeurLimitee (FLOP)"
    )
    assert agent._solveur_subgame is not None, (
        "creer_agent doit activer SolveurSousJeu (TURN/RIVER)"
    )


def test_creer_agent_avec_oft_desactive():
    """creer_agent(enable_oft=False) doit produire un mixer identité (no-op).

    Régression P10.A : permet de mesurer le delta winrate apporté par OFT.
    """
    import numpy as np
    from ai.agent import creer_agent

    agent = creer_agent(enable_oft=False, verbose=False)
    dist = np.array([0.1, 0.2, 0.3, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1])
    ajustee = agent.mixer.ajuster(dist, 0, 'NLHE_3MAX')
    np.testing.assert_array_almost_equal(dist, ajustee)
