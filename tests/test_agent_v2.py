"""
Tests TDD pour ai/agent.py — migration V1 → V2 (Étape F.4).
"""


# =============================================================================
# TEST F.4 — AgentAxiom utilise AbstractionCartesV2 pour les deux instances
# =============================================================================

def test_agent_uses_v2_abstraction():
    """AgentAxiom doit utiliser AbstractionCartesV2 pour _abs_cartes et _abs_cartes_hu."""
    from ai.agent import AgentAXIOM as AgentAxiom
    from abstraction.card_abstraction import AbstractionCartesV2

    agent = AgentAxiom()

    assert isinstance(agent._abs_cartes, AbstractionCartesV2), (
        f"_abs_cartes doit être AbstractionCartesV2, obtenu {type(agent._abs_cartes)}")
    assert isinstance(agent._abs_cartes_hu, AbstractionCartesV2), (
        f"_abs_cartes_hu doit être AbstractionCartesV2, obtenu {type(agent._abs_cartes_hu)}")
