"""
Tests P7 — Refonte abstraction historique + stacks (Spin & Rush).

Voir docs/investigations/P7-hist-stacks-abstraction/spec.md
TDD strict : 1 test → 1 stub → 1 commit.
"""


# =============================================================================
# RED.1 — _abstraire_sizing : frontières S / M / L (Variante B)
# =============================================================================

def test_abstraire_sizing_S_M_L_frontieres():
    """Mapping bucket _discretiser_raise_frac → S/M/L (Variante B Spin & Rush).
       r1 → S (micro), r2 → M (half-pot), r3 → L (pot/commit), r4 → L (overbet).
    """
    from abstraction.info_set import _abstraire_sizing
    assert _abstraire_sizing(1) == 'S'
    assert _abstraire_sizing(2) == 'M'
    assert _abstraire_sizing(3) == 'L'
    assert _abstraire_sizing(4) == 'L'


# =============================================================================
# RED.2 — _abstraire_sizing : défensif bucket 0 → S
# =============================================================================

def test_abstraire_sizing_defensif_bucket_0():
    """Bucket 0 (théoriquement non généré) → 'S' par sécurité défensive."""
    from abstraction.info_set import _abstraire_sizing
    assert _abstraire_sizing(0) == 'S'
