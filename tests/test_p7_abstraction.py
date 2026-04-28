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


# =============================================================================
# RED.3 — _format_hist_avec_cap : sizing remplace chiffres r{N} → r{S/M/L}
# =============================================================================

def test_format_hist_sizing_remplace_chiffres():
    """'xr1r3' (raw) → 'xrSrL' (Variante B : r1→S, r3→L)."""
    from abstraction.info_set import _format_hist_avec_cap
    assert _format_hist_avec_cap('xr1r3') == 'xrSrL'
