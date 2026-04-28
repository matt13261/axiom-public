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


# =============================================================================
# RED.4 — _format_hist_avec_cap : cap=6 garde les 6 dernières actions
# =============================================================================

def test_format_hist_cap_garde_dernieres_actions():
    """8 actions raw → cap=6 → 6 dernières actions abstraites."""
    from abstraction.info_set import _format_hist_avec_cap
    # raw 'xr1r2r3cr1r2a' = 8 tokens : x, r1, r2, r3, c, r1, r2, a
    # abstraits Variante B : x, rS, rM, rL, c, rS, rM, a
    # cap=6 garde les 6 dernières : rM, rL, c, rS, rM, a → 'rMrLcrSrMa'
    assert _format_hist_avec_cap('xr1r2r3cr1r2a') == 'rMrLcrSrMa'


# =============================================================================
# RED.5 — _format_hist_avec_cap : hist court ≤ cap inchangé (sauf abstraction)
# =============================================================================

def test_format_hist_inferieur_cap_inchange():
    """Hist ≤ cap actions : abstraction sizing seule, aucune action droppée."""
    from abstraction.info_set import _format_hist_avec_cap
    # 4 tokens (≤ 6) : xr2r3c → x, rM, rL, c → 'xrMrLc'
    assert _format_hist_avec_cap('xr2r3c') == 'xrMrLc'
    # hist vide reste vide
    assert _format_hist_avec_cap('') == ''


# =============================================================================
# RED.6 — PALIERS_STACK_SPIN_RUSH : 7 niveaux exacts
# =============================================================================

def test_paliers_stack_spin_rush_7_niveaux():
    """Constante PALIERS_STACK_SPIN_RUSH = [0, 5, 8, 13, 19, 27, 41]."""
    from abstraction.info_set import PALIERS_STACK_SPIN_RUSH
    assert PALIERS_STACK_SPIN_RUSH == [0, 5, 8, 13, 19, 27, 41]
    assert len(PALIERS_STACK_SPIN_RUSH) == 7


# =============================================================================
# RED.7 — Format clé infoset 7 segments INCHANGÉ après P7
# =============================================================================

def test_cle_infoset_format_7_segments_preserve():
    """Invariant projet : la clé doit avoir 7 segments séparés par |.
       Format : PHASE|pos=P|bucket=B|pot=N|stacks=(...)|hist=H|raise=R"""
    from engine.player import Joueur, TypeJoueur
    from engine.game_state import EtatJeu
    from abstraction.info_set import construire_cle_infoset
    j1 = Joueur("J1", TypeJoueur.AXIOM, 1500, 0)
    j2 = Joueur("J2", TypeJoueur.AXIOM, 1500, 1)
    j3 = Joueur("J3", TypeJoueur.AXIOM, 1500, 2)
    etat = EtatJeu([j1, j2, j3], petite_blinde=10, grande_blinde=20)
    etat.nouvelle_main()
    cle = construire_cle_infoset(etat, j1)
    segments = cle.split('|')
    assert len(segments) == 7, f"Attendu 7 segments, obtenu {len(segments)}: {cle}"
    assert segments[0] in ('PREFLOP', 'FLOP', 'TURN', 'RIVER')
    assert segments[1].startswith('pos=')
    assert segments[2].startswith('bucket=')
    assert segments[3].startswith('pot=')
    assert segments[4].startswith('stacks=(') and segments[4].endswith(')')
    assert segments[5].startswith('hist=')
    assert segments[6].startswith('raise=')


# =============================================================================
# RED.8 — Clé infoset utilise PALIERS_STACK_SPIN_RUSH (7 niveaux, P3..P50)
# =============================================================================

def test_cle_infoset_utilise_paliers_spin_rush():
    """Stacks dans la clé doivent appartenir à PALIERS_STACK_SPIN_RUSH.
       Avec stacks=1500 jetons et grande_blinde=20 → 75 BB → palier 41 (P50)."""
    from engine.player import Joueur, TypeJoueur
    from engine.game_state import EtatJeu
    from abstraction.info_set import construire_cle_infoset, PALIERS_STACK_SPIN_RUSH
    j1 = Joueur("J1", TypeJoueur.AXIOM, 1500, 0)
    j2 = Joueur("J2", TypeJoueur.AXIOM, 1500, 1)
    j3 = Joueur("J3", TypeJoueur.AXIOM, 1500, 2)
    etat = EtatJeu([j1, j2, j3], petite_blinde=10, grande_blinde=20)
    etat.nouvelle_main()
    cle = construire_cle_infoset(etat, j1)
    # segment stacks="stacks=(A,B,C)" — extraire les 3 valeurs
    seg_stacks = cle.split('|')[4]              # "stacks=(...)"
    inner = seg_stacks[len('stacks=('):-1]      # "A,B,C"
    valeurs = [int(v) for v in inner.split(',')]
    for v in valeurs:
        assert v in PALIERS_STACK_SPIN_RUSH, (
            f"Stack {v} pas dans PALIERS_STACK_SPIN_RUSH={PALIERS_STACK_SPIN_RUSH}"
        )


# =============================================================================
# RED.9 — Clé infoset : segment hist contient rS/rM/rL pas r1/r2/r3/r4
# =============================================================================

def test_cle_infoset_hist_abstrait_format_S_M_L():
    """Le segment hist= doit utiliser le format abstrait Variante B."""
    from engine.player import Joueur, TypeJoueur
    from engine.game_state import EtatJeu
    from abstraction.info_set import construire_cle_infoset
    j1 = Joueur("J1", TypeJoueur.AXIOM, 1500, 0)
    j2 = Joueur("J2", TypeJoueur.AXIOM, 1500, 1)
    j3 = Joueur("J3", TypeJoueur.AXIOM, 1500, 2)
    etat = EtatJeu([j1, j2, j3], petite_blinde=10, grande_blinde=20)
    etat.nouvelle_main()
    # Injecter directement un hist raw 'xr1r3' dans la phase courante (PREFLOP=0)
    etat.historique_phases[0] = 'xr1r3'
    cle = construire_cle_infoset(etat, j1)
    seg_hist = cle.split('|')[5]                # "hist=..."
    valeur = seg_hist[len('hist='):]
    # Format abstrait attendu : 'xrSrL' (r1→S, r3→L Variante B)
    assert valeur == 'xrSrL', f"Hist non abstrait : {valeur!r}"
    # Aucun chiffre ne doit subsister dans le segment hist
    assert not any(c.isdigit() for c in valeur), f"Chiffre dans hist : {valeur!r}"


# =============================================================================
# RED.10 — mccfr._cle_infoset alignée avec info_set (cohérence cross-module)
# =============================================================================

def test_mccfr_cle_infoset_aligne_avec_info_set():
    """mccfr.MCCFRHoldEm._cle_infoset doit utiliser le format P7 (rS/M/L + paliers Spin & Rush)."""
    from ai.mccfr import MCCFRHoldEm
    from abstraction.info_set import PALIERS_STACK_SPIN_RUSH
    trainer = MCCFRHoldEm()
    # Etat dict minimal requis par _cle_infoset
    etat = {
        'phase'         : 0,                # PREFLOP
        'grande_blinde' : 20,
        'pot'           : 30,
        'stacks'        : [1500, 1500, 1500],
        'mise_courante' : 20,
        'buckets'       : [[3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]],
        'hist_phases'   : ['xr1r3', '', '', ''],
    }
    cle = trainer._cle_infoset(etat, joueur_idx=0)
    seg_hist = cle.split('|')[5]
    valeur_hist = seg_hist[len('hist='):]
    assert valeur_hist == 'xrSrL', f"Hist mccfr non abstrait : {valeur_hist!r}"
    seg_stacks = cle.split('|')[4]
    inner = seg_stacks[len('stacks=('):-1]
    valeurs = [int(v) for v in inner.split(',')]
    for v in valeurs:
        assert v in PALIERS_STACK_SPIN_RUSH, (
            f"Stack mccfr {v} pas dans PALIERS_STACK_SPIN_RUSH={PALIERS_STACK_SPIN_RUSH}"
        )


# =============================================================================
# RED.11 — train_hu._cle_infoset alignée avec format P7 (HU 2 stacks)
# =============================================================================

def test_train_hu_cle_infoset_aligne():
    """train_hu.MCCFRHeadsUp._cle_infoset doit utiliser le format P7 (rS/M/L + paliers Spin & Rush)."""
    from train_hu import MCCFRHeadsUp
    from abstraction.info_set import PALIERS_STACK_SPIN_RUSH
    trainer = MCCFRHeadsUp()
    etat = {
        'phase'         : 0,                # PREFLOP HU
        'grande_blinde' : 20,
        'pot'           : 30,
        'stacks'        : [1500, 1500],     # 2 stacks (HU)
        'mise_courante' : 20,
        'buckets'       : [[3, 0, 0, 0], [4, 0, 0, 0]],
        'hist_phases'   : ['xr2r4', '', '', ''],
    }
    cle = trainer._cle_infoset(etat, joueur_idx=0)
    seg_hist = cle.split('|')[5]
    valeur_hist = seg_hist[len('hist='):]
    # Variante B : r2→M, r4→L → 'xrMrL'
    assert valeur_hist == 'xrMrL', f"Hist HU non abstrait : {valeur_hist!r}"
    seg_stacks = cle.split('|')[4]
    inner = seg_stacks[len('stacks=('):-1]
    valeurs = [int(v) for v in inner.split(',')]
    assert len(valeurs) == 2, "HU doit avoir 2 stacks"
    for v in valeurs:
        assert v in PALIERS_STACK_SPIN_RUSH, (
            f"Stack HU {v} pas dans PALIERS_STACK_SPIN_RUSH={PALIERS_STACK_SPIN_RUSH}"
        )


# =============================================================================
# RED.12 — Cardinalité hist < 2500 après MCCFR random play (matche calcul §7)
# =============================================================================

def test_cardinalite_hist_random_play_inferieure_2500():
    """Après 2K iter MCCFR, le segment hist doit avoir < 2500 valeurs uniques.
       Anti-régression : si dépassé en P7.5, tighten cap=4 ou itérer mapping
       (pas de relâchement du seuil — anti-TDD)."""
    from ai.mccfr import MCCFRHoldEm
    trainer = MCCFRHoldEm()
    trainer.entrainer(nb_iterations=2000, verbose=False, save_every=0)
    hists = set()
    for cle in trainer.noeuds.keys():
        seg_hist = cle.split('|')[5]    # "hist=..."
        hists.add(seg_hist[len('hist='):])
    assert len(hists) < 2500, f"Cardinalité hist = {len(hists)} (>= 2500)"


# =============================================================================
# RED.13 — Cardinalité stacks < 400 (7³=343 + marge densité)
# =============================================================================

def test_cardinalite_stacks_random_play_inferieure_400():
    """Après 1K iter MCCFR, le segment stacks doit avoir < 400 valeurs uniques.
       Théorique 7³ = 343 avec PALIERS_STACK_SPIN_RUSH ; 400 = marge densité."""
    from ai.mccfr import MCCFRHoldEm
    trainer = MCCFRHoldEm()
    trainer.entrainer(nb_iterations=1000, verbose=False, save_every=0)
    stacks_set = set()
    for cle in trainer.noeuds.keys():
        seg_stacks = cle.split('|')[4]
        stacks_set.add(seg_stacks)
    assert len(stacks_set) < 400, f"Cardinalité stacks = {len(stacks_set)} (>= 400)"


# =============================================================================
# RED.14 — Convergence Kuhn préservée (GUARDRAIL non-régression)
# =============================================================================

def test_kuhn_convergence_preservee():
    """Vanilla CFR sur Kuhn Poker doit toujours converger vers -1/18 (Nash).
       GUARDRAIL : déjà GREEN, vérifie que P7.4 ne casse pas la convergence."""
    from ai.mccfr import CFRKUHN, _KP_VALEUR_NASH as VALEUR_NASH
    cfr = CFRKUHN()
    valeur = cfr.entrainer(nb_iterations=10_000, verbose=False)
    assert abs(valeur - VALEUR_NASH) < 0.005, (
        f"Convergence Kuhn cassée : valeur={valeur:.6f} vs Nash={VALEUR_NASH:.6f}"
    )
