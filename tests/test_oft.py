"""
TDD — Opponent Frequency Tracker (Exp 04, H4).

OpponentTracker + ExploitMixer non implementes : tous les tests sont RED
(ImportError attendu en phase RED).
"""
import numpy as np
import pytest

from ai.opponent_tracker import OpponentTracker
from ai.exploit_mixer import ExploitMixer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blueprint_equilibre():
    """Vecteur blueprint uniforme sur 9 actions (somme = 1.0)."""
    return np.full(9, 1 / 9)


def _tracker_calling_station(seat=0, n=30):
    """Tracker avec vpip > 0.60 sur n mains (70% CALL preflop)."""
    tracker = OpponentTracker()
    n_call = int(n * 0.70)
    n_fold = n - n_call
    for _ in range(n_call):
        tracker.observer_action(seat, action=2, contexte={"phase": "preflop"})
    for _ in range(n_fold):
        tracker.observer_action(seat, action=0, contexte={"phase": "preflop"})
    return tracker


def _tracker_fold_prone(seat=0, n=30):
    """Tracker avec fold_to_cbet > 0.65 sur n mains."""
    tracker = OpponentTracker()
    n_fold_cbet = int(n * 0.70)
    n_call_cbet = n - n_fold_cbet
    for _ in range(n_fold_cbet):
        tracker.observer_action(seat, action=0, contexte={"est_cbet_opp": True})
    for _ in range(n_call_cbet):
        tracker.observer_action(seat, action=2, contexte={"est_cbet_opp": True})
    return tracker


def _tracker_neutre(seat=0, n=30):
    """Tracker avec stats dans la zone GTO-like (profil neutre).

    vpip ≈ 0.35  (zone [0.20, 0.60])
    fold_to_cbet ≈ 0.50  (zone [0.40, 0.65])
    aggression_freq ≈ 0.40  (zone [0.25, 0.60])
    """
    tracker = OpponentTracker()
    # 35% CALL preflop, 65% FOLD → vpip ≈ 0.35
    for _ in range(int(n * 0.35)):
        tracker.observer_action(seat, action=2, contexte={"phase": "preflop"})
    for _ in range(int(n * 0.65)):
        tracker.observer_action(seat, action=0, contexte={"phase": "preflop"})
    # 50% fold sur cbet opp → fold_to_cbet = 0.50
    for _ in range(5):
        tracker.observer_action(seat, action=0, contexte={"est_cbet_opp": True})
        tracker.observer_action(seat, action=2, contexte={"est_cbet_opp": True})
    # 40% raise postflop → aggression ≈ 0.40
    for _ in range(4):
        tracker.observer_action(seat, action=3, contexte={"phase": "postflop"})
    for _ in range(6):
        tracker.observer_action(seat, action=2, contexte={"phase": "postflop"})
    return tracker


# ---------------------------------------------------------------------------
# Test 1 — confiance == 0.0 sous 5 mains
# ---------------------------------------------------------------------------

def test_oft_confiance_zero_sous_5_mains():
    """Sous 5 mains observees, confiance doit etre strictement 0.0."""
    tracker = OpponentTracker()
    for _ in range(4):
        tracker.observer_action(seat_index=0, action=2, contexte={})
    assert tracker.confiance(0) == 0.0, (
        f"confiance({tracker.mains_observees(0)} mains) = {tracker.confiance(0):.3f}, "
        f"attendu 0.0"
    )


# ---------------------------------------------------------------------------
# Test 1bis — confiance ≈ 0.48 a 17 mains (interpolation lineaire)
# ---------------------------------------------------------------------------

def test_oft_confiance_interpolee_a_17_mains():
    """A 17 mains : confiance = (17 - 5) / 25 = 0.48."""
    tracker = OpponentTracker()
    for _ in range(17):
        tracker.observer_action(seat_index=0, action=2, contexte={})
    attendu = (17 - 5) / 25  # 0.48
    assert abs(tracker.confiance(0) - attendu) < 0.01, (
        f"confiance(17 mains) = {tracker.confiance(0):.3f}, attendu ≈ {attendu:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 2 — confiance == 1.0 a 30 mains (WINDOW)
# ---------------------------------------------------------------------------

def test_oft_confiance_max_a_30_mains():
    """A 30 mains (WINDOW), confiance doit etre 1.0."""
    tracker = OpponentTracker()
    for _ in range(30):
        tracker.observer_action(seat_index=0, action=2, contexte={})
    assert tracker.confiance(0) == 1.0, (
        f"confiance(30 mains) = {tracker.confiance(0):.3f}, attendu 1.0"
    )


# ---------------------------------------------------------------------------
# Test 3 — vpip calling_station detecte
# ---------------------------------------------------------------------------

def test_oft_calling_station_vpip():
    """70% CALL preflop sur 30 mains → vpip > 0.60 (calling_station)."""
    tracker = _tracker_calling_station(seat=0, n=30)
    assert tracker.vpip(0) > 0.60, (
        f"vpip = {tracker.vpip(0):.2f}, attendu > 0.60"
    )


# ---------------------------------------------------------------------------
# Test 4 — calling_station → value bet augmente (directionnel + amplitude)
# ---------------------------------------------------------------------------

def test_exploit_calling_station_value_bet():
    """Adversaire calling_station fort (vpip=0.80, 30 mains, confiance=1.0) :
    - Directionnel : RAISE total augmente vs blueprint
    - Amplitude : RAISE_FORT (idx 7) >= 0.50 et +0.30 vs blueprint
    Spec : calling_station → RAISE_FORT 80%."""
    # Tracker fort : 80% CALL preflop → vpip = 0.80
    tracker = OpponentTracker()
    for _ in range(24):
        tracker.observer_action(0, action=2, contexte={"phase": "preflop"})
    for _ in range(6):
        tracker.observer_action(0, action=0, contexte={"phase": "preflop"})

    mixer = ExploitMixer(tracker)
    # Blueprint avec RAISE_FORT (idx 7) = 0.10
    bp    = np.array([0.20, 0.20, 0.20, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10])

    result = mixer.ajuster(bp, seat_index=0, game_type='NLHE_3MAX')

    # Directionnel
    assert result[3:].sum() > bp[3:].sum(), (
        f"calling_station : RAISE blueprint={bp[3:].sum():.3f}, "
        f"apres exploit={result[3:].sum():.3f} — attendu augmentation"
    )
    # Amplitude
    assert result[7] >= 0.50, (
        f"RAISE_FORT (idx 7) = {result[7]:.3f}, attendu >= 0.50"
    )
    assert result[7] > bp[7] + 0.30, (
        f"Augmentation RAISE_FORT insuffisante : {result[7]:.3f} vs {bp[7]:.3f} + 0.30"
    )


# ---------------------------------------------------------------------------
# Test 5 — fold_prone → RAISE augmente (directionnel + amplitude)
# ---------------------------------------------------------------------------

def test_exploit_fold_prone_cbet():
    """Adversaire fold_prone fort (fold_to_cbet=0.80, 30 mains) :
    - Directionnel : RAISE total augmente vs blueprint
    - Amplitude : RAISE total >= 0.50 et +0.30 vs blueprint
    Spec : fold_prone → RAISE 70%."""
    # Tracker fort : 80% FOLD sur cbet opp → fold_to_cbet = 0.80
    tracker = OpponentTracker()
    for _ in range(24):
        tracker.observer_action(0, action=0, contexte={"est_cbet_opp": True})
    for _ in range(6):
        tracker.observer_action(0, action=2, contexte={"est_cbet_opp": True})

    mixer = ExploitMixer(tracker)
    # Blueprint avec RAISE total (idx 3-8) = 0.20
    bp    = np.array([0.30, 0.30, 0.20, 0.04, 0.04, 0.04, 0.04, 0.04, 0.00])

    result = mixer.ajuster(bp, seat_index=0, game_type='NLHE_3MAX')

    # Directionnel
    assert result[3:].sum() > bp[3:].sum(), (
        f"fold_prone : RAISE blueprint={bp[3:].sum():.3f}, "
        f"apres exploit={result[3:].sum():.3f} — attendu augmentation"
    )
    # Amplitude
    assert result[3:].sum() >= 0.50, (
        f"RAISE total = {result[3:].sum():.3f}, attendu >= 0.50"
    )
    assert result[3:].sum() > bp[3:].sum() + 0.30, (
        f"Augmentation RAISE insuffisante : {result[3:].sum():.3f} vs {bp[3:].sum():.3f} + 0.30"
    )


# ---------------------------------------------------------------------------
# Test 6 — confiance 0 → blueprint retourne identique
# ---------------------------------------------------------------------------

def test_exploit_mixer_confiance_zero_retourne_blueprint():
    """Tracker vide (0 mains, confiance=0.0) :
    ajuster() doit retourner le blueprint strictement identique."""
    tracker = OpponentTracker()
    mixer   = ExploitMixer(tracker)
    bp      = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.05, 0.05, 0.0])

    result = mixer.ajuster(bp, seat_index=0, game_type='NLHE_3MAX')

    np.testing.assert_array_equal(result, bp,
        err_msg="confiance=0 : resultat doit etre identique au blueprint")


# ---------------------------------------------------------------------------
# Test 7 — profil neutre → blueprint identique (protection GTO)
# ---------------------------------------------------------------------------

def test_neutre_retourne_blueprint():
    """Stats GTO-like (zone neutre), confiance=1.0 :
    ajuster() doit retourner le blueprint identique.
    Protege TAG et regulier contre toute regression."""
    tracker = _tracker_neutre(seat=0, n=30)
    mixer   = ExploitMixer(tracker)
    bp      = _blueprint_equilibre()

    result = mixer.ajuster(bp, seat_index=0, game_type='NLHE_3MAX')

    np.testing.assert_array_equal(result, bp,
        err_msg="profil neutre : resultat doit etre identique au blueprint")


# ---------------------------------------------------------------------------
# Test 8 — bypass game_type Kuhn
# ---------------------------------------------------------------------------

def test_kuhn_bypass_oft():
    """game_type='KUHN' : bypass total OFT, retourne blueprint identique
    meme avec confiance=1.0 et profil calling_station."""
    tracker = _tracker_calling_station(seat=0, n=30)
    mixer   = ExploitMixer(tracker)
    bp      = np.array([0.3, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = mixer.ajuster(bp, seat_index=0, game_type='KUHN')

    np.testing.assert_array_equal(result, bp,
        err_msg="game_type=KUHN : bypass OFT — resultat doit etre identique au blueprint")


# ---------------------------------------------------------------------------
# Test 9 — integration agent : pipeline complet enregistrer_action → OFT → action
# ---------------------------------------------------------------------------

def test_agent_oft_modifie_strategie_apres_calibration():
    """Test d'integration : AgentAXIOM doit voir sa distribution changer
    apres 30 mains contre un adversaire calling_station identifie.

    Valide le pipeline complet :
      enregistrer_action() → OpponentTracker → ExploitMixer → distribution.

    Methodes requises (Etape 4) :
      agent.enregistrer_action(seat, action, street, vpip_action)
      agent.obtenir_distribution(etat, joueur, adversaire_actif)
      agent.tracker.vpip(seat), agent.tracker.confiance(seat)
    """
    from ai.agent import AgentAXIOM

    agent = AgentAXIOM(mode_deterministe=False)

    class _MockEtat:
        mise_courante = 0
        board         = []
        game_type     = 'NLHE_3MAX'

    class _MockJoueur:
        position  = 0
        mise_tour = 0
        cartes    = []

    etat   = _MockEtat()
    joueur = _MockJoueur()

    # Distribution avant calibration (adversaire siege 1 inconnu)
    dist_avant = agent.obtenir_distribution(etat, joueur, adversaire_actif=1)

    # Calibration : 30 CALL preflop de l'adversaire siege 1
    for _ in range(30):
        agent.enregistrer_action(
            seat=1,
            action='CALL',
            street='preflop',
            vpip_action=True,
        )

    # Verifie le profil detecte
    assert agent.tracker.vpip(1) >= 0.6, (
        f"calling_station non detecte : vpip={agent.tracker.vpip(1):.2f}"
    )
    assert agent.tracker.confiance(1) == 1.0, (
        f"confiance={agent.tracker.confiance(1):.2f}, attendu 1.0"
    )

    # Distribution apres calibration
    dist_apres = agent.obtenir_distribution(etat, joueur, adversaire_actif=1)

    assert not np.allclose(dist_avant, dist_apres, atol=0.01), (
        "Distribution inchangee malgre calibration calling_station (30 mains).\n"
        f"avant={np.round(dist_avant, 3)}\n"
        f"apres={np.round(dist_apres, 3)}"
    )


# ---------------------------------------------------------------------------
# Exp 04 bis — Test 10 : raise-only detecte comme hyper_agressif
# ---------------------------------------------------------------------------

def test_raise_only_detecte_comme_hyper_agressif():
    """Test 10 — raise-only doit etre classe hyper_agressif, pas calling_station.

    vpip = 1.0 (30 RAISE preflop : action=3 >= 2 donc compte comme vpip)
    pfr  = 1.0 (30 RAISE preflop : action=3 >= 3 donc compte comme pfr)
    profil attendu : 'hyper_agressif' (pas 'calling_station')
    """
    tracker = OpponentTracker()
    for _ in range(30):
        tracker.observer_action(0, action=3, contexte={"phase": "preflop"})

    assert tracker.vpip(0) == 1.0, f"vpip={tracker.vpip(0)}, attendu 1.0"
    assert tracker.pfr(0)  == 1.0, f"pfr={tracker.pfr(0)}, attendu 1.0"
    assert tracker.confiance(0) == 1.0

    mixer = ExploitMixer(tracker)
    profil = mixer._detecter_profil(0)
    assert profil == 'hyper_agressif', (
        f"profil={profil!r}, attendu 'hyper_agressif'\n"
        f"raise-only classifie a tort comme calling_station : bug PFR non corrige"
    )


# ---------------------------------------------------------------------------
# Exp 04 bis — Test 11 : calling_station requiert pfr bas
# ---------------------------------------------------------------------------

def test_calling_station_requiert_pfr_bas():
    """Test 11 — calling_station exige vpip eleve ET pfr bas.

    vpip = 0.8 (24 CALL + 6 FOLD = 80% entrees volontaires)
    pfr  = 0.0 (aucun raise preflop)
    profil attendu : 'calling_station'
    """
    tracker = OpponentTracker()
    for _ in range(24):
        tracker.observer_action(0, action=2, contexte={"phase": "preflop"})
    for _ in range(6):
        tracker.observer_action(0, action=0, contexte={"phase": "preflop"})

    assert tracker.vpip(0) == pytest.approx(24 / 30, abs=0.01)
    assert tracker.pfr(0)  == 0.0, f"pfr={tracker.pfr(0)}, attendu 0.0"

    mixer = ExploitMixer(tracker)
    profil = mixer._detecter_profil(0)
    assert profil == 'calling_station', (
        f"profil={profil!r}, attendu 'calling_station'\n"
        f"vpip={tracker.vpip(0):.2f} pfr={tracker.pfr(0):.2f}"
    )


# ---------------------------------------------------------------------------
# Exp 04 bis — Test 12 : haut vpip + haut pfr → hyper_agressif
# ---------------------------------------------------------------------------

def test_haut_vpip_haut_pfr_est_hyper_agressif_pas_calling_station():
    """Test 12 — vpip eleve + pfr eleve → hyper_agressif (pas calling_station).

    vpip = 0.8 (21 RAISE + 3 CALL = 24 entrees sur 30)
    pfr  = 0.7 (21 RAISE sur 30)
    profil attendu : 'hyper_agressif'
    """
    tracker = OpponentTracker()
    for _ in range(21):
        tracker.observer_action(0, action=3, contexte={"phase": "preflop"})
    for _ in range(3):
        tracker.observer_action(0, action=2, contexte={"phase": "preflop"})
    for _ in range(6):
        tracker.observer_action(0, action=0, contexte={"phase": "preflop"})

    assert tracker.vpip(0) == pytest.approx(24 / 30, abs=0.01)
    assert tracker.pfr(0)  == pytest.approx(21 / 30, abs=0.01)

    mixer = ExploitMixer(tracker)
    profil = mixer._detecter_profil(0)
    assert profil == 'hyper_agressif', (
        f"profil={profil!r}, attendu 'hyper_agressif'\n"
        f"vpip={tracker.vpip(0):.2f} pfr={tracker.pfr(0):.2f}\n"
        f"haut vpip + haut pfr doit etre hyper_agressif, pas calling_station"
    )


# ---------------------------------------------------------------------------
# Exp 04 bis — Test 13 : exploit hyper_agressif (CHECK trap)
# ---------------------------------------------------------------------------

def test_exploit_hyper_agressif_check_trap():
    """Test 13 — exploit hyper_agressif booste CHECK (trap) en priorite.

    Contre un joueur qui raise tout, on check/call pour controler le pot.
    Blueprint avec CHECK (idx 1) a 0.10.
    Apres exploit avec confiance=1.0 : result[1] >= 0.50 (CHECK majoritaire).
    """
    tracker = OpponentTracker()
    for _ in range(30):
        tracker.observer_action(0, action=3, contexte={"phase": "preflop"})

    assert tracker.vpip(0) == 1.0
    assert tracker.pfr(0)  == 1.0
    assert tracker.confiance(0) == 1.0

    bp = np.zeros(9)
    bp[0] = 0.10   # FOLD
    bp[1] = 0.10   # CHECK
    bp[2] = 0.40   # CALL
    bp[7] = 0.40   # RAISE_FORT

    mixer  = ExploitMixer(tracker)
    result = mixer.ajuster(bp, seat_index=0, game_type='NLHE_3MAX')

    assert result[1] >= 0.50, (
        f"CHECK (idx 1) = {result[1]:.3f}, attendu >= 0.50\n"
        f"Exploit hyper_agressif doit booster CHECK (check-trap)\n"
        f"distribution={np.round(result, 3)}"
    )
    assert abs(result.sum() - 1.0) < 1e-6, f"somme={result.sum():.6f}, attendu 1.0"
