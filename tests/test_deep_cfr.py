# =============================================================================
# AXIOM — tests/test_deep_cfr.py
# Tests de la Phase 4 : Deep CFR + GPU
#
# Lance ce fichier avec : python tests/test_deep_cfr.py
# Tous les tests doivent afficher OK pour valider la Phase 4.
#
# Tests couverts :
#   1. Détection du device (GPU/CPU)
#   2. Encodeur d'infoset → vecteur de features
#   3. Réseaux de neurones (forward pass, regret matching, softmax)
#   4. Reservoir buffers (insertion, sampling, propriété uniforme)
#   5. Entraîneurs (gradient descent, masque actions, pondération)
#   6. Traversée Deep CFR (alimentation des buffers)
#   7. Stratégie en production (obtenir_strategie)
#   8. Sauvegarde et chargement des réseaux
#   9. Cohérence conservation des jetons pendant la traversée
#  10. Smoke test entraînement complet (3 itérations)
# =============================================================================

import sys
import os
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.network import (
    ReseauRegret, ReseauStrategie, creer_reseaux,
    encoder_infoset, encoder_infosets_batch,
    DIM_INPUT, NB_ACTIONS_MAX, DEVICE,
    sauvegarder_reseau, charger_reseau,
    DIM_PHASE, DIM_POSITION, DIM_BUCKET, DIM_POT, DIM_STACKS, DIM_OOP, DIM_HIST,
)
from ai.reservoir import (
    ReservoirBufferRegret, ReservoirBufferStrategie,
    creer_buffers,
)
from ai.trainer import (
    EntraineurRegret, EntraineurStrategie, creer_entraineurs,
)
from ai.deep_cfr import DeepCFR


# =============================================================================
# TEST 1 — Détection du device
# =============================================================================

def test_device():
    print("TEST 1 : Détection du device...", end=" ")

    assert isinstance(DEVICE, torch.device), \
        f"DEVICE doit être un torch.device, obtenu : {type(DEVICE)}"

    # Le device doit être l'un des 3 types supportés
    assert DEVICE.type in ('cuda', 'mps', 'cpu'), \
        f"Device inconnu : {DEVICE.type}"

    # Un tenseur de test doit se déplacer sur le device sans erreur
    t = torch.zeros(4, 4).to(DEVICE)
    assert t.device.type == DEVICE.type

    print("OK")


# =============================================================================
# TEST 2 — Encodeur d'infoset
# =============================================================================

def test_encodeur():
    print("TEST 2 : Encodeur d'infoset...", end=" ")

    etat = {
        'phase'        : 1,                  # FLOP
        'buckets'      : [[3, 5, 2, 4],
                          [6, 7, 1, 3],
                          [2, 4, 6, 5]],
        'pot'          : 150,
        'grande_blinde': 20,
        'stacks'       : [1350, 1290, 1310],
        'hist_phases'  : ['xr', 'c', '', ''],
    }

    # Encoder pour chaque joueur
    for joueur_idx in range(3):
        vec = encoder_infoset(etat, joueur_idx)

        # Shape et dtype
        assert vec.shape == (DIM_INPUT,), \
            f"Shape attendu ({DIM_INPUT},), obtenu {vec.shape}"
        assert vec.dtype == np.float32, \
            f"dtype attendu float32, obtenu {vec.dtype}"

        # Offsets calculés dynamiquement depuis les constantes DIM_*
        _off_pos    = DIM_PHASE
        _off_bucket = DIM_PHASE + DIM_POSITION
        _off_pot    = DIM_PHASE + DIM_POSITION + DIM_BUCKET
        _off_stacks = _off_pot + DIM_POT
        _off_hist   = _off_stacks + DIM_STACKS + DIM_OOP

        # One-hot phase (FLOP = index 1)
        assert vec[0] == 0.0 and vec[1] == 1.0 and vec[2] == 0.0 and vec[3] == 0.0, \
            f"One-hot phase FLOP incorrect : {vec[0:4]}"

        # One-hot position
        assert vec[_off_pos + joueur_idx] == 1.0, \
            f"One-hot position J{joueur_idx} incorrect"
        for j in range(3):
            if j != joueur_idx:
                assert vec[_off_pos + j] == 0.0

        # Texture cartes (DIM_BUCKET=4 scalaires) : valeurs finies dans [0,1]
        # [equite_courante, nut_adv, draw_pot, equite_preflop]
        bucket_vec = vec[_off_bucket : _off_bucket + DIM_BUCKET]
        assert np.all(np.isfinite(bucket_vec)), \
            f"Texture cartes contient NaN/Inf : {bucket_vec}"
        assert all(0.0 <= v <= 1.0 for v in bucket_vec), \
            f"Texture cartes hors [0,1] : {bucket_vec}"

        # Pot et stacks positifs (log1p > 0 pour des valeurs > 0)
        assert vec[_off_pot] > 0.0, "Pot log-normalisé devrait être > 0"
        assert all(vec[_off_stacks : _off_stacks + 3] > 0.0), \
            "Stacks log-normalisés devraient être > 0"

        # Valeurs dans des plages raisonnables (pas de NaN, pas d'inf)
        assert np.all(np.isfinite(vec)), "Vecteur contient NaN ou Inf"

    # Test encodage batch
    paires = [(etat, i) for i in range(3)]
    batch_t = encoder_infosets_batch(paires)
    assert batch_t.shape == (3, DIM_INPUT), \
        f"Batch shape attendu (3, {DIM_INPUT}), obtenu {batch_t.shape}"
    assert batch_t.device.type == DEVICE.type

    # Cas limite : stacks à zéro
    etat_zero = {**etat, 'stacks': [0, 0, 0], 'pot': 0}
    vec_zero = encoder_infoset(etat_zero, 0)
    assert np.all(np.isfinite(vec_zero)), "Vecteur avec stacks=0 contient NaN/Inf"

    # Cas limite : PREFLOP sans historique
    etat_pf = {**etat, 'phase': 0, 'hist_phases': ['', '', '', '']}
    vec_pf = encoder_infoset(etat_pf, 0)
    assert vec_pf[0] == 1.0, "Phase PREFLOP (index 0) attendue"
    # Historique vide → toutes les dimensions hist à 0
    _off_hist_pf = DIM_PHASE + DIM_POSITION + DIM_BUCKET + DIM_POT + DIM_STACKS + DIM_OOP
    assert np.all(vec_pf[_off_hist_pf:] == 0.0), "Historique vide → dims hist doivent être 0"

    print("OK")


# =============================================================================
# TEST 3 — Réseaux de neurones
# =============================================================================

def test_reseaux():
    print("TEST 3 : Réseaux de neurones...", end=" ")

    r_nets, s_nets = creer_reseaux(DEVICE)
    assert len(r_nets) == 3 and len(s_nets) == 3

    batch_size = 32
    x = torch.randn(batch_size, DIM_INPUT).to(DEVICE)

    for i in range(3):
        # ReseauRegret — passe avant
        r_out = r_nets[i](x)
        assert r_out.shape == (batch_size, NB_ACTIONS_MAX), \
            f"ReseauRegret J{i} shape : {r_out.shape}"
        assert torch.all(torch.isfinite(r_out)), "ReseauRegret sortie NaN/Inf"
        # Sortie non bornée (pas de softmax)
        assert not torch.all(r_out >= 0), \
            "ReseauRegret devrait avoir des sorties négatives (pas de ReLU final)"

        # predire_strategie — regret matching
        strat = r_nets[i].predire_strategie(x)
        assert strat.shape == (batch_size, NB_ACTIONS_MAX)
        # Chaque ligne doit sommer à 1
        sommes = strat.sum(dim=-1)
        assert torch.all(torch.abs(sommes - 1.0) < 1e-5), \
            f"predire_strategie non normalisée : {sommes[:5]}"
        assert torch.all(strat >= 0.0), "Stratégie doit être ≥ 0"

        # ReseauStrategie — passe avant avec softmax
        s_out = s_nets[i](x)
        assert s_out.shape == (batch_size, NB_ACTIONS_MAX)
        sommes_s = s_out.sum(dim=-1)
        assert torch.all(torch.abs(sommes_s - 1.0) < 1e-5), \
            f"ReseauStrategie non normalisée : {sommes_s[:5]}"
        assert torch.all(s_out >= 0.0)

        # Nombre de paramètres raisonnable (> 0)
        assert r_nets[i].nb_parametres() > 0
        assert s_nets[i].nb_parametres() > 0

    # Cas extrême : tous les regrets négatifs → stratégie uniforme
    # On teste la logique de regret matching DIRECTEMENT (sans passer par le réseau,
    # car les biais et ReLU cachés peuvent produire des sorties positives même avec
    # des entrées très négatives — comportement correct du réseau, pas du test).
    regrets_negatifs = torch.full((1, NB_ACTIONS_MAX), -10.0, device=DEVICE)
    regrets_pos = torch.clamp(regrets_negatifs, min=0.0)
    somme        = regrets_pos.sum(dim=-1, keepdim=True)
    uniforme     = torch.full_like(regrets_pos, 1.0 / NB_ACTIONS_MAX)
    masque       = (somme > 1e-8).expand_as(regrets_pos)
    strat_unif   = torch.where(masque, regrets_pos / (somme + 1e-10), uniforme)
    esperance    = 1.0 / NB_ACTIONS_MAX
    assert torch.all(torch.abs(strat_unif - esperance) < 1e-4), \
        "Regret matching : regrets tous négatifs → stratégie uniforme attendue"

    # Vérifier aussi que predire_strategie retourne bien une distribution valide
    # (somme = 1, valeurs ≥ 0) quelle que soit l'entrée
    for x_test in [
        torch.full((1, DIM_INPUT),  100.0, device=DEVICE),
        torch.full((1, DIM_INPUT), -100.0, device=DEVICE),
        torch.zeros((1, DIM_INPUT),         device=DEVICE),
    ]:
        s = r_nets[0].predire_strategie(x_test)
        assert torch.all(s >= 0.0),                        "Stratégie doit être ≥ 0"
        assert torch.abs(s.sum() - 1.0) < 1e-5,           "Stratégie doit sommer à 1"
        assert torch.all(torch.isfinite(s)),               "Stratégie NaN/Inf"

    print("OK")


# =============================================================================
# TEST 4 — Reservoir Buffers
# =============================================================================

def test_reservoir():
    print("TEST 4 : Reservoir Buffers...", end=" ")

    # ── Buffer Regrets ──────────────────────────────────────────────────
    buf = ReservoirBufferRegret(taille_max=500)
    assert len(buf) == 0
    assert not buf.est_pret(128)

    # Remplissage partiel
    for i in range(300):
        vec     = np.random.randn(DIM_INPUT).astype(np.float32)
        regrets = np.random.randn(NB_ACTIONS_MAX).astype(np.float32)
        buf.ajouter(vec, regrets, NB_ACTIONS_MAX)

    assert len(buf) == 300
    assert buf.est_pret(128)
    assert buf.nb_total == 300

    # Remplissage au-delà de la capacité → reservoir sampling
    for i in range(1000):
        vec     = np.random.randn(DIM_INPUT).astype(np.float32)
        regrets = np.random.randn(NB_ACTIONS_MAX).astype(np.float32)
        buf.ajouter(vec, regrets, NB_ACTIONS_MAX)

    assert len(buf) == 500, f"Buffer plein attendu 500, obtenu {len(buf)}"
    assert buf.nb_total == 1300

    # Sampling
    batch = buf.echantillonner(64)
    assert batch is not None
    vecs_b, regrets_b, nb_b = batch
    assert vecs_b.shape    == (64, DIM_INPUT)
    assert regrets_b.shape == (64, NB_ACTIONS_MAX)
    assert len(nb_b) == 64

    # Pas assez d'échantillons
    buf_petit = ReservoirBufferRegret(taille_max=100)
    for i in range(50):
        buf_petit.ajouter(np.zeros(DIM_INPUT, np.float32),
                          np.zeros(NB_ACTIONS_MAX, np.float32), 3)
    assert buf_petit.echantillonner(128) is None

    # ── Buffer Stratégie ────────────────────────────────────────────────
    buf_s = ReservoirBufferStrategie(taille_max=200)
    for it in range(1, 601):
        vec   = np.random.randn(DIM_INPUT).astype(np.float32)
        strat = np.random.dirichlet(np.ones(NB_ACTIONS_MAX)).astype(np.float32)
        buf_s.ajouter(vec, strat, iteration=it)

    assert len(buf_s) == 200
    batch_s = buf_s.echantillonner(32)
    assert batch_s is not None
    vecs_s, strats_s, iters_s, nb_act_s = batch_s
    assert vecs_s.shape   == (32, DIM_INPUT)
    assert strats_s.shape == (32, NB_ACTIONS_MAX)
    # Stratégies normalisées
    sommes = strats_s.sum(axis=1)
    assert all(abs(s - 1.0) < 1e-5 for s in sommes), \
        f"Stratégies non normalisées : {sommes[:5]}"
    # Itérations > 0
    assert all(iters_s > 0), "Toutes les itérations doivent être > 0"
    assert len(nb_act_s) == 32

    # ── Réinitialisation ────────────────────────────────────────────────
    buf.reinitialiser()
    assert len(buf) == 0 and buf.nb_total == 0

    print("OK")


# =============================================================================
# TEST 5 — Entraîneurs
# =============================================================================

def test_entraineurs():
    print("TEST 5 : Entraîneurs PyTorch...", end=" ")

    r_nets, s_nets = creer_reseaux(DEVICE)
    e_regs, e_strats = creer_entraineurs(r_nets, s_nets)

    assert len(e_regs) == 3 and len(e_strats) == 3

    # Remplir les buffers
    r_bufs, s_bufs = creer_buffers(taille_max=10_000)
    N = 2_000
    for i in range(N):
        vec   = np.random.randn(DIM_INPUT).astype(np.float32)
        reg   = np.random.randn(NB_ACTIONS_MAX).astype(np.float32)
        strat = np.random.dirichlet(np.ones(NB_ACTIONS_MAX)).astype(np.float32)
        for j in range(3):
            r_bufs[j].ajouter(vec, reg, NB_ACTIONS_MAX)
            s_bufs[j].ajouter(vec, strat, iteration=i + 1)

    # Une epoch pour chaque joueur
    for j in range(3):
        sr = e_regs[j].entrainer_epoch(r_bufs[j], nb_batchs=5, batch_size=128)
        ss = e_strats[j].entrainer_epoch(s_bufs[j], nb_batchs=5, batch_size=128)

        assert not np.isnan(sr['perte_moy']), f"Perte regret J{j} NaN"
        assert not np.isnan(ss['perte_moy']), f"Perte stratégie J{j} NaN"
        assert sr['perte_moy'] >= 0, "Perte MSE doit être ≥ 0"
        assert ss['perte_moy'] >= 0

    # Vérifier que les poids changent
    params_avant = [p.clone() for p in r_nets[0].parameters()]
    e_regs[0].entrainer_epoch(r_bufs[0], nb_batchs=3, batch_size=128)
    params_apres = list(r_nets[0].parameters())
    nb_changes = sum(not torch.equal(a, b)
                     for a, b in zip(params_avant, params_apres))
    assert nb_changes > 0, "Les poids du réseau n'ont pas changé !"

    # Buffer insuffisant → retourner NaN proprement (pas de crash)
    r_buf_vide = ReservoirBufferRegret(taille_max=10_000)
    sr_vide = e_regs[0].entrainer_epoch(r_buf_vide, nb_batchs=5, batch_size=256)
    assert np.isnan(sr_vide['perte_moy']), \
        "Buffer vide → perte attendue NaN"

    print("OK")


# =============================================================================
# TEST 6 — Traversée Deep CFR (alimentation des buffers)
# =============================================================================

def test_traversee():
    print("TEST 6 : Traversée Deep CFR...", end=" ")

    dcfr = DeepCFR(taille_buffer=10_000)

    # Un deal aléatoire
    etat = dcfr._dealer_aleatoire(stacks=1500, pb=10, gb=20)

    # Vérifications de l'état de départ
    assert etat['pot'] == 30, f"Pot initial attendu 30, obtenu {etat['pot']}"
    assert etat['phase'] == 0   # PREFLOP
    assert len(etat['cartes']) == 3
    assert len(etat['board_complet']) == 5

    # 50 traversées → les buffers doivent se remplir
    for _ in range(50):
        etat_i = dcfr._dealer_aleatoire(stacks=1500, pb=10, gb=20)
        for joueur in range(3):
            etat_c = dcfr._copier_etat(etat_i)
            dcfr._traverser(etat_c, joueur, iteration=1)

    # Chaque buffer doit avoir reçu des échantillons
    for i in range(3):
        assert len(dcfr.buffers_regret[i])    > 0, \
            f"Buffer regret J{i} vide après traversées"
        assert len(dcfr.buffers_strategie[i]) > 0, \
            f"Buffer stratégie J{i} vide après traversées"

    # Les vecteurs d'infoset ne doivent pas contenir de NaN
    for i in range(3):
        n = len(dcfr.buffers_regret[i])
        if n > 0:
            vecs = dcfr.buffers_regret[i]._vecs[:n]
            assert np.all(np.isfinite(vecs)), \
                f"Vecteurs infoset J{i} contiennent NaN/Inf"

    print("OK")


# =============================================================================
# TEST 7 — Stratégie en production
# =============================================================================

def test_strategie_production():
    print("TEST 7 : Stratégie en production...", end=" ")

    dcfr = DeepCFR(taille_buffer=5_000)
    etat = dcfr._dealer_aleatoire(stacks=1500, pb=10, gb=20)

    for i in range(3):
        strat = dcfr.obtenir_strategie(etat, i)

        # Shape et dtype
        assert strat.shape == (NB_ACTIONS_MAX,), \
            f"J{i} shape stratégie : {strat.shape}"
        assert strat.dtype == np.float32 or strat.dtype == np.float64

        # Distribution normalisée
        assert abs(strat.sum() - 1.0) < 1e-5, \
            f"J{i} stratégie non normalisée : somme={strat.sum()}"

        # Toutes les probabilités ≥ 0
        assert np.all(strat >= 0.0), f"J{i} probabilités négatives"

        # Pas de NaN/Inf
        assert np.all(np.isfinite(strat)), f"J{i} stratégie NaN/Inf"

    # Tester sur plusieurs états différents (pas de crash)
    for _ in range(10):
        etat_rand = dcfr._dealer_aleatoire(stacks=1500, pb=10, gb=20)
        for i in range(3):
            s = dcfr.obtenir_strategie(etat_rand, i)
            assert abs(s.sum() - 1.0) < 1e-5

    print("OK")


# =============================================================================
# TEST 8 — Sauvegarde et chargement des réseaux
# =============================================================================

def test_sauvegarde_chargement():
    print("TEST 8 : Sauvegarde et chargement des réseaux...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        chemin_r = os.path.join(tmpdir, 'models', 'regret_net.pt')
        chemin_s = os.path.join(tmpdir, 'models', 'strategy_net.pt')
        chemin_l = os.path.join(tmpdir, 'logs', 'training_log.csv')

        dcfr = DeepCFR(
            taille_buffer    = 1_000,
            chemin_regret    = chemin_r,
            chemin_strategie = chemin_s,
            chemin_log       = chemin_l,
        )

        # Modifier légèrement un poids pour tester la sauvegarde
        with torch.no_grad():
            for net in dcfr.reseaux_regret:
                for p in net.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
                    break  # modifier juste le premier paramètre

        # Récupérer les poids avant sauvegarde
        poids_avant = [
            p.clone().cpu()
            for p in dcfr.reseaux_regret[0].parameters()
        ]

        # Sauvegarder
        dcfr.sauvegarder(verbose=False)

        # Vérifier que les fichiers existent
        for i in range(3):
            chemin_ri = chemin_r.replace('.pt', f'_j{i}.pt')
            chemin_si = chemin_s.replace('.pt', f'_j{i}.pt')
            assert os.path.exists(chemin_ri), f"Fichier manquant : {chemin_ri}"
            assert os.path.exists(chemin_si), f"Fichier manquant : {chemin_si}"

        # Créer une nouvelle instance et charger
        dcfr2 = DeepCFR(
            taille_buffer    = 1_000,
            chemin_regret    = chemin_r,
            chemin_strategie = chemin_s,
            chemin_log       = chemin_l,
        )
        dcfr2.charger(verbose=False)

        # Vérifier que les poids sont identiques après chargement
        poids_apres = [
            p.cpu() for p in dcfr2.reseaux_regret[0].parameters()
        ]
        for a, b in zip(poids_avant, poids_apres):
            assert torch.allclose(a, b), \
                "Poids différents après sauvegarde/chargement !"

        # Vérifier sauvegarder_reseau / charger_reseau unitaires
        reseau_test   = ReseauRegret().to(DEVICE)
        chemin_unit   = os.path.join(tmpdir, 'test_unit.pt')
        sauvegarder_reseau(reseau_test, chemin_unit)
        assert os.path.exists(chemin_unit)

        reseau_charge = ReseauRegret().to(DEVICE)
        charger_reseau(reseau_charge, chemin_unit, DEVICE)
        for p1, p2 in zip(reseau_test.parameters(),
                           reseau_charge.parameters()):
            assert torch.allclose(p1, p2), "Poids unitaires différents après chargement"

    print("OK")


# =============================================================================
# TEST 9 — Conservation des jetons
# =============================================================================

def test_conservation_jetons():
    print("TEST 9 : Conservation des jetons (traversée)...", end=" ")

    dcfr = DeepCFR(taille_buffer=5_000)

    stacks_init = 1500
    nb_joueurs  = 3
    total_init  = stacks_init * nb_joueurs

    violations = 0
    for _ in range(100):
        etat = dcfr._dealer_aleatoire(stacks=stacks_init, pb=10, gb=20)

        # Vérifier l'état initial
        total = sum(etat['stacks']) + etat['pot']
        if abs(total - total_init) > 1:
            violations += 1
            continue

        # Simuler quelques actions manuelles
        actions = dcfr._actions_abstraites(etat, 0)
        if actions:
            etat_c = dcfr._copier_etat(etat)
            etat_c['joueurs_en_attente'] = list(etat['joueurs_en_attente'][1:])
            dcfr._appliquer_action(etat_c, 0, actions[0])
            total_apres = sum(etat_c['stacks']) + etat_c['pot']
            if abs(total_apres - total_init) > 1:
                violations += 1

    assert violations == 0, \
        f"{violations}/100 états violent la conservation des jetons"

    print("OK")


# =============================================================================
# TEST 10 — Smoke test entraînement complet
# =============================================================================

def test_entrainement_complet():
    print("TEST 10 : Smoke test entraînement complet (3 itérations)...", end=" ")

    dcfr = DeepCFR(taille_buffer=20_000)

    dcfr.entrainer(
        nb_iterations = 3,
        nb_traversees = 30,
        nb_batchs     = 3,
        batch_size    = 64,
        verbose       = False,
        save_every    = 0,
    )

    assert dcfr.iteration == 3, \
        f"Itération attendue 3, obtenue {dcfr.iteration}"

    # Buffers non vides
    for i in range(3):
        assert len(dcfr.buffers_regret[i])    > 0, \
            f"Buffer regret J{i} vide après entraînement"
        assert len(dcfr.buffers_strategie[i]) > 0, \
            f"Buffer stratégie J{i} vide après entraînement"

    # Stratégie toujours valide après entraînement
    etat = dcfr._dealer_aleatoire(stacks=1500, pb=10, gb=20)
    for i in range(3):
        s = dcfr.obtenir_strategie(etat, i)
        assert abs(s.sum() - 1.0) < 1e-5, \
            f"Stratégie J{i} invalide après entraînement : somme={s.sum()}"
        assert np.all(np.isfinite(s)), \
            f"Stratégie J{i} contient NaN/Inf après entraînement"

    # Les pertes des entraîneurs doivent avoir un historique
    for i in range(3):
        assert len(dcfr.entraineurs_regret[i].historique_perte)    > 0
        assert len(dcfr.entraineurs_strategie[i].historique_perte) > 0

    print("OK")


# =============================================================================
# LANCEMENT
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  AXIOM — Tests Phase 4 : Deep CFR + GPU")
    print("="*60 + "\n")

    try:
        test_device()
        test_encodeur()
        test_reseaux()
        test_reservoir()
        test_entraineurs()
        test_traversee()
        test_strategie_production()
        test_sauvegarde_chargement()
        test_conservation_jetons()
        test_entrainement_complet()

        print("\n" + "="*60)
        print("  ✅ Tous les tests sont passés — Phase 4 validée !")
        print(f"  Device utilisé : {DEVICE}")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n❌ ÉCHEC : {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 ERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
