# =============================================================================
# AXIOM — tests/test_mccfr.py
# Tests de l'algorithme CFR — Phase 3a (Kuhn Poker)
#
# Lance ce fichier avec : python tests/test_mccfr.py
# Tous les tests doivent afficher OK pour valider la Phase 3a.
#
# Ces tests vérifient :
# 1. La logique de jeu du Kuhn Poker (nœuds terminaux, gains)
# 2. La convergence de CFR vers la solution analytique (-1/18)
# 3. Que la stratégie apprise respecte les propriétés du Nash Equilibrium
# 4. Que l'exploitabilité diminue avec les itérations
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.mccfr import (
    NoeudCFR, CFRKUHN,
    # Fonctions internes Kuhn Poker (préfixées _kp_) exposées pour les tests
    _kp_est_terminal   as est_terminal,
    _kp_joueur_actif   as joueur_actif,
    _kp_gain_terminal  as gain_terminal,
    _kp_cle_infoset    as cle_infoset_kuhn,
    _kp_tous_les_deals as tous_les_deals,
    _KP_ACTIONS        as ACTIONS,
    _KP_NB_ACTIONS     as NB_ACTIONS,
    _KP_VALEUR_NASH    as VALEUR_NASH,
)


# =============================================================================
# TESTS DU MOTEUR KUHN POKER
# =============================================================================

def test_noeuds_terminaux():
    print("TEST 1 : Kuhn Poker — nœuds terminaux...", end=" ")

    # Toutes les séquences terminales
    terminales = ['pp', 'bp', 'bb', 'pbp', 'pbb']
    for seq in terminales:
        assert est_terminal(seq), f"'{seq}' devrait être terminal"

    # Séquences NON terminales
    non_terminales = ['', 'p', 'b', 'pb']
    for seq in non_terminales:
        assert not est_terminal(seq), f"'{seq}' ne devrait pas être terminal"

    print("OK")


def test_joueur_actif():
    print("TEST 2 : Kuhn Poker — joueur actif...", end=" ")

    assert joueur_actif('')    == 0, "Au début, c'est J0 qui joue"
    assert joueur_actif('p')   == 1, "Après 1 action, c'est J1 qui joue"
    assert joueur_actif('b')   == 1, "Après 1 action, c'est J1 qui joue"
    assert joueur_actif('pb')  == 0, "Après 2 actions, c'est J0 qui joue"
    assert joueur_actif('pp')  == 0, "Même si terminal, la formule est cohérente"

    print("OK")


def test_gains_terminaux():
    print("TEST 3 : Kuhn Poker — gains aux nœuds terminaux...", end=" ")

    # J0 a K (=2), J1 a J (=0) → J0 gagne au showdown
    assert gain_terminal(2, 0, 'pp')  ==  1.0, "K vs J, showdown 1-1 : +1 pour J0"
    assert gain_terminal(2, 0, 'bb')  ==  2.0, "K vs J, showdown 2-2 : +2 pour J0"
    assert gain_terminal(2, 0, 'pbb') ==  2.0, "K vs J, showdown 2-2 : +2 pour J0"

    # J0 a J (=0), J1 a K (=2) → J1 gagne au showdown
    assert gain_terminal(0, 2, 'pp')  == -1.0, "J vs K, showdown 1-1 : -1 pour J0"
    assert gain_terminal(0, 2, 'bb')  == -2.0, "J vs K, showdown 2-2 : -2 pour J0"
    assert gain_terminal(0, 2, 'pbb') == -2.0, "J vs K, showdown 2-2 : -2 pour J0"

    # Fold — indépendant des cartes
    assert gain_terminal(0, 2, 'bp')  ==  1.0, "J0 mise, J1 fold : +1 pour J0"
    assert gain_terminal(2, 0, 'pbp') == -1.0, "J0 passe, J1 mise, J0 fold : -1 pour J0"

    print("OK")


def test_cle_infoset():
    print("TEST 4 : Kuhn Poker — clés d'information set...", end=" ")

    assert cle_infoset_kuhn(0, '')    == 'J',   "J0 a un J, début de main"
    assert cle_infoset_kuhn(1, 'p')   == 'Qp',  "J1 a une Q, J0 a passé"
    assert cle_infoset_kuhn(2, 'pb')  == 'Kpb', "J0 a un K, séquence pb"
    assert cle_infoset_kuhn(0, 'b')   == 'Jb',  "J1 a un J, J0 a misé"

    print("OK")


def test_tous_les_deals():
    print("TEST 5 : Kuhn Poker — deals possibles...", end=" ")

    deals = tous_les_deals()
    assert len(deals) == 6, f"6 deals attendus, obtenu {len(deals)}"

    # Tous les deals doivent être différents et valides
    for c0, c1 in deals:
        assert c0 != c1, f"Un joueur ne peut pas avoir deux fois la même carte"
        assert c0 in [0, 1, 2] and c1 in [0, 1, 2]

    # Vérifier qu'on couvre toutes les combinaisons
    paires = set(deals)
    assert len(paires) == 6, "Les 6 deals doivent être uniques"

    print("OK")


# =============================================================================
# TESTS DU NŒUD CFR
# =============================================================================

def test_noeud_cfr_uniforme():
    print("TEST 6 : NoeudCFR — stratégie uniforme si aucun regret...", end=" ")

    noeud = NoeudCFR(nb_actions=2)

    # Sans regrets → stratégie uniforme
    strat = noeud.strategie_courante(proba_reach=1.0)
    assert abs(strat[0] - 0.5) < 1e-9, f"Prob passer attendue = 0.5, obtenu {strat[0]}"
    assert abs(strat[1] - 0.5) < 1e-9, f"Prob miser attendue = 0.5, obtenu {strat[1]}"

    strat_moy = noeud.strategie_moyenne()
    assert abs(strat_moy[0] - 0.5) < 1e-9
    assert abs(strat_moy[1] - 0.5) < 1e-9

    print("OK")


def test_noeud_cfr_regret_matching():
    print("TEST 7 : NoeudCFR — regret matching correct...", end=" ")

    noeud = NoeudCFR(nb_actions=2)
    # Forcer des regrets : regret passer = 0, regret miser = 3
    # → stratégie courante doit miser à 100%
    noeud.regrets_cumules = [0.0, 3.0]

    strat = noeud.strategie_courante(proba_reach=1.0)
    assert abs(strat[0] - 0.0) < 1e-9, f"Passer doit être 0, obtenu {strat[0]}"
    assert abs(strat[1] - 1.0) < 1e-9, f"Miser doit être 1, obtenu {strat[1]}"

    # Regrets : regret passer = 1, regret miser = 3 → 25%/75%
    noeud2 = NoeudCFR(nb_actions=2)
    noeud2.regrets_cumules = [1.0, 3.0]
    strat2 = noeud2.strategie_courante(proba_reach=1.0)
    assert abs(strat2[0] - 0.25) < 1e-9, f"Passer doit être 0.25, obtenu {strat2[0]}"
    assert abs(strat2[1] - 0.75) < 1e-9, f"Miser doit être 0.75, obtenu {strat2[1]}"

    print("OK")


def test_noeud_cfr_strategie_somme():
    print("TEST 8 : NoeudCFR — accumulation stratégie somme...", end=" ")

    noeud = NoeudCFR(nb_actions=2)

    # Appeler 3 fois avec différentes probas de reach
    noeud.strategie_courante(proba_reach=1.0)
    noeud.strategie_courante(proba_reach=0.5)
    noeud.strategie_courante(proba_reach=0.5)

    # La stratégie moyenne doit être normalisée
    strat_moy = noeud.strategie_moyenne()
    assert abs(sum(strat_moy) - 1.0) < 1e-9, "La stratégie moyenne doit sommer à 1"
    assert noeud.nb_visites == 3, f"3 visites attendues, obtenu {noeud.nb_visites}"

    print("OK")


# =============================================================================
# TESTS DE CONVERGENCE CFR
# =============================================================================

def test_cfr_valeur_convergence():
    print("TEST 9 : CFR — convergence vers la valeur Nash (-1/18)...", end=" ")

    cfr = CFRKUHN()
    # 10 000 itérations : vanilla CFR sur Kuhn Poker converge en O(1/T),
    # garantit error < 0.003 (test tolérance = 0.005).
    valeur = cfr.entrainer(nb_iterations=10_000, verbose=False)

    # La valeur doit être proche de -1/18 ≈ -0.05556
    valeur_nash = VALEUR_NASH
    tolerance   = 0.005   # tolérance conservative (convergence typique : < 0.002)

    assert abs(valeur - valeur_nash) < tolerance, \
        (f"Convergence insuffisante :\n"
         f"  Valeur obtenue : {valeur:.6f}\n"
         f"  Valeur Nash    : {valeur_nash:.6f}\n"
         f"  Tolérance      : ±{tolerance}")

    print(f"OK  (valeur={valeur:.5f}, cible={valeur_nash:.5f})")


def test_cfr_noeuds_crees():
    print("TEST 10 : CFR — 12 nœuds d'information sets créés...", end=" ")

    cfr = CFRKUHN()
    cfr.entrainer(nb_iterations=100, verbose=False)

    # Le Kuhn Poker a exactement 12 information sets non-terminaux :
    # J0 (3 cartes) × {début, p, b, pb} = 12 infosets
    # En pratique : J, Q, K, Jp, Qp, Kp, Jb, Qb, Kb, Jpb, Qpb, Kpb
    nb_noeuds_attendus = 12
    assert len(cfr.noeuds) == nb_noeuds_attendus, \
        (f"Nombre de nœuds incorrect : attendu={nb_noeuds_attendus}, "
         f"obtenu={len(cfr.noeuds)}\n"
         f"  Nœuds créés : {sorted(cfr.noeuds.keys())}")

    print(f"OK  ({len(cfr.noeuds)} nœuds)")


def test_cfr_nash_q_never_bets():
    """
    Invariant Nash : le joueur 0 avec une Q ne mise JAMAIS.
    La Q est une main médiane — ni assez forte pour value bet, ni bluff utile.
    """
    print("TEST 11 : CFR — Nash : J0 avec Q ne mise jamais...", end=" ")

    cfr = CFRKUHN()
    cfr.entrainer(nb_iterations=5000, verbose=False)

    strat_Q = cfr.noeuds['Q'].strategie_moyenne()
    prob_bet_Q = strat_Q[1]  # probabilité de miser avec Q

    assert prob_bet_Q < 0.05, \
        (f"J0 avec Q doit rarement/jamais miser (Nash = 0)\n"
         f"  Prob miser obtenue : {prob_bet_Q:.4f} (tolérance: < 0.05)")

    print(f"OK  (prob bet Q = {prob_bet_Q:.4f}, attendu ≈ 0)")


def test_cfr_nash_k_bets_more_than_j():
    """
    Invariant Nash : J0 avec K doit miser PLUS SOUVENT que J0 avec J.
    K est la meilleure main (value bet), J est la pire (bluff occasionnel).
    Plus précisément : prob(K bette) ≈ 3 × prob(J bette).
    """
    print("TEST 12 : CFR — Nash : J0 mise plus avec K qu'avec J...", end=" ")

    cfr = CFRKUHN()
    cfr.entrainer(nb_iterations=5000, verbose=False)

    strat_J = cfr.noeuds['J'].strategie_moyenne()
    strat_K = cfr.noeuds['K'].strategie_moyenne()
    prob_bet_J = strat_J[1]
    prob_bet_K = strat_K[1]

    assert prob_bet_K > prob_bet_J, \
        (f"J0 doit miser plus souvent avec K qu'avec J\n"
         f"  Prob bet K = {prob_bet_K:.4f}\n"
         f"  Prob bet J = {prob_bet_J:.4f}")

    print(f"OK  (bet K={prob_bet_K:.3f} > bet J={prob_bet_J:.3f})")


def test_cfr_nash_j1_k_always_bets_after_check():
    """
    Invariant Nash : J1 avec K mise TOUJOURS après un check de J0.
    Un K est la main dominante → il faut maximiser la valeur.
    """
    print("TEST 13 : CFR — Nash : J1 avec K mise toujours après check...", end=" ")

    cfr = CFRKUHN()
    cfr.entrainer(nb_iterations=5000, verbose=False)

    strat_Kp = cfr.noeuds['Kp'].strategie_moyenne()  # J1 a K, J0 a passé
    prob_bet_Kp = strat_Kp[1]

    assert prob_bet_Kp > 0.90, \
        (f"J1 avec K après check doit presque toujours miser (Nash = 1.0)\n"
         f"  Prob miser = {prob_bet_Kp:.4f} (tolérance: > 0.90)")

    print(f"OK  (prob bet Kp = {prob_bet_Kp:.4f}, attendu ≈ 1.0)")


def test_cfr_nash_j1_j_bluffs_after_check():
    """
    Invariant Nash : J1 avec J doit miser avec probabilité ≈ 1/3 après un check de J0.

    Pourquoi J bluff ? Pour rendre J0 INDIFFÉRENT entre call et fold quand il a Q.
    Contrainte d'indifférence : β_K = 3 × β_J → β_J = 1/3 quand β_K = 1.

    C'est un résultat clé de la théorie des jeux appliquée au poker :
    J DOIT bluffer exactement 1/3 du temps pour que l'adversaire ne puisse pas
    exploiter la stratégie en foldant ou callant systématiquement.
    """
    print("TEST 14 : CFR — Nash : J1 bluff ≈ 1/3 après check de J0...", end=" ")

    cfr = CFRKUHN()
    cfr.entrainer(nb_iterations=5000, verbose=False)

    strat_Jp = cfr.noeuds['Jp'].strategie_moyenne()  # J1 a J, J0 a passé
    prob_bet_Jp = strat_Jp[1]

    # Nash = 1/3 ≈ 0.333 — tolérance large car convergence à 5000 iterations
    assert 0.15 < prob_bet_Jp < 0.50, \
        (f"J1 avec J doit bluffer ≈ 1/3 après check (Nash = 1/3 ≈ 0.333)\n"
         f"  Prob miser = {prob_bet_Jp:.4f} (tolérance: 0.15 < x < 0.50)")

    print(f"OK  (prob bet Jp = {prob_bet_Jp:.4f}, attendu ≈ 1/3)")


def test_cfr_nash_j1_k_always_calls_bet():
    """
    Invariant Nash : J1 avec K call TOUJOURS une mise de J0.
    Un K ne peut pas être battu au showdown → jamais folder.
    """
    print("TEST 15 : CFR — Nash : J1 avec K call toujours une mise...", end=" ")

    cfr = CFRKUHN()
    cfr.entrainer(nb_iterations=5000, verbose=False)

    strat_Kb = cfr.noeuds['Kb'].strategie_moyenne()  # J1 a K, J0 a misé
    prob_call_Kb = strat_Kb[1]  # 'b' = call ici

    assert prob_call_Kb > 0.90, \
        (f"J1 avec K après une mise doit presque toujours call (Nash = 1.0)\n"
         f"  Prob call = {prob_call_Kb:.4f} (tolérance: > 0.90)")

    print(f"OK  (prob call Kb = {prob_call_Kb:.4f}, attendu ≈ 1.0)")


def test_cfr_nash_j1_j_always_folds_to_bet():
    """
    Invariant Nash : J1 avec J fold TOUJOURS face à une mise de J0.
    Un J est battu par Q et K → toujours perdre si J0 a misé (J0 a Q ou K).
    """
    print("TEST 16 : CFR — Nash : J1 avec J fold toujours face à une mise...", end=" ")

    cfr = CFRKUHN()
    cfr.entrainer(nb_iterations=5000, verbose=False)

    strat_Jb = cfr.noeuds['Jb'].strategie_moyenne()  # J1 a J, J0 a misé
    prob_fold_Jb = strat_Jb[0]  # 'p' = fold ici

    assert prob_fold_Jb > 0.90, \
        (f"J1 avec J face à une mise doit presque toujours fold (Nash = 1.0)\n"
         f"  Prob fold = {prob_fold_Jb:.4f} (tolérance: > 0.90)")

    print(f"OK  (prob fold Jb = {prob_fold_Jb:.4f}, attendu ≈ 1.0)")


def test_cfr_exploitabilite_diminue():
    """
    Le NashConv doit DIMINUER avec les itérations.
    NashConv = gain_BR_J0 + gain_BR_J1.
    Pour un Nash parfait : NashConv = -1/18 + 1/18 = 0.
    """
    print("TEST 17 : CFR — NashConv diminue avec les itérations...", end=" ")

    cfr_peu = CFRKUHN()
    cfr_peu.entrainer(nb_iterations=100, verbose=False)
    nashconv_100 = cfr_peu.exploitabilite()

    cfr_plus = CFRKUHN()
    cfr_plus.entrainer(nb_iterations=5000, verbose=False)
    nashconv_5000 = cfr_plus.exploitabilite()

    assert nashconv_5000 < nashconv_100, \
        (f"Le NashConv doit diminuer avec plus d'itérations\n"
         f"  NashConv à 100 its  : {nashconv_100:.6f}\n"
         f"  NashConv à 5000 its : {nashconv_5000:.6f}")

    # Après 5000 itérations, NashConv doit être proche de 0 pour Kuhn Poker
    assert nashconv_5000 < 0.05, \
        f"NashConv trop élevé après 5000 itérations : {nashconv_5000:.6f}"

    print(f"OK  (NashConv 100={nashconv_100:.5f}, 5000={nashconv_5000:.5f})")


def test_cfr_reinitialisation():
    print("TEST 18 : CFR — réinitialisation efface les nœuds...", end=" ")

    cfr = CFRKUHN()
    cfr.entrainer(nb_iterations=100, verbose=False)
    assert len(cfr.noeuds) > 0

    cfr.reinitialiser()
    assert len(cfr.noeuds) == 0, "Après réinitialisation, les nœuds doivent être vides"

    print("OK")


# =============================================================================
# TEST COMPLET AVEC AFFICHAGE (démonstration visuelle)
# =============================================================================

def test_demo_affichage():
    """Test de démonstration — entraîne et affiche la stratégie complète."""
    print("TEST 19 : Démonstration — entraînement complet avec affichage...")

    cfr = CFRKUHN()
    valeur = cfr.entrainer(nb_iterations=10_000, verbose=True)
    cfr.afficher_strategie()

    exploit = cfr.exploitabilite()
    valeur_jeu = cfr.valeur_du_jeu()

    print(f"  Valeur du jeu (stratégie moy) : {valeur_jeu:+.6f}")
    print(f"  Valeur Nash analytique        : {VALEUR_NASH:+.6f}")
    print(f"  Exploitabilité finale         : {exploit:.6f}")

    assert abs(valeur_jeu - VALEUR_NASH) < 0.003, \
        f"Convergence insuffisante : {valeur_jeu:.6f} vs {VALEUR_NASH:.6f}"

    print("OK")


# =============================================================================
# LANCEMENT
# =============================================================================



# =============================================================================
# TESTS PHASE 3b — EXTERNAL SAMPLING MCCFR TEXAS HOLD'EM
#
# Ces tests valident :
# 1. La structure de l'état de jeu léger (deal, blindes, buckets)
# 2. Les transitions d'état (actions, streets, file d'attente)
# 3. La logique MCCFR (infosets créés, stratégie évoluant avec les itérations)
# 4. Les gains terminaux (fold et showdown)
# 5. La sauvegarde/chargement du blueprint (ai/strategy.py)
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.mccfr import MCCFRHoldEm, _H_ACTIF, _H_FOLD, _H_ALLIN
from ai.mccfr import _H_PREFLOP, _H_FLOP, _H_TURN, _H_RIVER
from ai.mccfr import NoeudCFR
from engine.actions import TypeAction


# =============================================================================
# TESTS INSTANCIATION ET STRUCTURE
# =============================================================================

def test_20_instanciation():
    print("TEST 20 : MCCFRHoldEm — instanciation...", end=" ")
    mccfr = MCCFRHoldEm()
    assert hasattr(mccfr, 'noeuds'),     "doit avoir l'attribut 'noeuds'"
    assert hasattr(mccfr, 'iterations'), "doit avoir l'attribut 'iterations'"
    assert isinstance(mccfr.noeuds, dict)
    assert mccfr.iterations == 0
    assert len(mccfr.noeuds) == 0
    print("OK")


def test_21_deal_structure():
    print("TEST 21 : MCCFRHoldEm — structure du deal...", end=" ")
    mccfr = MCCFRHoldEm()
    etat  = mccfr._dealer_aleatoire(stacks=1500, pb=10, gb=20)

    # 3 joueurs, 2 cartes chacun
    assert len(etat['cartes']) == 3
    for j in range(3):
        assert len(etat['cartes'][j]) == 2, f"joueur {j} doit avoir 2 cartes"

    # Board complet de 5 cartes
    assert len(etat['board_complet']) == 5

    # Toutes les cartes sont distinctes
    toutes = etat['cartes'][0] + etat['cartes'][1] + etat['cartes'][2] + etat['board_complet']
    assert len(toutes) == len(set(toutes)), "toutes les cartes doivent être uniques"

    # Phase initiale = preflop
    assert etat['phase'] == _H_PREFLOP

    # File d'attente preflop = [0, 1, 2] (BTN, SB, BB)
    assert etat['joueurs_en_attente'] == [0, 1, 2]
    print("OK")


def test_22_blindes_postees():
    print("TEST 22 : MCCFRHoldEm — blindes postées correctement...", end=" ")
    mccfr = MCCFRHoldEm()
    etat  = mccfr._dealer_aleatoire(stacks=1500, pb=10, gb=20)

    # Pot = SB + BB = 10 + 20 = 30
    assert etat['pot'] == 30, f"pot attendu=30, obtenu={etat['pot']}"

    # Stacks : BTN intact, SB = 1490, BB = 1480
    assert etat['stacks'][0] == 1500, "BTN ne poste pas de blinde"
    assert etat['stacks'][1] == 1490, f"SB doit avoir 1490, obtenu {etat['stacks'][1]}"
    assert etat['stacks'][2] == 1480, f"BB doit avoir 1480, obtenu {etat['stacks'][2]}"

    # Contributions : BTN=0, SB=10, BB=20
    assert etat['contributions'][0] == 0
    assert etat['contributions'][1] == 10
    assert etat['contributions'][2] == 20

    # Mise courante = grande blinde
    assert etat['mise_courante'] == 20
    print("OK")


def test_23_buckets_precomputes():
    print("TEST 23 : MCCFRHoldEm — buckets précalculés...", end=" ")
    mccfr = MCCFRHoldEm()
    etat  = mccfr._dealer_aleatoire(stacks=1500, pb=10, gb=20)

    buckets = etat['buckets']

    # 3 joueurs × 4 phases
    assert len(buckets) == 3, "3 joueurs"
    for j in range(3):
        assert len(buckets[j]) == 4, f"joueur {j} doit avoir 4 buckets (une par phase)"
        for phase in range(4):
            b = buckets[j][phase]
            assert 0 <= b <= 7, f"bucket hors plage : joueur={j}, phase={phase}, bucket={b}"
    print("OK")


# =============================================================================
# TESTS DES ACTIONS ET TRANSITIONS D'ÉTAT
# =============================================================================

def test_24_action_fold():
    print("TEST 24 : MCCFRHoldEm — action FOLD...", end=" ")
    from engine.actions import Action, TypeAction

    mccfr = MCCFRHoldEm()
    etat  = mccfr._dealer_aleatoire(stacks=1500, pb=10, gb=20)
    etat_copie = mccfr._copier_etat(etat)

    # Retirer BTN de la file avant d'appliquer (comme le fait _es_mccfr)
    etat_copie['joueurs_en_attente'] = etat_copie['joueurs_en_attente'][1:]

    action = Action(TypeAction.FOLD)
    mccfr._appliquer_action(etat_copie, joueur_idx=0, action=action)

    # BTN doit être en statut FOLD
    assert etat_copie['statuts'][0] == _H_FOLD, "BTN doit être fold"
    # Historique doit contenir 'f'
    assert 'f' in etat_copie['hist_phases'][_H_PREFLOP]
    # SB et BB toujours actifs
    assert etat_copie['statuts'][1] == _H_ACTIF
    assert etat_copie['statuts'][2] == _H_ACTIF
    print("OK")


def test_25_action_call():
    print("TEST 25 : MCCFRHoldEm — action CALL...", end=" ")
    from engine.actions import Action, TypeAction

    mccfr = MCCFRHoldEm()
    etat  = mccfr._dealer_aleatoire(stacks=1500, pb=10, gb=20)

    # BTN doit payer 20 (mise_courante=20, mises_tour[0]=0)
    a_payer = etat['mise_courante'] - etat['mises_tour'][0]
    assert a_payer == 20

    etat_copie = mccfr._copier_etat(etat)
    etat_copie['joueurs_en_attente'] = etat_copie['joueurs_en_attente'][1:]

    action = Action(TypeAction.CALL, montant=20)
    mccfr._appliquer_action(etat_copie, joueur_idx=0, action=action)

    # Stack BTN = 1500 - 20 = 1480
    assert etat_copie['stacks'][0] == 1480, f"stack BTN={etat_copie['stacks'][0]}, attendu 1480"
    # mises_tour BTN = 20 (= mise_courante)
    assert etat_copie['mises_tour'][0] == 20
    # Pot = 30 + 20 = 50
    assert etat_copie['pot'] == 50, f"pot={etat_copie['pot']}, attendu 50"
    # Contribution BTN = 20
    assert etat_copie['contributions'][0] == 20
    # Historique 'c'
    assert 'c' in etat_copie['hist_phases'][_H_PREFLOP]
    print("OK")


def test_26_raise_reinsere_adversaires():
    print("TEST 26 : MCCFRHoldEm — raise réinsère les adversaires dans la file...", end=" ")
    from engine.actions import Action, TypeAction

    mccfr = MCCFRHoldEm()
    etat  = mccfr._dealer_aleatoire(stacks=1500, pb=10, gb=20)

    # BTN raise à 60
    etat_copie = mccfr._copier_etat(etat)
    etat_copie['joueurs_en_attente'] = etat_copie['joueurs_en_attente'][1:]  # retiré BTN

    action = Action(TypeAction.RAISE, montant=60)
    mccfr._appliquer_action(etat_copie, joueur_idx=0, action=action)

    # Après raise par BTN(0), SB(1) et BB(2) doivent répondre
    file = etat_copie['joueurs_en_attente']
    assert 1 in file, "SB(1) doit être dans la file après le raise de BTN"
    assert 2 in file, "BB(2) doit être dans la file après le raise de BTN"
    assert 0 not in file, "BTN(0) ne doit pas être dans la file (vient de raiser)"
    print("OK")


def test_27_actions_abstraites_preflop():
    print("TEST 27 : MCCFRHoldEm — actions abstraites preflop...", end=" ")
    mccfr = MCCFRHoldEm()
    etat  = mccfr._dealer_aleatoire(stacks=1500, pb=10, gb=20)

    # BTN doit payer la grande blinde → pas de CHECK possible
    actions = mccfr._actions_abstraites(etat, joueur_idx=0)
    types   = [a.type for a in actions]

    assert TypeAction.FOLD   in types, "FOLD doit être disponible (BTN doit payer)"
    assert TypeAction.CALL   in types, "CALL doit être disponible"
    assert TypeAction.RAISE  in types, "RAISE doit être disponible"
    assert TypeAction.ALL_IN in types, "ALL_IN doit être disponible"
    assert TypeAction.CHECK  not in types, "CHECK ne doit pas être disponible (mise à suivre)"

    # Pas de doublons
    cles = [(a.type, a.montant) for a in actions]
    assert len(cles) == len(set(cles)), "pas de doublons dans les actions"

    # BB avec option : rien à payer → CHECK possible
    actions_bb = mccfr._actions_abstraites(etat, joueur_idx=2)
    types_bb   = [a.type for a in actions_bb]
    assert TypeAction.CHECK  in types_bb, "BB doit pouvoir checker (mise déjà égale)"
    assert TypeAction.FOLD   not in types_bb, "BB ne peut pas fold (rien à payer)"
    print("OK")


# =============================================================================
# TESTS D'ENTRAÎNEMENT MCCFR
# =============================================================================

def test_28_une_iteration():
    print("TEST 28 : MCCFRHoldEm — une itération sans erreur...", end=" ")
    mccfr = MCCFRHoldEm()
    mccfr.entrainer(nb_iterations=1, verbose=False)
    assert mccfr.iterations == 1
    assert len(mccfr.noeuds) > 0, "au moins un infoset doit être créé"
    print(f"OK  ({len(mccfr.noeuds)} infosets créés)")


def test_29_infosets_preflop():
    print("TEST 29 : MCCFRHoldEm — infosets PREFLOP créés avec bon format...", end=" ")
    mccfr = MCCFRHoldEm()
    mccfr.entrainer(nb_iterations=10, verbose=False)

    infosets_preflop = [k for k in mccfr.noeuds if k.startswith('PREFLOP|')]
    assert len(infosets_preflop) > 0, "des infosets PREFLOP doivent être créés"

    # Vérifier le format de la clé
    cle_exemple = infosets_preflop[0]
    parties = cle_exemple.split('|')
    assert len(parties) == 7, \
        f"la clé doit avoir 7 parties séparées par '|', obtenu {len(parties)}: {cle_exemple}"
    assert parties[0] == 'PREFLOP'
    assert parties[1].startswith('pos=')
    assert parties[2].startswith('bucket=')
    assert parties[3].startswith('pot=')
    assert parties[4].startswith('stacks=(')
    assert parties[5].startswith('hist=')
    assert parties[6].startswith('raise=')
    print(f"OK  ({len(infosets_preflop)} infosets PREFLOP)")


def test_30_infosets_multi_phases():
    print("TEST 30 : MCCFRHoldEm — infosets créés sur ≥2 phases après 50 it...", end=" ")
    mccfr = MCCFRHoldEm()
    mccfr.entrainer(nb_iterations=50, verbose=False)

    phases_presentes = set()
    for cle in mccfr.noeuds:
        for phase_nom in ['PREFLOP', 'FLOP', 'TURN', 'RIVER']:
            if cle.startswith(phase_nom + '|'):
                phases_presentes.add(phase_nom)
                break

    assert len(phases_presentes) >= 2, \
        f"au moins 2 phases attendues, obtenu : {phases_presentes}"
    print(f"OK  (phases présentes : {sorted(phases_presentes)})")


def test_31_strategie_evolue():
    print("TEST 31 : MCCFRHoldEm — stratégie évolue avec les itérations...", end=" ")
    import random
    random.seed(42)

    mccfr = MCCFRHoldEm()
    mccfr.entrainer(nb_iterations=5, verbose=False)
    nb_noeuds_5 = len(mccfr.noeuds)

    # Capturer les stratégies après 5 itérations
    strats_avant = {}
    for cle, noeud in list(mccfr.noeuds.items())[:10]:
        strats_avant[cle] = list(noeud.strategie_somme)

    mccfr.entrainer(nb_iterations=50, verbose=False)

    # Vérifier que les stratégies ont changé pour au moins un nœud
    nb_changes = 0
    for cle, strat_avant in strats_avant.items():
        if cle in mccfr.noeuds:
            strat_apres = mccfr.noeuds[cle].strategie_somme
            if strat_apres != strat_avant:
                nb_changes += 1

    assert nb_changes > 0, \
        "les stratégies doivent évoluer avec les itérations"
    print(f"OK  ({nb_changes} nœuds ont évolué)")


def test_32_nb_infosets_croit():
    print("TEST 32 : MCCFRHoldEm — nb infosets croît avec les itérations...", end=" ")
    mccfr = MCCFRHoldEm()

    mccfr.entrainer(nb_iterations=10, verbose=False)
    nb_10 = len(mccfr.noeuds)

    mccfr.entrainer(nb_iterations=50, verbose=False)
    nb_60 = len(mccfr.noeuds)

    assert nb_60 >= nb_10, \
        f"le nombre d'infosets doit croître : {nb_10} → {nb_60}"
    print(f"OK  ({nb_10} → {nb_60} infosets)")


# =============================================================================
# TESTS DES GAINS TERMINAUX
# =============================================================================

def test_33_gains_terminaux():
    print("TEST 33 : MCCFRHoldEm — gains terminaux cohérents...", end=" ")
    mccfr = MCCFRHoldEm()

    # Construire un état terminal : BTN et SB ont fold, BB gagne le pot
    etat = {
        'cartes'        : [[1, 2], [3, 4], [5, 6]],
        'board_complet' : [7, 8, 9, 10, 11],
        'board_visible' : [7, 8, 9, 10, 11],
        'stacks'        : [1500, 1490, 1480],
        'contributions' : [0, 10, 20],
        'mises_tour'    : [0, 10, 20],
        'mise_courante' : 20,
        'pot'           : 30,
        'statuts'       : [_H_FOLD, _H_FOLD, _H_ACTIF],   # BTN et SB fold
        'phase'         : _H_PREFLOP,
        'joueurs_en_attente': [],
        'hist_phases'   : ['ff', '', '', ''],
        'grande_blinde' : 20,
        'buckets'       : [[0]*4, [0]*4, [7]*4],
    }

    # BB (joueur 2) gagne le pot
    gain_bb  = mccfr._gain_fold(etat, joueur_traversant=2)
    gain_btn = mccfr._gain_fold(etat, joueur_traversant=0)
    gain_sb  = mccfr._gain_fold(etat, joueur_traversant=1)

    # BB : pot(30) - contribution(20) = +10
    assert gain_bb == 10.0, f"gain BB attendu=+10, obtenu={gain_bb}"
    # BTN : contribution(0) = 0
    assert gain_btn == 0.0, f"gain BTN attendu=0, obtenu={gain_btn}"
    # SB : -contribution(10) = -10
    assert gain_sb == -10.0, f"gain SB attendu=-10, obtenu={gain_sb}"
    print("OK")


def test_34_reinitialisation():
    print("TEST 34 : MCCFRHoldEm — réinitialisation...", end=" ")
    mccfr = MCCFRHoldEm()
    mccfr.entrainer(nb_iterations=10, verbose=False)
    assert len(mccfr.noeuds) > 0
    assert mccfr.iterations > 0

    mccfr.reinitialiser()
    assert len(mccfr.noeuds) == 0, "les nœuds doivent être vides après réinitialisation"
    assert mccfr.iterations == 0, "le compteur doit être remis à 0"
    print("OK")


# =============================================================================
# TESTS SAUVEGARDE / CHARGEMENT BLUEPRINT
# =============================================================================

def test_35_sauvegarde_chargement():
    print("TEST 35 : strategy.py — sauvegarde et chargement pickle...", end=" ")
    import tempfile
    import os
    from ai.strategy import sauvegarder_blueprint, charger_blueprint

    mccfr = MCCFRHoldEm()
    mccfr.entrainer(nb_iterations=20, verbose=False)
    nb_noeuds = len(mccfr.noeuds)

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        chemin_tmp = f.name

    try:
        sauvegarder_blueprint(mccfr.noeuds, chemin_tmp)
        assert os.path.exists(chemin_tmp), "le fichier pkl doit exister"
        assert os.path.getsize(chemin_tmp) > 0, "le fichier ne doit pas être vide"

        noeuds_charges = charger_blueprint(chemin_tmp)
        assert len(noeuds_charges) == nb_noeuds, \
            f"nombre d'infosets : sauvé={nb_noeuds}, chargé={len(noeuds_charges)}"

        # Vérifier que les clés et les stratégies sont identiques
        for cle in list(mccfr.noeuds.keys())[:5]:
            assert cle in noeuds_charges, f"clé absente : {cle}"
            strat_orig    = mccfr.noeuds[cle].strategie_moyenne()
            strat_chargee = noeuds_charges[cle].strategie_moyenne()
            for s_o, s_c in zip(strat_orig, strat_chargee):
                assert abs(s_o - s_c) < 1e-9, "stratégies différentes après rechargement"

    finally:
        os.unlink(chemin_tmp)

    print("OK")


def test_36_charger_fichier_absent():
    print("TEST 36 : strategy.py — FileNotFoundError si fichier absent...", end=" ")
    from ai.strategy import charger_blueprint

    erreur_levee = False
    try:
        charger_blueprint('chemin/inexistant/blueprint.pkl')
    except FileNotFoundError:
        erreur_levee = True

    assert erreur_levee, "FileNotFoundError attendue pour un fichier inexistant"
    print("OK")


def test_37_obtenir_strategie():
    print("TEST 37 : strategy.py — obtenir_strategie()...", end=" ")
    from ai.strategy import obtenir_strategie

    mccfr = MCCFRHoldEm()
    mccfr.entrainer(nb_iterations=10, verbose=False)

    # Clé connue : la stratégie doit être retournée et sommer à 1
    cle_connue = list(mccfr.noeuds.keys())[0]
    strat = obtenir_strategie(mccfr.noeuds, cle_connue)
    assert strat is not None, "stratégie ne doit pas être None pour une clé connue"
    assert abs(sum(strat) - 1.0) < 1e-9, \
        f"stratégie doit sommer à 1, obtenu {sum(strat)}"

    # Clé inconnue : doit retourner None
    strat_inconnue = obtenir_strategie(mccfr.noeuds, 'CLE_INEXISTANTE')
    assert strat_inconnue is None, "doit retourner None pour une clé inconnue"
    print("OK")


def test_38_afficher_stats_blueprint():
    print("TEST 38 : strategy.py — afficher_stats_blueprint() sans exception...", end=" ")
    from ai.strategy import afficher_stats_blueprint

    mccfr = MCCFRHoldEm()
    mccfr.entrainer(nb_iterations=20, verbose=False)

    # Ne doit pas lever d'exception
    try:
        afficher_stats_blueprint(mccfr.noeuds)
    except Exception as e:
        assert False, f"afficher_stats_blueprint a levé une exception : {e}"

    # Aussi tester avec blueprint vide
    try:
        afficher_stats_blueprint({})
    except Exception as e:
        assert False, f"afficher_stats_blueprint({{}}) a levé une exception : {e}"

    print("OK")


# =============================================================================
# TESTS DE FORMAT ET COHÉRENCE
# =============================================================================

def test_39_format_cles_infoset():
    print("TEST 39 : MCCFRHoldEm — format clés infoset compatible info_set.py...", end=" ")
    mccfr = MCCFRHoldEm()
    mccfr.entrainer(nb_iterations=30, verbose=False)

    for cle in list(mccfr.noeuds.keys())[:20]:
        parties = cle.split('|')

        # 7 parties
        assert len(parties) == 7, \
            f"clé invalide (doit avoir 7 parties) : '{cle}'"

        # Phase connue
        assert parties[0] in ['PREFLOP', 'FLOP', 'TURN', 'RIVER'], \
            f"phase invalide : '{parties[0]}'"

        # pos=X avec X ∈ {0, 1, 2}
        assert parties[1].startswith('pos='), f"format pos invalide : '{parties[1]}'"
        pos = int(parties[1].split('=')[1])
        assert pos in (0, 1, 2), f"pos hors plage : {pos}"

        # bucket=Y avec Y ∈ {0..7}
        assert parties[2].startswith('bucket='), f"format bucket invalide"
        bucket = int(parties[2].split('=')[1])
        assert 0 <= bucket <= 7, f"bucket hors plage : {bucket}"

        # pot=Z et stacks=(a,b,c)
        assert parties[3].startswith('pot='),    f"format pot invalide"
        assert parties[4].startswith('stacks=('), f"format stacks invalide"
        assert parties[4].endswith(')'),           f"stacks doit se terminer par ')'"

        # hist= (peut être vide)
        assert parties[5].startswith('hist='), f"format hist invalide"

        # raise= (bucket de raise fraction pour pseudo-harmonic mapping)
        assert parties[6].startswith('raise='), f"format raise invalide"

    print(f"OK  ({len(mccfr.noeuds)} clés vérifiées)")


# =============================================================================
# TEST DE DÉMONSTRATION FINAL
# =============================================================================

def test_40_demo_holdem():
    """
    Démonstration complète : 200 itérations d'entraînement avec affichage.
    Vérifie que les infosets sont créés pour toutes les phases.
    """
    print("TEST 40 : Démonstration — 200 itérations Hold'em avec affichage...")

    mccfr = MCCFRHoldEm()
    mccfr.entrainer(nb_iterations=200, verbose=True)
    mccfr.afficher_stats()

    # Vérifications de base après 200 itérations
    assert mccfr.iterations == 200
    assert len(mccfr.noeuds) > 0, "des infosets doivent être créés"

    # Vérifier les stratégies : chaque nœud doit avoir une stratégie valide
    for cle, noeud in list(mccfr.noeuds.items())[:10]:
        strat = noeud.strategie_moyenne()
        assert abs(sum(strat) - 1.0) < 1e-6, \
            f"stratégie invalide (somme≠1) pour {cle}"
        assert all(0.0 <= p <= 1.0 for p in strat), \
            f"probabilité hors [0,1] pour {cle}"

    print(f"OK  ({len(mccfr.noeuds):,} infosets après 200 itérations)")


# =============================================================================
# LANCEMENT DES TESTS PHASE 3b
# =============================================================================

def lancer_tests_3b():
    print("\n" + "="*60)
    print("  AXIOM — Tests Phase 3b : ES-MCCFR Texas Hold'em")
    print("="*60 + "\n")

    try:
        test_20_instanciation()
        test_21_deal_structure()
        test_22_blindes_postees()
        test_23_buckets_precomputes()
        test_24_action_fold()
        test_25_action_call()
        test_26_raise_reinsere_adversaires()
        test_27_actions_abstraites_preflop()
        test_28_une_iteration()
        test_29_infosets_preflop()
        test_30_infosets_multi_phases()
        test_31_strategie_evolue()
        test_32_nb_infosets_croit()
        test_33_gains_terminaux()
        test_34_reinitialisation()
        test_35_sauvegarde_chargement()
        test_36_charger_fichier_absent()
        test_37_obtenir_strategie()
        test_38_afficher_stats_blueprint()
        test_39_format_cles_infoset()
        test_40_demo_holdem()

        print("\n" + "="*60)
        print("  ✅ Tous les tests Phase 3b sont passés !")
        print("  External Sampling MCCFR Hold'em opérationnel.")
        print("="*60 + "\n")
        return True

    except AssertionError as e:
        print(f"\n❌ ÉCHEC : {e}")
        return False
    except Exception as e:
        print(f"\n💥 ERREUR : {e}")
        import traceback
        traceback.print_exc()
        return False



# =============================================================================
# LANCEMENT COMPLET — Phase 3a + Phase 3b
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AXIOM — Tests Phase 3a : CFR Kuhn Poker")
    print("="*60 + "\n")

    try:
        # ── Phase 3a : Kuhn Poker ─────────────────────────────────────
        test_noeuds_terminaux()
        test_joueur_actif()
        test_gains_terminaux()
        test_cle_infoset()
        test_tous_les_deals()
        test_noeud_cfr_uniforme()
        test_noeud_cfr_regret_matching()
        test_noeud_cfr_strategie_somme()
        test_cfr_valeur_convergence()
        test_cfr_noeuds_crees()
        test_cfr_nash_q_never_bets()
        test_cfr_nash_k_bets_more_than_j()
        test_cfr_nash_j1_k_always_bets_after_check()
        test_cfr_nash_j1_j_bluffs_after_check()
        test_cfr_nash_j1_k_always_calls_bet()
        test_cfr_nash_j1_j_always_folds_to_bet()
        test_cfr_exploitabilite_diminue()
        test_cfr_reinitialisation()
        test_demo_affichage()

        print("\n" + "="*60)
        print("  ✅ Phase 3a validée — 19/19 tests passés")
        print("="*60)

        # ── Phase 3b : Hold'em ────────────────────────────────────────
        ok_3b = lancer_tests_3b()
        if not ok_3b:
            sys.exit(1)

    except AssertionError as e:
        print(f"\n❌ ÉCHEC Phase 3a : {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 ERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)