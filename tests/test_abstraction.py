# =============================================================================
# AXIOM — tests/test_abstraction.py
# Tests du module d'abstraction (Phase 2).
# Lance ce fichier avec : python tests/test_abstraction.py
# Tous les tests doivent afficher OK pour valider la Phase 2.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treys import Card
from engine.player import Joueur, TypeJoueur
from engine.actions import TypeAction
from engine.game_state import EtatJeu, Phase
from abstraction.action_abstraction import AbstractionAction, abstraction_action
from abstraction.card_abstraction import AbstractionCartes, abstraction_cartes, _cle_preflop_abstraite
from abstraction.info_set import InfoSet, construire_cle_infoset


# -----------------------------------------------------------------------------
# TESTS ACTION ABSTRACTION
# -----------------------------------------------------------------------------

def test_abstraction_action_check():
    print("TEST 1 : AbstractionAction — check possible si aucune mise...", end=" ")
    j = Joueur("Test", TypeJoueur.AXIOM, 1000, 0)
    j.mise_tour = 0

    actions = abstraction_action.actions_abstraites(
        joueur         = j,
        mise_a_suivre  = 0,
        pot            = 100,
        mise_min_raise = 20
    )
    types = [a.type for a in actions]
    assert TypeAction.CHECK in types,  "CHECK doit être possible si mise_a_suivre=0"
    assert TypeAction.FOLD  not in types, "FOLD ne doit pas être proposé si mise_a_suivre=0"
    assert TypeAction.ALL_IN in types, "ALL_IN doit toujours être possible"
    print("OK")


def test_abstraction_action_fold_call():
    print("TEST 2 : AbstractionAction — fold et call si mise à suivre...", end=" ")
    j = Joueur("Test", TypeJoueur.AXIOM, 1000, 0)
    j.mise_tour = 0

    actions = abstraction_action.actions_abstraites(
        joueur         = j,
        mise_a_suivre  = 100,
        pot            = 200,
        mise_min_raise = 100
    )
    types = [a.type for a in actions]
    assert TypeAction.FOLD in types,   "FOLD doit être possible si mise à suivre"
    assert TypeAction.CALL in types,   "CALL doit être possible"
    assert TypeAction.RAISE in types,  "RAISE doit être possible"
    assert TypeAction.ALL_IN in types, "ALL_IN doit être possible"
    print("OK")


def test_abstraction_action_raises_abstraits():
    print("TEST 3 : AbstractionAction — raises en fractions du pot...", end=" ")
    from config.settings import TAILLES_MISE

    j = Joueur("Test", TypeJoueur.AXIOM, 2000, 0)
    j.mise_tour = 0
    pot = 400

    actions = abstraction_action.actions_abstraites(
        joueur         = j,
        mise_a_suivre  = 0,
        pot            = pot,
        mise_min_raise = 20
    )
    raises = [a for a in actions if a.type == TypeAction.RAISE]

    # Il doit y avoir au plus len(TAILLES_MISE) raises
    assert len(raises) <= len(TAILLES_MISE), \
        f"Trop de raises : {len(raises)} > {len(TAILLES_MISE)}"

    # Chaque raise doit être > 0
    for r in raises:
        assert r.montant > 0, f"Montant de raise invalide : {r.montant}"

    print("OK")


def test_abstraction_action_nb_max():
    print("TEST 4 : AbstractionAction — nb_actions_max cohérent...", end=" ")
    from config.settings import TAILLES_MISE

    nb_attendu = 3 + len(TAILLES_MISE) + 1   # fold+check+call + raises + all-in
    assert abstraction_action.nb_actions_max() == nb_attendu, \
        f"nb_actions_max attendu={nb_attendu}, obtenu={abstraction_action.nb_actions_max()}"
    print("OK")


def test_abstraction_action_allin_pas_doublon():
    print("TEST 5 : AbstractionAction — pas de doublon all-in / raise...", end=" ")
    j = Joueur("Test", TypeJoueur.AXIOM, 100, 0)
    j.mise_tour = 0

    actions = abstraction_action.actions_abstraites(
        joueur         = j,
        mise_a_suivre  = 0,
        pot            = 400,
        mise_min_raise = 20
    )
    # Vérifier l'absence de doublons
    cles = [(a.type, a.montant) for a in actions]
    assert len(cles) == len(set(cles)), "Des actions en double ont été détectées"
    print("OK")


# -----------------------------------------------------------------------------
# TESTS CARD ABSTRACTION
# -----------------------------------------------------------------------------

def test_cle_preflop_paire():
    print("TEST 6 : CardAbstraction — clé preflop paire...", end=" ")
    cartes = [Card.new('As'), Card.new('Ah')]
    cle = _cle_preflop_abstraite(cartes)
    assert cle == "AA", f"Attendu 'AA', obtenu '{cle}'"
    print("OK")


def test_cle_preflop_suited():
    print("TEST 7 : CardAbstraction — clé preflop suited...", end=" ")
    cartes = [Card.new('Ks'), Card.new('Qs')]
    cle = _cle_preflop_abstraite(cartes)
    assert cle == "KQs", f"Attendu 'KQs', obtenu '{cle}'"
    print("OK")


def test_cle_preflop_offsuit():
    print("TEST 8 : CardAbstraction — clé preflop offsuit...", end=" ")
    cartes = [Card.new('As'), Card.new('Kh')]
    cle = _cle_preflop_abstraite(cartes)
    assert cle == "AKo", f"Attendu 'AKo', obtenu '{cle}'"
    print("OK")


def test_bucket_preflop_forte():
    print("TEST 9 : CardAbstraction — bucket preflop main forte...", end=" ")
    # AA doit être dans le bucket le plus élevé
    cartes = [Card.new('As'), Card.new('Ah')]
    bucket = abstraction_cartes.bucket_preflop(cartes)
    assert bucket == abstraction_cartes.nb_buckets_preflop - 1, \
        f"AA doit être dans le bucket max ({abstraction_cartes.nb_buckets_preflop - 1}), obtenu {bucket}"
    print("OK")


def test_bucket_preflop_faible():
    print("TEST 10 : CardAbstraction — bucket preflop main faible...", end=" ")
    # 32o doit être dans un bucket bas
    cartes = [Card.new('3s'), Card.new('2h')]
    bucket = abstraction_cartes.bucket_preflop(cartes)
    assert bucket <= 2, \
        f"32o doit être dans un bucket bas (≤2), obtenu {bucket}"
    print("OK")


def test_bucket_postflop_range():
    print("TEST 11 : CardAbstraction — bucket postflop dans la bonne plage...", end=" ")
    cartes = [Card.new('As'), Card.new('Kh')]
    board  = [Card.new('Qd'), Card.new('Jc'), Card.new('Ts')]   # quinte royale !
    bucket = abstraction_cartes.bucket_postflop(cartes, board)
    assert 0 <= bucket < abstraction_cartes.nb_buckets_postflop, \
        f"Bucket postflop hors plage : {bucket}"
    # Avec As-Kh sur Q-J-T on a une quinte → bucket élevé
    assert bucket >= abstraction_cartes.nb_buckets_postflop - 2, \
        f"Quinte sur le board : bucket attendu élevé, obtenu {bucket}"
    print("OK")


def test_bucket_methode_unifiee():
    print("TEST 12 : CardAbstraction — méthode bucket() unifiée...", end=" ")
    cartes = [Card.new('As'), Card.new('Ah')]

    # Sans board → preflop
    b_preflop  = abstraction_cartes.bucket(cartes, board=[])
    # Avec board → postflop
    board      = [Card.new('Kd'), Card.new('Qd'), Card.new('Jd')]
    b_postflop = abstraction_cartes.bucket(cartes, board=board)

    assert 0 <= b_preflop  < abstraction_cartes.nb_buckets_preflop,  "Bucket preflop hors plage"
    assert 0 <= b_postflop < abstraction_cartes.nb_buckets_postflop, "Bucket postflop hors plage"
    print("OK")


# -----------------------------------------------------------------------------
# TESTS INFO SET
# -----------------------------------------------------------------------------

def test_infoset_construction():
    print("TEST 13 : InfoSet — construction de la clé...", end=" ")
    j1 = Joueur("AXIOM-1", TypeJoueur.AXIOM, 1500, 0)
    j2 = Joueur("AXIOM-2", TypeJoueur.AXIOM, 1500, 1)
    j3 = Joueur("AXIOM-3", TypeJoueur.AXIOM, 1500, 2)

    etat = EtatJeu([j1, j2, j3], petite_blinde=10, grande_blinde=20)
    etat.nouvelle_main()

    infoset = InfoSet(etat, j1)

    # La clé doit contenir les éléments attendus
    assert "PREFLOP"  in infoset.cle, "La clé doit contenir la phase"
    assert "pos="     in infoset.cle, "La clé doit contenir la position"
    assert "bucket="  in infoset.cle, "La clé doit contenir le bucket"
    assert "pot="     in infoset.cle, "La clé doit contenir le pot"
    assert "stacks="  in infoset.cle, "La clé doit contenir les stacks"
    assert "hist="    in infoset.cle, "La clé doit contenir l'historique"
    print("OK")


def test_infoset_cles_differentes():
    print("TEST 14 : InfoSet — deux situations → deux clés différentes...", end=" ")
    j1 = Joueur("AXIOM-1", TypeJoueur.AXIOM, 1500, 0)
    j2 = Joueur("AXIOM-2", TypeJoueur.AXIOM, 1500, 1)
    j3 = Joueur("AXIOM-3", TypeJoueur.AXIOM, 1500, 2)

    # Situation 1 : preflop, pas d'action
    etat1 = EtatJeu([j1, j2, j3], petite_blinde=10, grande_blinde=20)
    etat1.nouvelle_main()
    cle1 = construire_cle_infoset(etat1, j1)

    # Situation 2 : passer au flop (board différent)
    j1b = Joueur("AXIOM-1", TypeJoueur.AXIOM, 1500, 0)
    j2b = Joueur("AXIOM-2", TypeJoueur.AXIOM, 1500, 1)
    j3b = Joueur("AXIOM-3", TypeJoueur.AXIOM, 1500, 2)
    etat2 = EtatJeu([j1b, j2b, j3b], petite_blinde=10, grande_blinde=20)
    etat2.nouvelle_main()
    etat2.passer_phase_suivante()   # → flop
    cle2 = construire_cle_infoset(etat2, j1b)

    assert cle1 != cle2, "Preflop et flop doivent avoir des clés différentes"
    print("OK")


def test_infoset_memes_situations_meme_cle():
    print("TEST 15 : InfoSet — même situation abstraite → même clé...", end=" ")
    # Deux joueurs avec des cartes différentes MAIS dans le même bucket
    # doivent avoir la même clé (c'est le but de l'abstraction !)

    # On force deux mains dans le même bucket preflop (bucket max = AA et KK)
    # AA et KK sont tous deux dans le bucket 7 (le plus haut)
    j1a = Joueur("J1", TypeJoueur.AXIOM, 1500, 0)
    j1a.cartes = [Card.new('As'), Card.new('Ah')]   # AA → bucket 7

    j1b = Joueur("J1", TypeJoueur.AXIOM, 1500, 0)
    j1b.cartes = [Card.new('Ks'), Card.new('Kh')]   # KK → bucket 7

    j2 = Joueur("J2", TypeJoueur.AXIOM, 1500, 1)
    j3 = Joueur("J3", TypeJoueur.AXIOM, 1500, 2)

    etat_a = EtatJeu([j1a, j2, j3], petite_blinde=10, grande_blinde=20)
    etat_a.nouvelle_main()
    etat_a.joueurs[0].cartes = [Card.new('As'), Card.new('Ah')]   # forcer les cartes

    etat_b = EtatJeu([j1b, j2, j3], petite_blinde=10, grande_blinde=20)
    etat_b.nouvelle_main()
    etat_b.joueurs[0].cartes = [Card.new('Ks'), Card.new('Kh')]

    cle_a = construire_cle_infoset(etat_a, etat_a.joueurs[0])
    cle_b = construire_cle_infoset(etat_b, etat_b.joueurs[0])

    bucket_a = abstraction_cartes.bucket_preflop([Card.new('As'), Card.new('Ah')])
    bucket_b = abstraction_cartes.bucket_preflop([Card.new('Ks'), Card.new('Kh')])

    if bucket_a == bucket_b:
        assert cle_a == cle_b, \
            f"AA et KK dans le même bucket → même clé attendue\n  cle_a={cle_a}\n  cle_b={cle_b}"
    else:
        # Si les buckets sont différents, les clés seront différentes — c'est correct
        assert cle_a != cle_b, "Buckets différents → clés différentes"
    print("OK")


def test_infoset_hashable():
    print("TEST 16 : InfoSet — utilisable comme clé de dictionnaire...", end=" ")
    j1 = Joueur("AXIOM-1", TypeJoueur.AXIOM, 1500, 0)
    j2 = Joueur("AXIOM-2", TypeJoueur.AXIOM, 1500, 1)
    j3 = Joueur("AXIOM-3", TypeJoueur.AXIOM, 1500, 2)

    etat = EtatJeu([j1, j2, j3], petite_blinde=10, grande_blinde=20)
    etat.nouvelle_main()

    infoset = InfoSet(etat, j1)

    # Simuler ce que MCCFR va faire : stocker des regrets dans un dict
    table_regrets = {}
    table_regrets[infoset.cle] = [0.0, 0.0, 0.5, 0.3, 0.2]

    assert infoset.cle in table_regrets, "La clé doit être utilisable dans un dict"
    assert table_regrets[infoset.cle][2] == 0.5
    print("OK")


# -----------------------------------------------------------------------------
# LANCEMENT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  AXIOM — Tests Phase 2 : Abstraction")
    print("="*50 + "\n")

    try:
        # Tests action abstraction
        test_abstraction_action_check()
        test_abstraction_action_fold_call()
        test_abstraction_action_raises_abstraits()
        test_abstraction_action_nb_max()
        test_abstraction_action_allin_pas_doublon()

        # Tests card abstraction
        test_cle_preflop_paire()
        test_cle_preflop_suited()
        test_cle_preflop_offsuit()
        test_bucket_preflop_forte()
        test_bucket_preflop_faible()
        test_bucket_postflop_range()
        test_bucket_methode_unifiee()

        # Tests info set
        test_infoset_construction()
        test_infoset_cles_differentes()
        test_infoset_memes_situations_meme_cle()
        test_infoset_hashable()

        print("\n" + "="*50)
        print("  ✅ Tous les tests sont passés — Phase 2 validée !")
        print("="*50 + "\n")

    except AssertionError as e:
        print(f"\n❌ ÉCHEC : {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 ERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)