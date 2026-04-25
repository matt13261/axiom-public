# =============================================================================
# AXIOM — tests/test_engine.py
# Tests du moteur de jeu.
# Lance ce fichier avec : python tests/test_engine.py
# Tous les tests doivent afficher OK pour valider la Phase 1.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.card import DeckAXIOM, carte_en_texte, cartes_en_texte
from engine.hand_evaluator import score_main, classe_main, determiner_gagnants, calculer_equite
from engine.player import Joueur, StatutJoueur, TypeJoueur
from engine.actions import Action, TypeAction, actions_legales
from engine.blind_structure import StructureBlinde
from engine.game_state import EtatJeu, Phase
from engine.game import creer_partie


def test_deck():
    print("TEST 1 : Deck...", end=" ")
    deck = DeckAXIOM()
    assert deck.nb_restantes() == 52, "Le deck doit avoir 52 cartes"
    cartes = deck.distribuer(5)
    assert len(cartes) == 5, "distribuer(5) doit retourner 5 cartes"
    assert deck.nb_restantes() == 47, "Il doit rester 47 cartes"
    deck.melanger()
    assert deck.nb_restantes() == 52, "Après mélange, retour à 52 cartes"
    print("OK")


def test_evaluation_main():
    print("TEST 2 : Évaluation des mains...", end=" ")
    from treys import Card
    # Main : 9 de pique + 8 de pique, Board : 7-6-5 de pique + 2h + 3d
    # = Quinte flush (9 haute, non royale)
    main  = [Card.new('9s'), Card.new('8s')]
    board = [Card.new('7s'), Card.new('6s'), Card.new('5s'), Card.new('2h'), Card.new('3d')]
    score = score_main(main, board)
    combi = classe_main(score)
    assert combi == "Straight Flush", f"Attendu Straight Flush, obtenu {combi}"
    print("OK")


def test_determiner_gagnants():
    print("TEST 3 : Détermination du gagnant...", end=" ")
    from treys import Card

    j1 = Joueur("Alice", TypeJoueur.HUMAIN, 1500, 0)
    j2 = Joueur("Bob",   TypeJoueur.HUMAIN, 1500, 1)

    # Alice : paire d'As
    j1.cartes = [Card.new('As'), Card.new('Ah')]
    # Bob : paire de 2
    j2.cartes = [Card.new('2s'), Card.new('2h')]

    board = [Card.new('Kd'), Card.new('Qd'), Card.new('Jd'), Card.new('9c'), Card.new('3s')]

    gagnants = determiner_gagnants([j1, j2], board)
    assert len(gagnants) == 1, "Un seul gagnant attendu"
    assert gagnants[0].nom == "Alice", f"Alice doit gagner, pas {gagnants[0].nom}"
    print("OK")


def test_joueur():
    print("TEST 4 : Joueur et mises...", end=" ")
    j = Joueur("Test", TypeJoueur.AXIOM, 1000, 0)
    assert j.stack == 1000
    j.miser(200)
    assert j.stack == 800
    assert j.mise_tour == 200
    j.miser(900)   # all-in forcé
    assert j.stack == 0
    assert j.statut == StatutJoueur.ALL_IN
    j.recevoir(500)
    assert j.stack == 500
    print("OK")


def test_actions_legales():
    print("TEST 5 : Actions légales...", end=" ")
    j = Joueur("Test", TypeJoueur.HUMAIN, 1000, 0)
    j.mise_tour = 0

    # Cas 1 : aucune mise (check possible)
    legales = actions_legales(j, mise_a_suivre=0, pot=100, mise_min_raise=20)
    types = [a.type for a in legales]
    assert TypeAction.CHECK in types, "Check doit être possible si mise=0"
    assert TypeAction.FOLD not in types, "Fold ne doit pas être proposé si mise=0"

    # Cas 2 : mise à suivre (fold et call possibles)
    legales2 = actions_legales(j, mise_a_suivre=100, pot=200, mise_min_raise=100)
    types2 = [a.type for a in legales2]
    assert TypeAction.FOLD in types2, "Fold doit être possible"
    assert TypeAction.CALL in types2, "Call doit être possible"
    print("OK")


def test_blindes():
    print("TEST 6 : Structure des blindes...", end=" ")
    b = StructureBlinde()
    assert b.petite_blinde == 10
    assert b.grande_blinde == 20
    assert b.niveau_actuel == 1

    # Avancer 6 mains → passage au niveau 2 (duree=6 mains par niveau)
    for _ in range(6):
        b.avancer_main()
    assert b.petite_blinde == 15, f"Niveau 2 : PB attendue=15, obtenu={b.petite_blinde}"
    assert b.grande_blinde == 30
    assert b.niveau_actuel == 2
    print("OK")


def test_etat_jeu():
    print("TEST 7 : État de jeu (nouvelle main)...", end=" ")
    j1 = Joueur("Alice", TypeJoueur.HUMAIN, 1500, 0)
    j2 = Joueur("AXIOM-1", TypeJoueur.AXIOM, 1500, 1)
    j3 = Joueur("AXIOM-2", TypeJoueur.AXIOM, 1500, 2)

    etat = EtatJeu([j1, j2, j3], petite_blinde=10, grande_blinde=20)
    etat.nouvelle_main()

    # Vérifier que chaque joueur a 2 cartes
    for j in [j1, j2, j3]:
        assert len(j.cartes) == 2, f"{j.nom} doit avoir 2 cartes"

    # Le pot doit contenir PB + BB = 30
    assert etat.pot == 30, f"Pot attendu=30, obtenu={etat.pot}"

    # On est en preflop
    assert etat.phase == Phase.PREFLOP
    print("OK")


def test_partie_complete():
    print("TEST 8 : Partie complète (3 mains)...", end=" ")
    # On crée une partie sans humain (3 bots) et on joue 3 mains manuellement
    jeu = creer_partie()

    for _ in range(3):
        actifs = jeu.etat.joueurs_non_elimines()
        if len(actifs) < 2:
            break
        jeu.jouer_une_main()

    # Vérifier que les stacks totaux sont cohérents (somme = stack total de départ)
    from config.settings import NB_JOUEURS, STACK_DEPART
    total = sum(j.stack for j in jeu.joueurs)
    assert total == NB_JOUEURS * STACK_DEPART, \
        f"Total jetons incohérent : {total} != {NB_JOUEURS * STACK_DEPART}"
    print("OK")


# -----------------------------------------------------------------------------
# LANCEMENT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  AXIOM — Tests Phase 1 : Moteur de jeu")
    print("="*50 + "\n")

    try:
        test_deck()
        test_evaluation_main()
        test_determiner_gagnants()
        test_joueur()
        test_actions_legales()
        test_blindes()
        test_etat_jeu()
        test_partie_complete()

        print("\n" + "="*50)
        print("  ✅ Tous les tests sont passés — Phase 1 validée !")
        print("="*50 + "\n")

    except AssertionError as e:
        print(f"\n❌ ÉCHEC : {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 ERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)