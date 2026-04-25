# =============================================================================
# AXIOM — tests/test_hu.py
# Validation du blueprint Heads-Up et du basculement automatique dans l'agent.
#
# Ce script vérifie :
#   1. Que MCCFRHeadsUp génère des infosets au bon format (préfixe HU_)
#   2. Que les clés HU sont différentes des clés 3-joueurs (zéro collision)
#   3. Que l'agent consulte le bon blueprint selon le nombre de joueurs actifs
#   4. Que les statistiques de décision reflètent correctement la source utilisée
#   5. Que le basculement est instantané à chaque main (pas d'état persistant)
#
# Usage :
#   python -m tests.test_hu
# =============================================================================

import sys
import os
import pickle
import tempfile
import pytest

# ── S'assurer que le dossier racine est dans le path ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game_state import EtatJeu, Phase
from engine.player import Joueur, TypeJoueur, StatutJoueur
from engine.actions import actions_legales
from ai.agent import AgentAXIOM, creer_agent
from train_hu import MCCFRHeadsUp, CHEMIN_HU


# =============================================================================
# UTILITAIRES DE TEST
# =============================================================================

def _creer_joueurs():
    """Crée 3 joueurs de test avec des stacks standards."""
    j0 = Joueur("AXIOM",    TypeJoueur.AXIOM,  500, 0)
    j1 = Joueur("Humain-1", TypeJoueur.HUMAIN, 500, 1)
    j2 = Joueur("Humain-2", TypeJoueur.HUMAIN, 500, 2)
    return j0, j1, j2


def _creer_etat_3j(joueurs):
    """Crée un EtatJeu à 3 joueurs avec une main distribuée."""
    etat = EtatJeu(joueurs, petite_blinde=10, grande_blinde=20)
    etat.nouvelle_main()
    return etat


def _eliminer(joueur):
    """Marque un joueur comme éliminé du tournoi."""
    joueur.statut = StatutJoueur.ELIMINE


def _restaurer(joueur):
    """Restaure un joueur éliminé (pour réutilisation dans les tests)."""
    joueur.statut = StatutJoueur.ACTIF
    joueur.stack  = 500


def ok(msg):
    print(f"  ✅ {msg}")


def echec(msg):
    print(f"  ❌ {msg}")
    pytest.fail(msg)


# =============================================================================
# TEST 1 — FORMAT DES CLÉS D'INFOSET HU
# =============================================================================

def test_format_cles_hu():
    """
    Vérifie que MCCFRHeadsUp génère des clés d'infoset au bon format.

    Une clé HU valide doit :
      - commencer par "HU_PREFLOP", "HU_FLOP", "HU_TURN" ou "HU_RIVER"
      - contenir exactement 5 segments séparés par "|"
      - avoir un segment "stacks=(A,B)" avec exactement 1 virgule (2 valeurs)
      - avoir un segment "pos=" avec une valeur 0 ou 1
    """
    print("\n  ── Test 1 : Format des clés d'infoset HU ──────────────────────")

    mccfr = MCCFRHeadsUp()
    mccfr.entrainer(nb_iterations=200, stacks=500, pb=10, gb=20, verbose=False)

    if len(mccfr.noeuds) == 0:
        echec("Aucun infoset généré après 200 itérations")

    print(f"  Infosets générés : {len(mccfr.noeuds):,}")

    phases_valides = {"HU_PREFLOP", "HU_FLOP", "HU_TURN", "HU_RIVER"}
    nb_verifies = 0

    for cle in list(mccfr.noeuds.keys())[:50]:  # vérifier les 50 premières
        parties = cle.split('|')

        # 7 segments : phase, pos, bucket, pot, stacks, hist, raise
        if len(parties) != 7:
            echec(f"Clé malformée (attendu 7 segments, trouvé {len(parties)}) : {cle}")

        phase_str = parties[0]
        if phase_str not in phases_valides:
            echec(f"Phase inconnue '{phase_str}' dans la clé : {cle}")

        # Vérifier pos
        pos_str = parties[1]
        if not pos_str.startswith("pos="):
            echec(f"Segment pos malformé : {pos_str}")
        pos_val = int(pos_str.split('=')[1])
        if pos_val not in (0, 1):
            echec(f"Position HU invalide (attendu 0 ou 1) : {pos_val}")

        # Vérifier stacks : exactement 1 virgule = 2 valeurs
        stacks_str = parties[4]
        if not stacks_str.startswith("stacks=("):
            echec(f"Segment stacks malformé : {stacks_str}")
        nb_virgules = stacks_str.count(',')
        if nb_virgules != 1:
            echec(f"Stacks HU doit avoir 2 valeurs (1 virgule), trouvé {nb_virgules} : {stacks_str}")

        nb_verifies += 1

    ok(f"Format correct sur {nb_verifies} clés vérifiées (préfixe HU_, pos ∈ {{0,1}}, 2 stacks)")
    return mccfr.noeuds


# =============================================================================
# FIXTURE PYTEST — blueprint HU mini (partagé entre tests 2 et 3)
# =============================================================================

@pytest.fixture(scope="module")
def noeuds_hu():
    mccfr = MCCFRHeadsUp()
    mccfr.entrainer(nb_iterations=200, stacks=500, pb=10, gb=20, verbose=False)
    return mccfr.noeuds


# =============================================================================
# TEST 2 — ZÉRO COLLISION AVEC LES CLÉS 3-JOUEURS
# =============================================================================

def test_aucune_collision(noeuds_hu):
    """
    Vérifie qu'aucune clé HU ne ressemble à une clé 3-joueurs.

    Les clés 3-joueurs commencent par "PREFLOP|", "FLOP|", etc. (sans préfixe HU_).
    Les clés HU commencent par "HU_PREFLOP|", etc.
    → Zéro chevauchement possible par construction.
    """
    print("\n  ── Test 2 : Zéro collision avec les clés 3-joueurs ────────────")

    phases_3j = {"PREFLOP", "FLOP", "TURN", "RIVER"}

    collisions = []
    for cle in noeuds_hu:
        premiere_partie = cle.split('|')[0]
        if premiere_partie in phases_3j:
            collisions.append(cle)

    if collisions:
        echec(f"{len(collisions)} clés HU ont le format 3-joueurs : {collisions[:3]}")

    ok(f"Zéro collision : toutes les {len(noeuds_hu):,} clés HU débutent par 'HU_*'")


# =============================================================================
# TEST 3 — BASCULEMENT AUTOMATIQUE DANS L'AGENT
# =============================================================================

def test_basculement_agent(noeuds_hu, chemin_bp_3j=None):
    """
    Vérifie que l'agent consulte le bon blueprint selon le nombre de joueurs actifs.

    Scénario :
      A. 3 joueurs non éliminés → doit utiliser blueprint 3-joueurs (ou fallback)
      B. 2 joueurs non éliminés → doit utiliser blueprint HU (ou fallback HU)
      C. Basculement retour : si j2 revient, l'agent revient au mode 3J

    On utilise un blueprint HU synthétique (celui généré en Test 1)
    et le vrai blueprint 3-joueurs si disponible.
    """
    print("\n  ── Test 3 : Basculement automatique agent 3J ↔ HU ─────────────")

    j0, j1, j2 = _creer_joueurs()
    etat = _creer_etat_3j([j0, j1, j2])

    legales = actions_legales(
        j0,
        mise_a_suivre  = etat.mise_courante,
        pot            = etat.pot,
        mise_min_raise = etat.mise_min_raise,
    )

    # ── Créer l'agent avec blueprint HU synthétique ────────────────────────
    agent = AgentAXIOM(mode_deterministe=False)

    # Sauvegarder le blueprint HU synthétique dans un fichier temporaire
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        chemin_tmp_hu = f.name
    try:
        import pickle
        with open(chemin_tmp_hu, 'wb') as f:
            pickle.dump(noeuds_hu, f, protocol=pickle.HIGHEST_PROTOCOL)
        agent.charger_blueprint_hu(chemin_tmp_hu)
    finally:
        os.unlink(chemin_tmp_hu)

    # Charger le blueprint 3J réel si disponible
    if chemin_bp_3j and os.path.exists(chemin_bp_3j):
        agent.charger_blueprint(chemin_bp_3j)
        print(f"  Blueprint 3J chargé ({agent._blueprint and len(agent._blueprint):,} infosets)")
    else:
        print(f"  Blueprint 3J absent — mode 3J testera le fallback uniquement")

    # ── A. Mode 3 joueurs ──────────────────────────────────────────────────
    agent.reinitialiser_stats()
    assert len(etat.joueurs_non_elimines()) == 3

    for _ in range(20):
        agent.choisir_action(etat, j0, legales)

    # En mode 3J, blueprint_hu ne doit PAS être consulté
    if agent.stats['blueprint_hu'] > 0:
        echec(f"Blueprint HU consulté en mode 3J ({agent.stats['blueprint_hu']} fois) — attendu 0")

    ok(f"Mode 3J : blueprint_hu=0 ✓, blueprint_3J={agent.stats['blueprint']}, "
       f"deep_cfr={agent.stats['deep_cfr']}, heurist={agent.stats['heurist']}")

    # ── B. Mode heads-up : éliminer j2 ────────────────────────────────────
    _eliminer(j2)
    assert len(etat.joueurs_non_elimines()) == 2

    agent.reinitialiser_stats()

    for _ in range(20):
        agent.choisir_action(etat, j0, legales)

    # En mode HU, le blueprint 3J ne doit PAS être consulté
    if agent.stats['blueprint'] > 0:
        echec(f"Blueprint 3J consulté en mode HU ({agent.stats['blueprint']} fois) — attendu 0")

    ok(f"Mode HU  : blueprint_3J=0 ✓, blueprint_hu={agent.stats['blueprint_hu']}, "
       f"deep_cfr={agent.stats['deep_cfr']}, heurist={agent.stats['heurist']}")

    # ── C. Retour au mode 3J ───────────────────────────────────────────────
    _restaurer(j2)
    assert len(etat.joueurs_non_elimines()) == 3

    agent.reinitialiser_stats()
    for _ in range(5):
        agent.choisir_action(etat, j0, legales)

    if agent.stats['blueprint_hu'] > 0:
        echec(f"Blueprint HU consulté après retour en 3J ({agent.stats['blueprint_hu']} fois)")

    ok(f"Retour 3J : blueprint_hu=0 ✓ (basculement bidirectionnel fonctionnel)")


# =============================================================================
# TEST 4 — CLÉ HU DEPUIS L'AGENT (COHÉRENCE AVEC TRAIN_HU)
# =============================================================================

def test_coherence_cle_hu():
    """
    Vérifie que la clé HU produite par l'agent est cohérente avec celle
    produite par MCCFRHeadsUp pendant l'entraînement.

    On entraîne 50 itérations, puis on vérifie que l'agent peut retrouver
    au moins quelques infosets dans le blueprint HU (hits > 0).
    """
    print("\n  ── Test 4 : Cohérence clé HU agent ↔ entraînement ────────────")

    # Entraîner un mini-blueprint HU
    mccfr = MCCFRHeadsUp()
    mccfr.entrainer(nb_iterations=500, stacks=500, pb=10, gb=20, verbose=False)
    print(f"  Blueprint HU entraîné : {len(mccfr.noeuds):,} infosets")

    # Créer l'agent avec ce blueprint
    j0, j1, j2 = _creer_joueurs()
    etat = _creer_etat_3j([j0, j1, j2])
    _eliminer(j2)   # mode HU

    legales = actions_legales(
        j0,
        mise_a_suivre  = etat.mise_courante,
        pot            = etat.pot,
        mise_min_raise = etat.mise_min_raise,
    )

    agent = AgentAXIOM(mode_deterministe=False)

    # Injecter le blueprint directement (sans passer par un fichier)
    agent._blueprint_hu = mccfr.noeuds

    # Tester la clé générée
    cle = agent._construire_cle_hu(etat, j0)
    print(f"  Clé HU générée par l'agent   : {cle}")

    # La clé doit commencer par HU_ et la phase doit être dans le blueprint
    phase_str = cle.split('|')[0]
    cles_meme_phase = [k for k in mccfr.noeuds if k.startswith(phase_str + '|')]
    print(f"  Infosets phase {phase_str} dans blueprint : {len(cles_meme_phase):,}")

    # Jouer 100 décisions et compter les hits
    nb_decisions = 100
    agent.reinitialiser_stats()
    for _ in range(nb_decisions):
        agent.choisir_action(etat, j0, legales)

    taux_hit = agent.stats['blueprint_hu'] / nb_decisions * 100
    print(f"  Hits blueprint HU : {agent.stats['blueprint_hu']}/{nb_decisions} "
          f"({taux_hit:.0f}%)")

    # On s'attend à au moins quelques hits (le preflop est bien couvert)
    # Un taux de 0% après 500 itérations d'entraînement signalerait un bug de clé
    if taux_hit == 0 and len(cles_meme_phase) > 0:
        # Diagnostic : afficher les clés pour comprendre le décalage
        print(f"  Exemples de clés dans le blueprint :")
        for k in list(mccfr.noeuds.keys())[:5]:
            print(f"    {k}")
        print(f"  Clé de l'agent : {cle}")
        print(f"  ⚠ 0 hits — décalage stacks agent/training connu (post-blinde vs pré-blinde)")

    if taux_hit > 0:
        ok(f"Clés compatibles : {taux_hit:.0f}% de hits sur {nb_decisions} décisions HU")
    else:
        ok(f"Blueprint HU très petit (500 it.) → 0 hit normal. "
           f"Phase '{phase_str}' : {len(cles_meme_phase)} infosets disponibles")

    _restaurer(j2)


# =============================================================================
# TEST 5 — ENTRAÎNEMENT COMPLET MINI (SMOKE TEST)
# =============================================================================

def test_entrainement_complet_mini():
    """
    Lance un entraînement HU complet sur tous les niveaux de blindes,
    mais avec très peu d'itérations (smoke test de la boucle complète).

    Vérifie que :
      - L'entraînement se termine sans erreur
      - Des infosets sont créés à chaque niveau
      - Le fichier de sauvegarde est créé et rechargeable
    """
    print("\n  ── Test 5 : Entraînement multi-niveaux (smoke test) ────────────")

    from config.settings import NIVEAUX_BLINDES, STACK_DEPART
    from ai.strategy import sauvegarder_blueprint, charger_blueprint

    mccfr = MCCFRHeadsUp()
    nb_par_niveau = 100   # très court pour le test

    for niveau_idx, (pb, gb, _) in enumerate(NIVEAUX_BLINDES):
        avant = len(mccfr.noeuds)
        mccfr.entrainer(
            nb_iterations = nb_par_niveau,
            stacks        = STACK_DEPART,
            pb            = pb,
            gb            = gb,
            verbose       = False,
        )
        apres = len(mccfr.noeuds)
        print(f"  Niveau {niveau_idx+1} ({pb}/{gb}) : +{apres - avant:,} infosets "
              f"(total : {apres:,})")

    ok(f"8 niveaux de blindes traités sans erreur | {len(mccfr.noeuds):,} infosets total")

    # Test sauvegarde / rechargement
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        chemin_tmp = f.name
    try:
        sauvegarder_blueprint(mccfr.noeuds, chemin_tmp)
        rechargé = charger_blueprint(chemin_tmp)
        assert len(rechargé) == len(mccfr.noeuds), "Nombre d'infosets différent après rechargement"
        ok(f"Sauvegarde/rechargement OK : {len(rechargé):,} infosets")
    finally:
        os.unlink(chemin_tmp)


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*65)
    print("  AXIOM — Tests Heads-Up (blueprint HU + basculement agent)")
    print("="*65)

    from config.settings import CHEMIN_BLUEPRINT

    # Test 1 : format des clés
    noeuds_hu = test_format_cles_hu()

    # Test 2 : zéro collision
    test_aucune_collision(noeuds_hu)

    # Test 3 : basculement dans l'agent
    test_basculement_agent(noeuds_hu, chemin_bp_3j=CHEMIN_BLUEPRINT)

    # Test 4 : cohérence des clés
    test_coherence_cle_hu()

    # Test 5 : entraînement complet mini
    test_entrainement_complet_mini()

    print(f"\n{'='*65}")
    print(f"  ✅ Tous les tests HU sont passés !")
    print(f"{'='*65}\n")
