# =============================================================================
# AXIOM — tests/test_agent.py
# Tests unitaires de la Phase 5 : Agent + Solveurs + Self-Play + Evaluateur.
#
# Lance avec :
#   python -m pytest tests/test_agent.py -v
# ou directement :
#   python tests/test_agent.py
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import pytest

# Imports moteur
from engine.game_state import EtatJeu, Phase
from engine.player import Joueur, TypeJoueur, StatutJoueur
from engine.actions import Action, TypeAction, actions_legales

# Imports Phase 5
from ai.agent import AgentAXIOM, creer_agent
from ai.network import NB_ACTIONS_MAX, DIM_INPUT, encoder_infoset
from solver.depth_limited import SolveurProfondeurLimitee
from solver.subgame_solver import SolveurSousJeu
from training.self_play import (
    MoteurSelfPlay, simuler,
    AgentAleatoire, AgentCallOnly, AgentRaiseOnly,
    ResultatPartie,
)
from training.evaluator import Evaluateur, RapportEvaluation, benchmark_rapide
from config.settings import NB_JOUEURS, STACK_DEPART


# =============================================================================
# FIXTURES PARTAGÉES
# =============================================================================

def creer_etat_preflop():
    """Crée un EtatJeu initialisé en PREFLOP (fixture réutilisable)."""
    j0 = Joueur("AXIOM",    TypeJoueur.AXIOM,  1500, 0)
    j1 = Joueur("Humain-1", TypeJoueur.HUMAIN, 1500, 1)
    j2 = Joueur("Humain-2", TypeJoueur.HUMAIN, 1500, 2)
    etat = EtatJeu([j0, j1, j2], petite_blinde=10, grande_blinde=20)
    etat.nouvelle_main()
    return etat, j0, j1, j2


def creer_agent_heuristique():
    """Crée un AgentAXIOM sans blueprint ni Deep CFR (fallback heuristique)."""
    return AgentAXIOM(mode_deterministe=False)


def _action_semantiquement_legale(action, legales, etat, joueur) -> bool:
    """
    Vérifie la légalité sémantique (pas égalité stricte d'Action).

    La perturbation de sizing (Point 9) peut produire un montant de RAISE
    différent de ceux de la liste abstraite `legales` — mais le raise reste
    valide au sens des règles du poker si :
        mise_courante + mise_min_raise  ≤  montant  <  mise_tour + stack
    (en dessous du stack complet pour éviter de transformer en ALL_IN).

    Pour les autres types, on vérifie qu'une action de même type existe
    dans legales.
    """
    types_legaux = {a.type for a in legales}
    if action.type not in types_legaux:
        return False

    if action.type == TypeAction.RAISE:
        mini = etat.mise_courante + max(etat.mise_min_raise, 1)
        maxi = joueur.mise_tour + joueur.stack - 1
        return mini <= action.montant <= maxi

    # FOLD / CHECK / CALL / ALL_IN : comparaison d'Action dans legales
    return any(a.type == action.type and
               (a.montant == action.montant or a.type == TypeAction.FOLD
                or a.type == TypeAction.CHECK)
               for a in legales)


# =============================================================================
# TESTS — ai/agent.py
# =============================================================================

class TestAgentAXIOM:
    """Tests de la classe AgentAXIOM."""

    # ── Construction ──────────────────────────────────────────────────────

    def test_creation_agent_stochastique(self):
        """Un agent stochastique se crée sans erreur."""
        agent = AgentAXIOM(mode_deterministe=False)
        assert agent.mode_deterministe is False
        assert agent._blueprint is None
        assert agent._reseaux_strategie is None

    def test_creation_agent_deterministe(self):
        """Un agent déterministe se crée sans erreur."""
        agent = AgentAXIOM(mode_deterministe=True)
        assert agent.mode_deterministe is True

    def test_repr(self):
        """__repr__ retourne une chaîne valide."""
        agent = AgentAXIOM()
        s = repr(agent)
        assert "AgentAXIOM" in s
        assert "heuristique" in s   # aucune source chargée

    # ── choisir_action ────────────────────────────────────────────────────

    def test_choisir_action_retourne_action_legale(self):
        """choisir_action() retourne toujours une action sémantiquement légale.

        La perturbation de sizing (Point 9) peut décaler le montant d'un RAISE
        hors de la liste abstraite `legales`. On vérifie donc la validité
        sémantique : type légal + montant dans la plage poker légale.
        """
        etat, j0, j1, j2 = creer_etat_preflop()
        agent  = creer_agent_heuristique()
        legales = actions_legales(j0, etat.mise_courante, etat.pot, etat.mise_min_raise)
        action = agent.choisir_action(etat, j0, legales)
        assert _action_semantiquement_legale(action, legales, etat, j0)

    def test_choisir_action_mode_deterministe_reproductible(self):
        """En mode déterministe, deux appels consécutifs donnent la même action."""
        etat, j0, _, _ = creer_etat_preflop()
        agent  = AgentAXIOM(mode_deterministe=True)
        legales = actions_legales(j0, etat.mise_courante, etat.pot, etat.mise_min_raise)
        a1 = agent.choisir_action(etat, j0, legales)
        a2 = agent.choisir_action(etat, j0, legales)
        assert a1.type == a2.type

    def test_choisir_action_100_fois_toujours_legale(self):
        """Sur 100 appels stochastiques, l'action reste sémantiquement légale.

        Note : on ne compare pas par égalité d'Action (action in legales) car
        _perturber_sizing peut décaler le montant d'un RAISE (ex : 40 → 38).
        On vérifie que :
          - le type est dans les types légaux
          - si RAISE : montant dans la plage [mise_courante+min_raise, stack_max]
          - autres types : correspond à l'Action légale de même type.
        """
        etat, j0, _, _ = creer_etat_preflop()
        agent   = creer_agent_heuristique()
        legales = actions_legales(j0, etat.mise_courante, etat.pot, etat.mise_min_raise)
        for _ in range(100):
            action = agent.choisir_action(etat, j0, legales)
            assert _action_semantiquement_legale(action, legales, etat, j0), \
                f"Action illégale : {action} (légales={legales})"

    def test_choisir_action_incrémente_stats(self):
        """Chaque appel à choisir_action() incrémente stats['total']."""
        etat, j0, _, _ = creer_etat_preflop()
        agent   = creer_agent_heuristique()
        legales = actions_legales(j0, etat.mise_courante, etat.pot, etat.mise_min_raise)
        for i in range(5):
            agent.choisir_action(etat, j0, legales)
        assert agent.stats['total'] == 5

    # ── _convertir_etat ───────────────────────────────────────────────────

    def test_convertir_etat_format_correct(self):
        """_convertir_etat() produit un dict compatible encoder_infoset."""
        etat, j0, _, _ = creer_etat_preflop()
        agent      = creer_agent_heuristique()
        etat_dict  = agent._convertir_etat(etat, 0)

        assert 'phase'         in etat_dict
        assert 'buckets'       in etat_dict
        assert 'pot'           in etat_dict
        assert 'grande_blinde' in etat_dict
        assert 'stacks'        in etat_dict
        assert 'hist_phases'   in etat_dict

        assert isinstance(etat_dict['phase'],          int)
        assert isinstance(etat_dict['pot'],            int)
        assert isinstance(etat_dict['grande_blinde'],  int)
        assert len(etat_dict['stacks'])      == NB_JOUEURS
        assert len(etat_dict['hist_phases']) == 4

    def test_convertir_etat_compatible_encoder(self):
        """Le dict produit peut être passé à encoder_infoset sans erreur."""
        etat, j0, _, _ = creer_etat_preflop()
        agent     = creer_agent_heuristique()
        etat_dict = agent._convertir_etat(etat, 0)
        vec       = encoder_infoset(etat_dict, 0)
        assert vec.shape  == (DIM_INPUT,)
        assert vec.dtype  == np.float32
        assert not np.any(np.isnan(vec))

    # ── _obtenir_distribution ─────────────────────────────────────────────

    def test_distribution_forme_correcte(self):
        """_obtenir_distribution() retourne un vecteur de NB_ACTIONS_MAX probas."""
        etat, j0, _, _ = creer_etat_preflop()
        agent = creer_agent_heuristique()
        dist  = agent._obtenir_distribution(etat, j0)
        assert dist.shape == (NB_ACTIONS_MAX,)
        assert abs(dist.sum() - 1.0) < 1e-5

    def test_distribution_non_negative(self):
        """Toutes les probabilités sont ≥ 0."""
        etat, j0, _, _ = creer_etat_preflop()
        agent = creer_agent_heuristique()
        dist  = agent._obtenir_distribution(etat, j0)
        assert np.all(dist >= 0)

    # ── _mapper_sur_legales ───────────────────────────────────────────────

    def test_mapper_somme_un(self):
        """Le mapping sur les actions légales produit une somme = 1."""
        etat, j0, _, _ = creer_etat_preflop()
        agent   = creer_agent_heuristique()
        legales = actions_legales(j0, etat.mise_courante, etat.pot, etat.mise_min_raise)
        dist    = agent._obtenir_distribution(etat, j0)
        probas  = agent._mapper_sur_legales(dist, legales, etat)
        assert len(probas) == len(legales)
        assert abs(probas.sum() - 1.0) < 1e-5
        assert np.all(probas >= 0)

    # ── creer_agent ───────────────────────────────────────────────────────

    def test_creer_agent_fichiers_absents(self):
        """creer_agent() sans fichiers disponibles ne lève pas d'exception."""
        agent = creer_agent(
            chemin_blueprint = "inexistant.pkl",
            chemin_strategie = "inexistant.pt",
            verbose          = False,
        )
        assert isinstance(agent, AgentAXIOM)
        assert agent._blueprint is None
        assert agent._reseaux_strategie is None

    # ── reinitialiser_stats ───────────────────────────────────────────────

    def test_reinitialiser_stats(self):
        """reinitialiser_stats() remet tous les compteurs à 0."""
        etat, j0, _, _ = creer_etat_preflop()
        agent   = creer_agent_heuristique()
        legales = actions_legales(j0, etat.mise_courante, etat.pot, etat.mise_min_raise)
        for _ in range(3):
            agent.choisir_action(etat, j0, legales)
        agent.reinitialiser_stats()
        assert all(v == 0 for v in agent.stats.values())


# =============================================================================
# TESTS — solver/depth_limited.py
# =============================================================================

class TestSolveurProfondeurLimitee:
    """Tests du solveur real-time à profondeur limitée."""

    def test_creation(self):
        """Le solveur se crée sans erreur."""
        solveur = SolveurProfondeurLimitee(profondeur=1, nb_iterations=10, temps_max=2.0)
        assert solveur.profondeur     == 1
        assert solveur.nb_iterations  == 10
        assert solveur.temps_max      == 2.0

    def test_resoudre_retourne_vecteur_correct(self):
        """resoudre() retourne un np.ndarray de NB_ACTIONS_MAX probas."""
        etat, j0, _, _ = creer_etat_preflop()
        agent   = creer_agent_heuristique()
        solveur = SolveurProfondeurLimitee(profondeur=1, nb_iterations=10, temps_max=2.0)
        strat   = solveur.resoudre(etat, j0, agent)
        assert strat.shape == (NB_ACTIONS_MAX,)
        assert abs(strat.sum() - 1.0) < 1e-4
        assert np.all(strat >= 0)

    def test_resoudre_respecte_temps_max(self):
        """resoudre() s'arrête avant temps_max + 0.5s de marge."""
        etat, j0, _, _ = creer_etat_preflop()
        agent   = creer_agent_heuristique()
        solveur = SolveurProfondeurLimitee(profondeur=2, nb_iterations=1000, temps_max=1.0)
        debut   = time.time()
        solveur.resoudre(etat, j0, agent)
        duree   = time.time() - debut
        assert duree < 1.0 + 0.5   # tolérance 0.5s

    def test_resoudre_incremente_stats(self):
        """Après resoudre(), stats['iterations'] > 0."""
        etat, j0, _, _ = creer_etat_preflop()
        agent   = creer_agent_heuristique()
        solveur = SolveurProfondeurLimitee(profondeur=1, nb_iterations=5, temps_max=5.0)
        solveur.resoudre(etat, j0, agent)
        assert solveur.stats['iterations'] > 0

    def test_equite_rapide_entre_zero_et_un(self):
        """_equite_rapide() retourne une valeur dans [0, 1]."""
        etat, j0, _, _ = creer_etat_preflop()
        solveur   = SolveurProfondeurLimitee(nb_simul_equite=20)
        etat_dict = solveur._convertir_etat(etat, 0)
        equite    = solveur._equite_rapide(etat_dict, 0)
        assert 0.0 <= equite <= 1.0

    def test_convertir_etat_format_correct(self):
        """_convertir_etat() produit un dict avec toutes les clés requises."""
        etat, j0, _, _ = creer_etat_preflop()
        solveur   = SolveurProfondeurLimitee()
        etat_dict = solveur._convertir_etat(etat, 0)
        for cle in ('cartes', 'board_complet', 'board_visible', 'stacks',
                    'contributions', 'mises_tour', 'mise_courante', 'pot',
                    'statuts', 'phase', 'joueurs_en_attente',
                    'hist_phases', 'grande_blinde', 'buckets'):
            assert cle in etat_dict, f"Clé manquante : {cle}"

    def test_repr(self):
        """__repr__ retourne une chaîne valide."""
        solveur = SolveurProfondeurLimitee(profondeur=2)
        assert "SolveurProfondeurLimitee" in repr(solveur)


# =============================================================================
# TESTS — solver/subgame_solver.py
# =============================================================================

class TestSolveurSousJeu:
    """Tests du solveur de sous-jeu."""

    def test_creation(self):
        """Le solveur de sous-jeu se crée sans erreur."""
        solveur = SolveurSousJeu(nb_scenarios=5, nb_iterations=10)
        assert solveur.nb_scenarios   == 5
        assert solveur.nb_iterations  == 10

    def test_resoudre_retourne_vecteur_correct(self):
        """resoudre() retourne un np.ndarray normalisé de NB_ACTIONS_MAX."""
        etat, j0, _, _ = creer_etat_preflop()
        agent   = creer_agent_heuristique()
        solveur = SolveurSousJeu(
            nb_scenarios    = 3,
            nb_iterations   = 5,
            temps_max       = 5.0,
            profondeur      = 1,
            nb_simul_equite = 5,
        )
        strat = solveur.resoudre(etat, j0, agent)
        assert strat.shape == (NB_ACTIONS_MAX,)
        assert abs(strat.sum() - 1.0) < 1e-4
        assert np.all(strat >= 0)

    def test_tirer_scenarios_non_vide(self):
        """_tirer_scenarios() retourne au moins 1 scénario en PREFLOP."""
        etat, j0, _, _ = creer_etat_preflop()
        solveur   = SolveurSousJeu(nb_scenarios=5)
        scenarios = solveur._tirer_scenarios(etat, 0)
        assert len(scenarios) > 0

    def test_tirer_scenarios_cartes_distinctes(self):
        """Dans chaque scénario, les cartes adverses ne doublonnent pas."""
        etat, j0, _, _ = creer_etat_preflop()
        solveur   = SolveurSousJeu(nb_scenarios=10)
        scenarios = solveur._tirer_scenarios(etat, 0)
        for scenario in scenarios:
            toutes = []
            for cartes in scenario.values():
                toutes.extend(cartes)
            assert len(toutes) == len(set(toutes)), "Cartes dupliquées dans un scénario"

    def test_stats_apres_resolution(self):
        """Après resoudre(), stats['scenarios_evalues'] >= 0."""
        etat, j0, _, _ = creer_etat_preflop()
        agent   = creer_agent_heuristique()
        solveur = SolveurSousJeu(nb_scenarios=2, nb_iterations=5, temps_max=5.0)
        solveur.resoudre(etat, j0, agent)
        assert solveur.stats['scenarios_evalues'] >= 0
        assert solveur.stats['duree_s'] > 0

    def test_repr(self):
        """__repr__ retourne une chaîne valide."""
        solveur = SolveurSousJeu(nb_scenarios=10)
        assert "SolveurSousJeu" in repr(solveur)


# =============================================================================
# TESTS — training/self_play.py
# =============================================================================

class TestSelfPlay:
    """Tests du moteur de self-play."""

    def test_agents_baseline_existent(self):
        """Les 3 agents de référence sont instanciables."""
        assert AgentAleatoire()   is not None
        assert AgentCallOnly()    is not None
        assert AgentRaiseOnly()   is not None

    def test_agent_aleatoire_retourne_action_legale(self):
        """AgentAleatoire retourne toujours une action légale."""
        etat, j0, _, _ = creer_etat_preflop()
        agent   = AgentAleatoire(graine=42)
        legales = actions_legales(j0, etat.mise_courante, etat.pot, etat.mise_min_raise)
        for _ in range(20):
            action = agent.choisir_action(etat, j0, legales)
            assert action in legales

    def test_agent_call_only_pas_raise(self):
        """AgentCallOnly ne raise jamais si call/check est possible."""
        etat, j0, _, _ = creer_etat_preflop()
        agent   = AgentCallOnly()
        legales = actions_legales(j0, etat.mise_courante, etat.pot, etat.mise_min_raise)
        action  = agent.choisir_action(etat, j0, legales)
        assert action.type != TypeAction.RAISE

    def test_simuler_nb_mains_correct(self):
        """simuler() joue exactement nb_mains mains."""
        agents   = [AgentAleatoire(1), AgentAleatoire(2), AgentAleatoire(3)]
        resultat = simuler(agents, nb_mains=50, verbose=False)
        assert resultat.nb_mains == 50

    def test_simuler_gains_zero_sum(self):
        """La somme des gains nets est nulle (jeu à somme nulle)."""
        agents   = [AgentAleatoire(1), AgentAleatoire(2), AgentAleatoire(3)]
        resultat = simuler(agents, nb_mains=100, graine=42, verbose=False)
        somme    = sum(resultat.gains_nets)
        # Tolérance : arrondi entier sur les pots partagés (max 1 jeton/main perdu)
        assert abs(somme) <= 100, f"Somme gains non nulle : {somme}"

    def test_resultat_partie_winrate_bb100(self):
        """winrate_bb100() retourne un float cohérent."""
        res = ResultatPartie(
            gains_nets       = [+200, -100, -100],
            nb_mains         = 100,
            nb_mains_preflop = 10,
            nb_showdowns     = 90,
        )
        wr = res.winrate_bb100(0, grande_blinde=20)
        # +200 jetons / 20 BB / 100 mains * 100 = +10.0 bb/100
        assert abs(wr - 10.0) < 0.01

    def test_moteur_self_play_nb_agents(self):
        """MoteurSelfPlay lève ValueError si pas exactement 3 agents."""
        with pytest.raises(ValueError):
            MoteurSelfPlay(agents=[AgentAleatoire(), AgentAleatoire()])

    def test_axiom_vs_aleatoire_complete(self):
        """AXIOM (heuristique) vs 2 aléatoires : 50 mains sans erreur."""
        agent   = creer_agent_heuristique()
        agents  = [agent, AgentAleatoire(10), AgentAleatoire(11)]
        resultat = simuler(agents, nb_mains=50, verbose=False)
        assert resultat.nb_mains == 50
        assert isinstance(resultat.gains_nets[0], int)

    def test_vitesse_simulation(self):
        """3 agents aléatoires jouent ≥ 500 mains/s."""
        agents = [AgentAleatoire(i) for i in range(3)]
        debut  = time.time()
        simuler(agents, nb_mains=500, verbose=False)
        duree  = time.time() - debut
        mains_par_s = 500 / max(duree, 0.001)
        assert mains_par_s >= 500, f"Trop lent : {mains_par_s:.0f} mains/s"


# =============================================================================
# TESTS — training/evaluator.py
# =============================================================================

class TestEvaluateur:
    """Tests du module d'évaluation."""

    def test_creation(self):
        """L'évaluateur se crée sans erreur."""
        ev = Evaluateur(nb_mains=50, graine=42)
        assert ev.nb_mains == 50

    def test_evaluer_retourne_rapport(self):
        """evaluer() retourne un RapportEvaluation valide."""
        agent  = creer_agent_heuristique()
        ev     = Evaluateur(nb_mains=50, graine=42, verbose=False)
        rapport = ev.evaluer(agent)
        assert isinstance(rapport, RapportEvaluation)
        assert isinstance(rapport.winrate_vs_aleatoire, float)
        assert isinstance(rapport.winrate_vs_call,      float)
        assert isinstance(rapport.winrate_vs_raise,     float)
        assert isinstance(rapport.winrate_moyen,        float)

    def test_winrate_moyen_coherent(self):
        """winrate_moyen est la moyenne des 6 scénarios (baselines + semi-pros)."""
        agent   = creer_agent_heuristique()
        ev      = Evaluateur(nb_mains=50, graine=42, verbose=False)
        rapport = ev.evaluer(agent)
        import numpy as np
        attendu = float(np.mean([
            rapport.winrate_vs_aleatoire,
            rapport.winrate_vs_call,
            rapport.winrate_vs_raise,
            rapport.winrate_vs_tag,
            rapport.winrate_vs_lag,
            rapport.winrate_vs_regulier,
        ]))
        assert abs(rapport.winrate_moyen - attendu) < 1e-6

    def test_comparer_retourne_dict_trie(self):
        """comparer() retourne un dict trié par winrate_moyen_semipro décroissant."""
        ev = Evaluateur(nb_mains=50, graine=42, verbose=False)
        rapports = ev.comparer({
            'AXIOM'     : creer_agent_heuristique(),
            'Aléatoire' : AgentAleatoire(42),
            'Call-Only' : AgentCallOnly(),
        })
        assert len(rapports) == 3
        winrates = [r.winrate_moyen_semipro for r in rapports.values()]
        assert winrates == sorted(winrates, reverse=True)

    def test_benchmark_rapide(self):
        """benchmark_rapide() retourne un float."""
        agent = creer_agent_heuristique()
        wr    = benchmark_rapide(agent, nb_mains=50)
        assert isinstance(wr, float)

    def test_score_global(self):
        """score_global() applique la pondération (baselines×1 + semi-pros×2) / 9."""
        rapport = RapportEvaluation(
            winrate_vs_aleatoire  = 10.0,
            winrate_vs_call       = 5.0,
            winrate_vs_raise      = -5.0,
            winrate_vs_tag        = 3.0,
            winrate_vs_lag        = 1.0,
            winrate_vs_regulier   = 2.0,
            winrate_moyen         = (10.0 + 5.0 - 5.0 + 3.0 + 1.0 + 2.0) / 6.0,
            exploitabilite_approx = float('nan'),
        )
        score = rapport.score_global()
        assert isinstance(score, float)
        # Formule : (alea + call + raise + 2*tag + 2*lag + 2*reg) / 9
        attendu = (10.0 + 5.0 + (-5.0) + 2*3.0 + 2*1.0 + 2*2.0) / 9.0
        assert abs(score - attendu) < 1e-6

    def test_rapport_afficher_sans_erreur(self):
        """afficher() ne lève pas d'exception."""
        rapport = RapportEvaluation(
            winrate_vs_aleatoire = 5.0,
            winrate_vs_call      = 3.0,
            winrate_vs_raise     = -2.0,
            winrate_moyen        = 2.0,
        )
        rapport.afficher()   # ne doit pas lever d'exception


# =============================================================================
# RUNNER PRINCIPAL
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  AXIOM — Tests Phase 5 : Agent + Solveurs + Self-Play")
    print("="*60)

    # Lister toutes les classes de tests
    suites = [
        ("ai/agent.py",             TestAgentAXIOM),
        ("solver/depth_limited.py", TestSolveurProfondeurLimitee),
        ("solver/subgame_solver.py",TestSolveurSousJeu),
        ("training/self_play.py",   TestSelfPlay),
        ("training/evaluator.py",   TestEvaluateur),
    ]

    total_ok  = 0
    total_err = 0

    for module_nom, classe in suites:
        instance = classe()
        methodes = [m for m in dir(instance) if m.startswith('test_')]
        ok = 0; err = 0

        print(f"\n  ── {module_nom} ({len(methodes)} tests) ──")

        for nom in methodes:
            try:
                getattr(instance, nom)()
                print(f"    ✅ {nom}")
                ok += 1
            except Exception as e:
                print(f"    ❌ {nom}")
                print(f"       {type(e).__name__}: {e}")
                err += 1

        total_ok  += ok
        total_err += err
        print(f"  → {ok}/{ok+err} tests passés")

    print(f"\n{'='*60}")
    if total_err == 0:
        print(f"  ✅ {total_ok}/{total_ok} tests passés — Phase 5 validée !")
    else:
        print(f"  ⚠️  {total_ok}/{total_ok+total_err} tests passés "
              f"({total_err} échec(s))")
    print(f"{'='*60}\n")

    sys.exit(0 if total_err == 0 else 1)
