# =============================================================================
# AXIOM — solver/subgame_solver.py
# Solveur de sous-jeu avec blueprint comme prior (Phase 5).
#
# ─────────────────────────────────────────────────────────────────────────────
# PRINCIPE : SUBGAME SOLVING (Pluribus, Brown & Sandholm 2019)
# ─────────────────────────────────────────────────────────────────────────────
#
# Le blueprint est une stratégie globale calculée AVANT la partie.
# Il est optimal EN MOYENNE, mais pas nécessairement face à un joueur
# spécifique qui aurait une ligne de jeu particulière.
#
# Le subgame solver résout un SOUS-JEU localement :
# à partir d'un point précis de la main, il recalcule une stratégie
# optimale en tenant compte de :
#   1. La distribution de mains adverses probable à ce stade
#      (inférée depuis l'historique des actions — "range estimation")
#   2. La stratégie blueprint comme valeur de référence aux feuilles
#
# C'est la technique "Resolving" utilisée dans Libratus et Pluribus.
#
# ─────────────────────────────────────────────────────────────────────────────
# DIFFÉRENCE AVEC depth_limited.py
# ─────────────────────────────────────────────────────────────────────────────
#
# depth_limited.py  : CFR temps réel depuis la position courante
#                     → affine la stratégie sur la main en cours
#                     → Monte Carlo sur les cartes inconnues
#
# subgame_solver.py : résout un sous-jeu ENTIER en tenant compte
#                     des ranges adverses estimées depuis le blueprint
#                     → plus précis mais plus coûteux en calcul
#                     → utilisé en fin de main (RIVER surtout)
#
# ─────────────────────────────────────────────────────────────────────────────
# ESTIMATION DES RANGES
# ─────────────────────────────────────────────────────────────────────────────
#
# À chaque street, les actions adverses révèlent de l'information sur
# leurs mains (Bayes). Si un adversaire raise preflop, il a plutôt
# une main forte. Le solveur pondère les scénarios selon cette probabilité.
#
# Implémentation simplifiée (compatible avec les ressources disponibles) :
#   - On tire N scénarios de mains adverses aléatoires
#   - On pondère chaque scénario par la vraisemblance sous le blueprint
#     Poids(scénario) = Π actions_adversaire P_blueprint(action | infoset)
#   - On résout le CFR pondéré par ces poids
#
# =============================================================================

import time
import random
import math
import numpy as np
from typing import Optional, List

from treys import Deck as _TreysDeck, Evaluator as _Evaluator

from config.settings import TAILLES_MISE, ALL_IN, TAILLES_MISE_PREFLOP, TAILLES_MISE_POSTFLOP
from ai.network import encoder_infoset, NB_ACTIONS_MAX
from abstraction.card_abstraction import AbstractionCartes, AbstractionCartesV2
from abstraction.info_set import _normaliser, PALIERS_POT, PALIERS_STACK
from engine.game_state import EtatJeu, Phase
from engine.player import Joueur, StatutJoueur
from solver.depth_limited import (
    SolveurProfondeurLimitee,
    _NoeudLocal, _ActionL, _T,
    _H_ACTIF, _H_FOLD, _H_ALLIN,
    _H_PREFLOP, _H_FLOP, _H_TURN, _H_RIVER,
    _H_NOM_PHASE, _ORDRE_PREFLOP, _ORDRE_POSTFLOP,
    _eval_h, _rng_sol,
)


# =============================================================================
# CONSTANTES
# =============================================================================

# Nombre de scénarios de mains adverses tirés pour l'estimation des ranges
NB_SCENARIOS_RANGE = 20

# Poids minimum pour un scénario (évite les underflows numériques)
_POIDS_MIN = 1e-6


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class SolveurSousJeu:
    """
    Résout un sous-jeu en tenant compte des ranges adverses estimées.

    Ce solveur étend SolveurProfondeurLimitee en ajoutant :
      - L'estimation des ranges adverses depuis l'historique + blueprint
      - La résolution pondérée par scénario (chaque scénario = une réalisation
        possible des mains adverses)
      - Un résultat plus précis en fin de main (TURN/RIVER)

    Usage typique
    -------------
        solveur_sg = SolveurSousJeu(nb_scenarios=20, nb_iterations=100)
        strategie  = solveur_sg.resoudre(etat_jeu, joueur_axiom, agent)

    Paramètres de construction
    --------------------------
    nb_scenarios  : int
        Nombre de scénarios de mains adverses à tirer (défaut: 20).
        Plus de scénarios = meilleure estimation des ranges, mais plus lent.
    nb_iterations : int
        Itérations CFR PAR scénario (défaut: 50).
    temps_max     : float
        Durée maximum totale en secondes (défaut: 5.0).
    profondeur    : int
        Profondeur de recherche dans chaque scénario (défaut: 1).
    nb_simul_equite : int
        Simulations Monte Carlo pour l'évaluation d'équité (défaut: 30).
    """

    def __init__(self,
                 nb_scenarios    : int   = NB_SCENARIOS_RANGE,
                 nb_iterations   : int   = 50,
                 temps_max       : float = 5.0,
                 profondeur      : int   = 1,
                 nb_simul_equite : int   = 30):

        self.nb_scenarios    = nb_scenarios
        self.nb_iterations   = nb_iterations
        self.temps_max       = temps_max
        self.profondeur      = profondeur

        # Solveur de base (réutilise toute la mécanique de depth_limited)
        self._solveur_base = SolveurProfondeurLimitee(
            profondeur      = profondeur,
            nb_iterations   = nb_iterations,
            temps_max       = temps_max,
            nb_simul_equite = nb_simul_equite,
        )

        # Abstraction des cartes (V2 — 50 buckets postflop K-means)
        self._abs_cartes = AbstractionCartesV2()

        # Statistiques
        self.stats = {
            'scenarios_evalues' : 0,
            'iterations_total'  : 0,
            'duree_s'           : 0.0,
        }

        print(f"  🎯 SolveurSousJeu initialisé "
              f"(scenarios={nb_scenarios}, iter/scenario={nb_iterations}, "
              f"profondeur={profondeur})")

    # ==================================================================
    # MÉTHODE PRINCIPALE
    # ==================================================================

    def resoudre(self,
                 etat_jeu     : EtatJeu,
                 joueur_axiom : Joueur,
                 agent,
                 verbose      : bool = False) -> np.ndarray:
        """
        Résout le sous-jeu courant avec estimation des ranges adverses.

        Paramètres
        ----------
        etat_jeu     : état de jeu complet (EtatJeu)
        joueur_axiom : joueur AXIOM dont c'est le tour d'agir
        agent        : AgentAXIOM (blueprint + Deep CFR comme oracle)
        verbose      : afficher les étapes

        Retourne
        --------
        np.ndarray (NB_ACTIONS_MAX,) — stratégie pondérée par les ranges
        """
        debut = time.time()
        self.stats = {'scenarios_evalues': 0, 'iterations_total': 0, 'duree_s': 0.0}

        joueur_idx = etat_jeu.joueurs.index(joueur_axiom)

        # ── 1. Tirer des scénarios de mains adverses ───────────────────────
        scenarios = self._tirer_scenarios(etat_jeu, joueur_idx)

        if not scenarios:
            # Fallback : solveur de base sans estimation de range
            if verbose:
                print("  ⚠️  Aucun scénario valide → fallback solveur de base")
            return self._solveur_base.resoudre(etat_jeu, joueur_axiom, agent, verbose)

        # ── 2. Pondérer les scénarios par vraisemblance blueprint ─────────
        poids = self._calculer_poids(scenarios, etat_jeu, joueur_idx, agent)

        # ── 3. CFR pondéré sur chaque scénario ────────────────────────────
        strat_accumulee = np.zeros(NB_ACTIONS_MAX, dtype=np.float64)
        poids_total     = 0.0

        for i, (scenario, poids_i) in enumerate(zip(scenarios, poids)):
            if (time.time() - debut) > self.temps_max * 0.9:
                break   # ne pas dépasser le budget temps

            if poids_i < _POIDS_MIN:
                continue

            # Construire l'état pour ce scénario
            etat_scenario = self._appliquer_scenario(etat_jeu, scenario, joueur_idx)

            # Lancer le solveur de base sur ce scénario
            nb_iter_scenario = max(10, int(self.nb_iterations * poids_i / (sum(poids) + 1e-9)))
            solveur_iter = SolveurProfondeurLimitee(
                profondeur      = self.profondeur,
                nb_iterations   = nb_iter_scenario,
                temps_max       = max(0.1, self.temps_max - (time.time() - debut)),
                nb_simul_equite = self._solveur_base.nb_simul_equite,
            )

            # Utiliser le joueur de la copie (pas l'original) pour éviter
            # l'erreur "not in list" quand etat_scenario a de nouveaux objets
            joueur_axiom_copie = etat_scenario.joueurs[joueur_idx]
            strat_i = solveur_iter.resoudre(etat_scenario, joueur_axiom_copie, agent)

            strat_accumulee += poids_i * strat_i
            poids_total     += poids_i
            self.stats['scenarios_evalues']  += 1
            self.stats['iterations_total']   += solveur_iter.stats['iterations']

        self.stats['duree_s'] = time.time() - debut

        # ── 4. Normaliser la stratégie finale ─────────────────────────────
        if poids_total > _POIDS_MIN:
            strat_finale = (strat_accumulee / poids_total).astype(np.float32)
        else:
            strat_finale = np.ones(NB_ACTIONS_MAX, dtype=np.float32) / NB_ACTIONS_MAX

        total = strat_finale.sum()
        if total > 1e-9:
            strat_finale /= total

        if verbose:
            print(f"  🎯 SousJeu : {self.stats['scenarios_evalues']} scénarios, "
                  f"{self.stats['iterations_total']} itérations, "
                  f"{self.stats['duree_s']:.2f}s")

        return strat_finale

    # ==================================================================
    # ESTIMATION DES RANGES
    # ==================================================================

    def _tirer_scenarios(self,
                          etat_jeu    : EtatJeu,
                          joueur_idx  : int) -> list:
        """
        Tire NB_SCENARIOS scénarios de mains adverses possibles.

        Un scénario = une assignation de 2 cartes à chaque adversaire.
        Les cartes déjà connues (main d'AXIOM + board) sont exclues.

        Retourne une liste de scénarios :
            scenario = dict {joueur_idx_adv → [carte1, carte2]}
        """
        # Cartes déjà utilisées (main d'AXIOM + board)
        utilisees = set(etat_jeu.board)
        for j in etat_jeu.joueurs:
            if j == etat_jeu.joueurs[joueur_idx]:
                for c in (j.cartes or []):
                    utilisees.add(c)
            else:
                # Les cartes adverses sont "inconnues" — on ne les exclut pas
                # (c'est justement ce qu'on tire aléatoirement)
                pass

        # Deck disponible pour les tirages adverses
        deck = _TreysDeck()
        disponibles = [c for c in deck.cards if c not in utilisees]

        adversaires = [i for i in range(len(etat_jeu.joueurs))
                       if i != joueur_idx
                       and not etat_jeu.joueurs[i].est_elimine]

        nb_cartes_needed = 2 * len(adversaires)
        if len(disponibles) < nb_cartes_needed:
            return []

        scenarios = []
        for _ in range(self.nb_scenarios):
            _rng_sol.shuffle(disponibles)
            scenario = {}
            offset   = 0
            for adv_idx in adversaires:
                scenario[adv_idx] = disponibles[offset:offset + 2]
                offset += 2
            scenarios.append(scenario)

        return scenarios

    def _calculer_poids(self,
                         scenarios  : list,
                         etat_jeu   : EtatJeu,
                         joueur_idx : int,
                         agent) -> list:
        """
        Calcule le poids de vraisemblance de chaque scénario.

        Le poids d'un scénario est proportionnel à la probabilité que
        les adversaires aient joué exactement comme ils l'ont fait,
        ÉTANT DONNÉ les mains tirées dans ce scénario.

        Poids(scénario) ∝ Π_{adversaire j} P_blueprint(action_j | infoset_j(scénario))

        Si le blueprint n'est pas disponible, tous les poids sont égaux.
        """
        if agent is None or agent._blueprint is None:
            # Poids uniformes si pas de blueprint
            return [1.0] * len(scenarios)

        poids = []
        for scenario in scenarios:
            poids_scenario = 1.0

            for adv_idx, cartes_adv in scenario.items():
                # Reconstruire un infoset fictif pour cet adversaire avec ces cartes
                log_vraisemblance = self._log_vraisemblance_scenario(
                    etat_jeu, adv_idx, cartes_adv, agent
                )
                poids_scenario *= math.exp(log_vraisemblance)

            poids.append(max(poids_scenario, _POIDS_MIN))

        return poids

    def _log_vraisemblance_scenario(self,
                                     etat_jeu  : EtatJeu,
                                     adv_idx   : int,
                                     cartes_adv: list,
                                     agent) -> float:
        """
        Log-vraisemblance des actions passées d'un adversaire (Point 8).

        Itère sur historique_phases[phase] (déjà segmenté par street)
        pour connaître la phase exacte de chaque action — corrige le bug
        qui utilisait la phase courante pour toutes les actions passées.

        Pour chaque action de l'adversaire dans chaque phase :
          - Si le blueprint couvre l'infoset reconstruit → P_blueprint(action)
          - Sinon → prior basé sur l'équité de la main hypothétique

        Retourne la log-vraisemblance totale (float, ≤ 0).
        """
        blueprint = getattr(agent, '_blueprint', None)
        board     = list(etat_jeu.board)
        boards_par_phase = [[], board[:3], board[:4], board[:5]]
        _NOM_PHASE = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']

        log_total = 0.0

        try:
            _abs = self._abs_cartes
            from abstraction.info_set import (
                _normaliser, PALIERS_POT, PALIERS_STACK, _discretiser_raise_frac
            )
        except Exception:
            return 0.0

        gb = max(etat_jeu.grande_blinde, 1)

        # Itérer phase par phase (la phase de chaque action est connue)
        for phase_idx, hist_str in enumerate(etat_jeu.historique_phases):
            if not hist_str:
                continue

            board_phase = boards_par_phase[min(phase_idx, 3)]
            phase_name  = _NOM_PHASE[phase_idx]

            # Pré-calculer l'équité une seule fois par phase
            try:
                equite = _abs.bucket_et_equite(cartes_adv, board_phase)[1]
                bucket = _abs.bucket(cartes_adv, board_phase)
            except Exception:
                equite = 0.5
                bucket = 3

            # Lire les actions de l'adversaire dans cette phase depuis l'historique.
            # On utilise le comptage depuis hist_str pour savoir combien de fois
            # l'adversaire a agi. On ne connaît pas son ordre exact parmi les
            # 3 joueurs, mais on peut consulter le blueprint sur l'infoset reconstruit
            # avec l'historique partiel.
            #
            # Stratégie : consulter le blueprint une seule fois par phase avec
            # l'infoset final de la phase → probabilité cumulée cohérente.
            pot_norm   = _normaliser(etat_jeu.pot / gb, PALIERS_POT)
            stacks_str = ','.join(
                str(_normaliser(j.stack / gb, PALIERS_STACK))
                for j in etat_jeu.joueurs
            )
            raise_b = _discretiser_raise_frac(
                etat_jeu.mise_courante / max(etat_jeu.pot, 1)
            )
            cle = (f"{phase_name}"
                   f"|pos={adv_idx}"
                   f"|bucket={bucket}"
                   f"|pot={pot_norm}"
                   f"|stacks=({stacks_str})"
                   f"|hist={hist_str}"
                   f"|raise={raise_b}")

            # Compter les actions agressives de l'adversaire dans cette phase
            # (r = raise, a = all-in) et passives (f, x, c)
            nb_raises = sum(1 for i, ch in enumerate(hist_str)
                            if ch == 'r' and i + 1 < len(hist_str)
                            and hist_str[i + 1].isdigit())
            nb_allin  = hist_str.count('a')
            nb_passif = hist_str.count('f') + hist_str.count('x') + hist_str.count('c')
            nb_actions_adv = nb_raises + nb_allin + nb_passif

            if nb_actions_adv == 0:
                continue

            if blueprint is not None and cle in blueprint:
                noeud = blueprint[cle]
                strat = noeud.strategie_moyenne()
                if strat and len(strat) > 0:
                    tailles_p = (TAILLES_MISE_PREFLOP
                                 if phase_idx == 0 else TAILLES_MISE_POSTFLOP)
                    # Masse de proba sur actions agressives vs passives
                    idx_allin = 3 + len(tailles_p)
                    p_agressif = sum(strat[3:idx_allin]) + (strat[idx_allin]
                                     if idx_allin < len(strat) else 0.0)
                    p_passif   = sum(strat[:3])
                    somme      = p_agressif + p_passif
                    if somme > 1e-9:
                        p_agressif /= somme
                        p_passif   /= somme

                    # Accumuler la log-vraisemblance pour chaque action observée
                    for _ in range(nb_raises + nb_allin):
                        log_total += math.log(max(p_agressif, 1e-6))
                    for _ in range(nb_passif):
                        log_total += math.log(max(p_passif, 1e-6))
                    continue

            # Prior basé sur l'équité si blueprint absent ou infoset non couvert
            p_agressif_prior = max(0.05, equite)
            p_passif_prior   = max(0.05, 1.0 - equite)
            for _ in range(nb_raises + nb_allin):
                log_total += math.log(p_agressif_prior)
            for _ in range(nb_passif):
                log_total += math.log(p_passif_prior)

        return log_total

    # ==================================================================
    # APPLICATION D'UN SCÉNARIO À L'ÉTAT DE JEU
    # ==================================================================

    def _appliquer_scenario(self,
                             etat_jeu    : EtatJeu,
                             scenario    : dict,
                             joueur_idx  : int) -> EtatJeu:
        """
        Crée une copie de EtatJeu où les mains adverses sont remplacées
        par celles du scénario (pour la simulation).

        Les cartes adverses originales ne sont JAMAIS révélées en production —
        cette copie est uniquement pour le calcul interne du solveur.

        Retourne un nouvel objet EtatJeu (copie superficielle).
        """
        # Créer des copies légères des joueurs
        from engine.player import Joueur as _Joueur, TypeJoueur as _TypeJoueur

        nouveaux_joueurs = []
        for i, j in enumerate(etat_jeu.joueurs):
            j_copie = _Joueur(j.nom, j.type, j.stack, j.position)
            j_copie.statut    = j.statut
            j_copie.mise_tour = j.mise_tour
            # Assigner les cartes du scénario pour les adversaires
            if i in scenario:
                j_copie.cartes = list(scenario[i])
            else:
                j_copie.cartes = list(j.cartes) if j.cartes else []
            nouveaux_joueurs.append(j_copie)

        # Créer un nouvel EtatJeu avec ces joueurs
        etat_copie = EtatJeu(
            nouveaux_joueurs,
            etat_jeu.petite_blinde,
            etat_jeu.grande_blinde,
        )
        etat_copie.pot           = etat_jeu.pot
        etat_copie.board         = list(etat_jeu.board)
        etat_copie.phase         = etat_jeu.phase
        etat_copie.index_actif   = etat_jeu.index_actif
        etat_copie.mise_courante = etat_jeu.mise_courante
        etat_copie.mise_min_raise= etat_jeu.mise_min_raise
        etat_copie.historique    = list(etat_jeu.historique)

        return etat_copie

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================

    def afficher_stats(self) -> None:
        """Affiche les statistiques de la dernière résolution."""
        print(f"\n{'─'*45}")
        print(f"  SolveurSousJeu — Statistiques")
        print(f"{'─'*45}")
        print(f"  Scénarios évalués  : {self.stats['scenarios_evalues']}")
        print(f"  Itérations totales : {self.stats['iterations_total']}")
        print(f"  Durée              : {self.stats['duree_s']:.2f}s")
        print(f"{'─'*45}\n")

    def __repr__(self) -> str:
        return (f"SolveurSousJeu("
                f"nb_scenarios={self.nb_scenarios}, "
                f"nb_iter={self.nb_iterations}, "
                f"profondeur={self.profondeur}, "
                f"temps_max={self.temps_max}s)")


# =============================================================================
# TEST RAPIDE
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  AXIOM — Test subgame_solver.py")
    print("="*60)

    import numpy as np
    from engine.player import Joueur, TypeJoueur
    from engine.game_state import EtatJeu
    from ai.agent import AgentAXIOM

    j0 = Joueur("AXIOM",    TypeJoueur.AXIOM,  1500, 0)
    j1 = Joueur("Humain-1", TypeJoueur.HUMAIN, 1500, 1)
    j2 = Joueur("Humain-2", TypeJoueur.HUMAIN, 1500, 2)

    etat = EtatJeu([j0, j1, j2], petite_blinde=10, grande_blinde=20)
    etat.nouvelle_main()

    agent  = AgentAXIOM(mode_deterministe=False)
    solveur = SolveurSousJeu(
        nb_scenarios    = 5,
        nb_iterations   = 10,
        temps_max       = 5.0,
        profondeur      = 1,
        nb_simul_equite = 5,
    )
    print(f"\n  {solveur}")

    print(f"\n  Résolution sous-jeu (PREFLOP)...")
    strat = solveur.resoudre(etat, j0, agent, verbose=True)

    assert strat.shape == (NB_ACTIONS_MAX,), f"Shape incorrect : {strat.shape}"
    assert abs(strat.sum() - 1.0) < 1e-4,   f"Non normalisé : {strat.sum()}"

    print(f"\n  Stratégie sous-jeu : {np.round(strat, 3)}")
    solveur.afficher_stats()

    print("  ✅ Test subgame_solver.py passé !")
    print("="*60 + "\n")
