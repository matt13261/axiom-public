# =============================================================================
# AXIOM — solver/depth_limited.py
# Solveur à profondeur limitée — real-time search (Phase 5).
#
# ─────────────────────────────────────────────────────────────────────────────
# PRINCIPE : DEPTH-LIMITED SOLVING (Pluribus, Brown & Sandholm 2019)
# ─────────────────────────────────────────────────────────────────────────────
#
# Pendant la partie, plutôt que de simplement consulter le blueprint (stratégie
# précalculée), AXIOM lance un mini-CFR en temps réel à partir de la situation
# courante. Ce mini-CFR est "depth-limited" : il ne descend que sur N streets
# supplémentaires avant d'utiliser le blueprint/Deep CFR comme oracle de valeur.
#
# Avantage clé : AXIOM peut s'adapter à la ligne de jeu SPÉCIFIQUE de la main
# en cours, là où le blueprint ne faisait qu'une moyenne sur toutes les parties.
#
# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHME
# ─────────────────────────────────────────────────────────────────────────────
#
# Pour chaque itération (jusqu'à temps_max ou nb_iterations) :
#
#   1. Compléter le board avec des cartes aléatoires (si FLOP/TURN/RIVER)
#      → Monte Carlo sur les cartes cachées
#
#   2. Depuis l'état courant, lancer ES-MCCFR local à profondeur D :
#        - Nœud de joueur AXIOM     → explorer toutes les actions (External Sampling)
#        - Nœud adversaire          → échantillonner 1 action selon blueprint
#        - Nœud chance (nouvelle street) → continuer ou arrêter
#        - Feuille (profondeur 0 ou terminal) → évaluer avec l'oracle
#
#   3. Accumuler les regrets pour le nœud racine (position actuelle d'AXIOM)
#
# Résultat : regret_matching sur les regrets accumulés → stratégie affinée
#
# ─────────────────────────────────────────────────────────────────────────────
# PROFONDEUR
# ─────────────────────────────────────────────────────────────────────────────
#
# La profondeur est mesurée en nombre de STREETS restantes à explorer :
#   profondeur=0 → on est à une feuille → évaluer avec l'oracle
#   profondeur=1 → explorer la street courante puis évaluer
#   profondeur=2 → explorer 2 streets puis évaluer
#
# Valeur par défaut : profondeur=2 (équilibre vitesse/qualité)
# En fin de main (RIVER), profondeur=1 suffit (une seule street restante).
#
# ─────────────────────────────────────────────────────────────────────────────
# ORACLE DE VALEUR (feuilles au-delà de la profondeur)
# ─────────────────────────────────────────────────────────────────────────────
#
# Aux nœuds feuilles, on estime la valeur par :
#   1. Blueprint MCCFR (lookup O(1)) si l'infoset est connu
#   2. Réseau Deep CFR (forward pass) sinon
#   3. Équité des mains (Monte Carlo rapide) si aucun modèle disponible
#
# L'équité est convertie en valeur en jetons par :
#   valeur ≈ équité × pot − contribution_versée
#
# =============================================================================

import time
import random
import numpy as np
from typing import Optional, List

from treys import Deck as _TreysDeck, Evaluator as _Evaluator

from config.settings import TAILLES_MISE, ALL_IN, TAILLES_MISE_PREFLOP, TAILLES_MISE_POSTFLOP
from ai.network import encoder_infoset, NB_ACTIONS_MAX
from abstraction.card_abstraction import AbstractionCartes, AbstractionCartesV2
from abstraction.info_set import (
    _normaliser, PALIERS_POT, PALIERS_STACK, PALIERS_STACK_SPIN_RUSH,
    _discretiser_raise_frac, _format_hist_avec_cap,
)
from engine.game_state import EtatJeu, Phase
from engine.player import Joueur, StatutJoueur


# =============================================================================
# CONSTANTES INTERNES (reprises de mccfr.py pour l'indépendance du module)
# =============================================================================

_H_ACTIF   = 0
_H_FOLD    = 1
_H_ALLIN   = 2

_H_PREFLOP = 0
_H_FLOP    = 1
_H_TURN    = 2
_H_RIVER   = 3
_H_NOM_PHASE = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']

_ORDRE_PREFLOP  = [0, 1, 2]
_ORDRE_POSTFLOP = [1, 2, 0]

# Enum d'action léger (évite l'import de engine.actions dans les boucles internes)
class _T:
    FOLD  = 0
    CHECK = 1
    CALL  = 2
    RAISE = 3
    ALLIN = 4

class _ActionL:
    """Action légère pour le solveur interne."""
    __slots__ = ('type', 'montant')
    def __init__(self, t: int, montant: int = 0):
        self.type    = t
        self.montant = montant
    def __repr__(self):
        noms = {0:'FOLD',1:'CHECK',2:'CALL',3:'RAISE',4:'ALL_IN'}
        return f"{noms[self.type]}({self.montant})"

_eval_h  = _Evaluator()
_rng_sol = random.Random()


# =============================================================================
# NŒUD CFR LOCAL
# =============================================================================

class _NoeudLocal:
    """
    Nœud CFR minimal pour le solveur temps réel.
    Stocke uniquement les regrets et la stratégie du nœud racine
    (le nœud courant de la décision d'AXIOM).
    """
    __slots__ = ('regrets', 'strategie_somme', 'nb_visites')

    def __init__(self, nb_actions: int):
        self.regrets         = np.zeros(nb_actions, dtype=np.float64)
        self.strategie_somme = np.zeros(nb_actions, dtype=np.float64)
        self.nb_visites      = 0

    def strategie_courante(self) -> np.ndarray:
        """Regret matching → distribution courante."""
        pos = np.maximum(self.regrets, 0.0)
        total = pos.sum()
        if total > 1e-9:
            return (pos / total).astype(np.float32)
        nb = len(self.regrets)
        return np.ones(nb, dtype=np.float32) / nb

    def strategie_moyenne(self) -> np.ndarray:
        """Stratégie moyenne accumulée → stratégie de Nash."""
        total = self.strategie_somme.sum()
        if total > 1e-9:
            return (self.strategie_somme / total).astype(np.float32)
        nb = len(self.strategie_somme)
        return np.ones(nb, dtype=np.float32) / nb


# =============================================================================
# SOLVEUR PRINCIPAL
# =============================================================================

class SolveurProfondeurLimitee:
    """
    Solveur real-time à profondeur limitée pour AXIOM (inspiré de Pluribus).

    Pendant la partie, ce solveur affine la stratégie du blueprint en lançant
    un mini-CFR depuis la situation courante, limité à D streets supplémentaires.

    Usage typique
    -------------
        solveur = SolveurProfondeurLimitee(profondeur=2, temps_max=3.0)
        strategie = solveur.resoudre(etat_jeu, joueur_axiom, agent)
        # strategie : np.ndarray (NB_ACTIONS_MAX,) — probas sur actions abstraites

    Paramètres de construction
    --------------------------
    profondeur  : int
        Nombre de streets supplémentaires à explorer (défaut: 2).
        0 → pas de recherche (utilise directement l'oracle)
        1 → explore la street courante uniquement
        2 → explore 2 streets (recommandé)
    nb_iterations : int
        Nombre d'itérations CFR (défaut: 200). Plus = meilleur, mais plus long.
    temps_max : float
        Durée maximum en secondes (défaut: 3.0). La boucle s'arrête dès dépassement.
    nb_simul_equite : int
        Simulations Monte Carlo pour l'évaluation d'équité en feuille (défaut: 50).
    """

    def __init__(self,
                 profondeur      : int   = 2,
                 nb_iterations   : int   = 200,
                 temps_max       : float = 3.0,
                 nb_simul_equite : int   = 50):

        self.profondeur      = profondeur
        self.nb_iterations   = nb_iterations
        self.temps_max       = temps_max
        self.nb_simul_equite = nb_simul_equite

        # Abstraction des cartes (partagée, instanciée une seule fois)
        self._abs_cartes = AbstractionCartesV2()

        # Statistiques (pour diagnostic)
        self.stats = {
            'iterations'    : 0,   # itérations CFR effectuées
            'feuilles'      : 0,   # feuilles évaluées
            'feuilles_bp'   : 0,   # évaluées via blueprint
            'feuilles_dcfr' : 0,   # évaluées via Deep CFR
            'feuilles_equite': 0,  # évaluées via équité brute
        }

    # ==================================================================
    # MÉTHODE PRINCIPALE
    # ==================================================================

    def resoudre(self,
                 etat_jeu     : EtatJeu,
                 joueur_axiom : Joueur,
                 agent,
                 verbose      : bool = False) -> np.ndarray:
        """
        Lance le real-time search depuis la position courante.

        Paramètres
        ----------
        etat_jeu     : état de jeu complet (EtatJeu du moteur)
        joueur_axiom : le Joueur AXIOM dont c'est le tour d'agir
        agent        : AgentAXIOM (pour l'oracle de valeur aux feuilles)
        verbose      : afficher les étapes

        Retourne
        --------
        np.ndarray (NB_ACTIONS_MAX,) — distribution de probabilités
        affinée par le real-time search
        """
        debut = time.time()
        self.stats = {k: 0 for k in self.stats}
        # Cache inférence oracle : évite de répéter les forward passes PyTorch
        # pour le même infoset dans une même résolution. Clé = (joueur, bytes).
        self._oracle_cache: dict = {}

        # ── Convertir EtatJeu → dict léger ────────────────────────────────
        joueur_idx = etat_jeu.joueurs.index(joueur_axiom)
        etat_dict  = self._convertir_etat(etat_jeu, joueur_idx)

        # ── Créer le nœud racine ───────────────────────────────────────────
        actions_racine = self._actions_abstraites(etat_dict, joueur_idx)
        nb_actions     = len(actions_racine)
        if nb_actions == 0:
            return np.ones(NB_ACTIONS_MAX, dtype=np.float32) / NB_ACTIONS_MAX

        noeud_racine = _NoeudLocal(nb_actions)

        # Cartes connues pour exclure du tirage aléatoire
        cartes_connues = self._cartes_connues(etat_jeu)

        # ── Boucle CFR ────────────────────────────────────────────────────
        iteration = 0
        while (iteration < self.nb_iterations
               and (time.time() - debut) < self.temps_max):

            # Compléter le board avec des cartes aléatoires si nécessaire
            etat_iter = self._completer_board(etat_dict, cartes_connues)

            # CFR : explorer toutes les actions du joueur AXIOM
            valeurs = np.zeros(nb_actions, dtype=np.float64)
            strategie = noeud_racine.strategie_courante()

            for i, action in enumerate(actions_racine):
                etat_copie = self._copier_etat(etat_iter)
                file = list(etat_iter['joueurs_en_attente'])
                if file and file[0] == joueur_idx:
                    etat_copie['joueurs_en_attente'] = file[1:]
                self._appliquer_action(etat_copie, joueur_idx, action)
                valeurs[i] = self._cfr(
                    etat_copie, joueur_idx, self.profondeur, agent
                )

            # Valeur du nœud sous la stratégie courante
            valeur_noeud = float(np.dot(strategie[:nb_actions], valeurs))

            # CFR+ : plancher des regrets à 0 (convergence ~2× plus rapide)
            regrets_delta = valeurs - valeur_noeud
            noeud_racine.regrets[:nb_actions] = np.maximum(
                0.0, noeud_racine.regrets[:nb_actions] + regrets_delta
            )
            noeud_racine.strategie_somme[:nb_actions] += strategie[:nb_actions]
            noeud_racine.nb_visites += 1

            iteration += 1
            self.stats['iterations'] += 1

        duree = time.time() - debut

        if verbose:
            print(f"  🔍 Solveur : {iteration} itérations en {duree:.2f}s | "
                  f"feuilles={self.stats['feuilles']} "
                  f"(bp={self.stats['feuilles_bp']}, "
                  f"dcfr={self.stats['feuilles_dcfr']}, "
                  f"eq={self.stats['feuilles_equite']})")

        # ── Stratégie finale : padder à NB_ACTIONS_MAX ────────────────────
        strat_locale = noeud_racine.strategie_moyenne()
        # Mapper les actions locales vers les indices abstraits globaux
        strat_globale = np.zeros(NB_ACTIONS_MAX, dtype=np.float32)
        phase_racine = etat_dict['phase']
        for i, action in enumerate(actions_racine):
            idx_global = self._index_global(action, phase_racine)
            if 0 <= idx_global < NB_ACTIONS_MAX:
                strat_globale[idx_global] += strat_locale[i]

        total = strat_globale.sum()
        if total > 1e-9:
            strat_globale /= total
        else:
            strat_globale[:] = 1.0 / NB_ACTIONS_MAX

        return strat_globale

    # ==================================================================
    # CFR RÉCURSIF
    # ==================================================================

    def _cfr(self,
             etat          : dict,
             joueur_traversant : int,
             profondeur    : int,
             agent) -> float:
        """
        CFR récursif à profondeur limitée.

        Retourne la valeur nette pour joueur_traversant dans cet état.
        """
        # ── Nœud terminal ─────────────────────────────────────────────────
        if self._est_terminal(etat):
            self.stats['feuilles'] += 1
            return self._gain_terminal(etat, joueur_traversant)

        # ── Fin de street → transition ou feuille selon profondeur ────────
        if not etat['joueurs_en_attente']:
            actifs = [i for i in range(3)
                      if etat['statuts'][i] == _H_ACTIF]
            if len(actifs) >= 2 and etat['phase'] < _H_RIVER:
                if profondeur <= 0:
                    # Profondeur atteinte → oracle
                    self.stats['feuilles'] += 1
                    return self._valeur_oracle(etat, joueur_traversant, agent)
                # Passer à la street suivante
                etat_suivant = self._copier_etat(etat)
                self._passer_street(etat_suivant)
                return self._cfr(etat_suivant, joueur_traversant,
                                  profondeur - 1, agent)
            else:
                # Showdown ou fin de main
                self.stats['feuilles'] += 1
                return self._gain_terminal(etat, joueur_traversant)

        joueur_idx = etat['joueurs_en_attente'][0]
        actions    = self._actions_abstraites(etat, joueur_idx)

        if not actions:
            self.stats['feuilles'] += 1
            return self._gain_terminal(etat, joueur_traversant)

        # ── Nœud du joueur AXIOM (traversant) : explorer tout ─────────────
        if joueur_idx == joueur_traversant:
            valeurs  = []
            for action in actions:
                etat_copie = self._copier_etat(etat)
                etat_copie['joueurs_en_attente'] = list(etat['joueurs_en_attente'][1:])
                self._appliquer_action(etat_copie, joueur_idx, action)
                v = self._cfr(etat_copie, joueur_traversant, profondeur, agent)
                valeurs.append(v)

            # Stratégie courante du traversant (via blueprint ou Deep CFR)
            strat = self._oracle_strategie(etat, joueur_idx, agent,
                                            nb_actions=len(actions))
            valeur_noeud = sum(strat[i] * valeurs[i] for i in range(len(actions)))
            return valeur_noeud

        # ── Nœud adversaire : échantillonner 1 action ─────────────────────
        else:
            strat = self._oracle_strategie(etat, joueur_idx, agent,
                                            nb_actions=len(actions))
            # Roulette wheel
            r = _rng_sol.random()
            cumul = 0.0
            idx_choisi = len(actions) - 1
            for i, p in enumerate(strat):
                cumul += p
                if r <= cumul:
                    idx_choisi = i
                    break

            etat_copie = self._copier_etat(etat)
            etat_copie['joueurs_en_attente'] = list(etat['joueurs_en_attente'][1:])
            self._appliquer_action(etat_copie, joueur_idx, actions[idx_choisi])
            return self._cfr(etat_copie, joueur_traversant, profondeur, agent)

    # ==================================================================
    # ORACLE DE VALEUR (feuilles)
    # ==================================================================

    def _valeur_oracle(self, etat: dict, joueur_traversant: int, agent) -> float:
        """
        Estime la valeur pour joueur_traversant à une feuille de profondeur.

        Ordre de tentative :
          1. ReseauValeur (forward pass direct → valeur scalaire)
          2. Blueprint MCCFR (équité × pot pondérée par la stratégie blueprint)
          3. Équité brute Monte Carlo × pot
        """
        import torch

        # ── 1. ReseauValeur : prédit directement la valeur nette ──────────
        # C'est la source la plus précise : entraîné sur les gains réels
        # observés pendant les traversées Deep CFR.
        reseaux_valeur = getattr(agent, '_reseaux_valeur', None)
        if reseaux_valeur is not None:
            try:
                vec = encoder_infoset(etat, joueur_traversant)
                cache = getattr(self, '_oracle_cache', None)
                cache_key = (100 + joueur_traversant, vec.tobytes())
                if cache is not None and cache_key in cache:
                    self.stats['feuilles_dcfr'] += 1
                    return cache[cache_key]
                x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    val_pred = reseaux_valeur[joueur_traversant](x)
                val = float(val_pred.squeeze())
                if cache is not None:
                    cache[cache_key] = val
                self.stats['feuilles_dcfr'] += 1
                return val
            except Exception:
                pass

        # ── 2. Blueprint MCCFR : équité MC pondérée ───────────────────────
        # Si le blueprint couvre cet infoset, on utilise l'équité Monte Carlo
        # comme proxy de valeur (meilleure que équité brute seule car on sait
        # que le joueur joue selon une stratégie proche de Nash dans ce nœud).
        if agent is not None and agent._blueprint is not None:
            cle = self._cle_infoset(etat, joueur_traversant)
            if cle in agent._blueprint:
                equite = self._equite_rapide(etat, joueur_traversant)
                val    = equite * etat['pot'] - etat['contributions'][joueur_traversant]
                self.stats['feuilles_bp'] += 1
                return val

        # ── 3. Équité brute Monte Carlo ────────────────────────────────────
        equite = self._equite_rapide(etat, joueur_traversant)
        val    = equite * etat['pot'] - etat['contributions'][joueur_traversant]
        self.stats['feuilles_equite'] += 1
        return val

    def _oracle_strategie(self,
                           etat       : dict,
                           joueur_idx : int,
                           agent,
                           nb_actions : int) -> list:
        """
        Stratégie d'un joueur dans un état intermédiaire.
        Utilisé pour les adversaires ET le traversant dans les nœuds internes.

        Retourne une liste de nb_actions probabilités (somme = 1).
        """
        # ── Blueprint ─────────────────────────────────────────────────────
        if agent is not None and agent._blueprint is not None:
            cle = self._cle_infoset(etat, joueur_idx)
            if cle in agent._blueprint:
                strat_bp = agent._blueprint[cle].strategie_moyenne()
                strat = list(strat_bp[:nb_actions])
                somme = sum(strat)
                if somme > 1e-9:
                    return [s / somme for s in strat]

        # ── Deep CFR ──────────────────────────────────────────────────────
        if agent is not None and agent._reseaux_strategie is not None:
            try:
                import torch
                vec = encoder_infoset(etat, joueur_idx)
                cache_key = (joueur_idx, vec.tobytes())
                cache = getattr(self, '_oracle_cache', None)
                if cache is not None and cache_key in cache:
                    strat_full = cache[cache_key]
                else:
                    x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(agent.device)
                    with torch.no_grad():
                        out = agent._reseaux_strategie[joueur_idx](x)
                    strat_full = out.squeeze(0).cpu().numpy()
                    if cache is not None:
                        cache[cache_key] = strat_full
                strat = list(strat_full[:nb_actions])
                somme = sum(strat)
                if somme > 1e-9:
                    return [s / somme for s in strat]
            except Exception:
                pass

        # ── Uniforme ──────────────────────────────────────────────────────
        return [1.0 / nb_actions] * nb_actions

    # ==================================================================
    # ÉQUITÉ MONTE CARLO RAPIDE
    # ==================================================================

    def _equite_rapide(self, etat: dict, joueur_idx: int) -> float:
        """
        Estime l'équité du joueur_idx par Monte Carlo avec tirage de mains adverses.

        À chaque simulation :
          1. Tire un board complet aléatoire pour les cartes non encore visibles.
          2. Tire 2 cartes inconnues pour chaque adversaire actif dont on ne
             connaît pas la main (cas normal en partie réelle).
          3. Évalue toutes les mains et détermine le gagnant.

        C'est une vraie Monte Carlo : chaque simulation est indépendante.
        Retourne un float dans [0, 1].
        """
        actifs = [i for i in range(3)
                  if etat['statuts'][i] in (_H_ACTIF, _H_ALLIN)]
        if not actifs or joueur_idx not in actifs:
            return 0.0
        if len(actifs) == 1:
            return 1.0

        board_visible    = etat['board_visible']
        nb_board_visible = len(board_visible)

        # Cartes déjà utilisées et connues (main du joueur + board visible)
        cartes_connues = set(board_visible)
        for c in etat['cartes'][joueur_idx]:
            cartes_connues.add(c)

        # Adversaires dont on doit tirer les mains
        adversaires_sans_main = [
            i for i in actifs
            if i != joueur_idx and not etat['cartes'][i]
        ]
        adversaires_avec_main = [
            i for i in actifs
            if i != joueur_idx and etat['cartes'][i]
        ]
        for i in adversaires_avec_main:
            for c in etat['cartes'][i]:
                cartes_connues.add(c)

        # Deck disponible pour les tirages
        from treys import Deck as _DeckMC
        deck_disponible = [c for c in _DeckMC().cards if c not in cartes_connues]

        nb_cartes_adv    = 2 * len(adversaires_sans_main)
        nb_cartes_board  = 5 - nb_board_visible
        nb_cartes_needed = nb_cartes_adv + nb_cartes_board

        if len(deck_disponible) < nb_cartes_needed:
            # Pas assez de cartes disponibles → fallback équité uniforme
            return 1.0 / len(actifs)

        nb_victoires = 0
        for _ in range(self.nb_simul_equite):
            # Tirer toutes les cartes inconnues d'un coup (shuffle + slice)
            _rng_sol.shuffle(deck_disponible)
            offset = 0

            # Assigner les mains adverses inconnues
            mains_sim = {i: etat['cartes'][i] for i in actifs}
            for i in adversaires_sans_main:
                mains_sim[i] = deck_disponible[offset:offset + 2]
                offset += 2

            # Compléter le board
            board_sim = list(board_visible) + deck_disponible[offset:offset + nb_cartes_board]

            # Évaluer toutes les mains des actifs
            meilleur_score = None
            gagnants       = []
            for i in actifs:
                try:
                    score = _eval_h.evaluate(board_sim, mains_sim[i])
                except Exception:
                    score = 9999
                if meilleur_score is None or score < meilleur_score:
                    meilleur_score = score
                    gagnants = [i]
                elif score == meilleur_score:
                    gagnants.append(i)

            if joueur_idx in gagnants:
                nb_victoires += 1.0 / len(gagnants)

        return nb_victoires / self.nb_simul_equite

    # ==================================================================
    # CONVERSION EtatJeu → DICT LÉGER
    # ==================================================================

    def _convertir_etat(self, etat_jeu: EtatJeu, joueur_idx: int) -> dict:
        """
        Convertit un EtatJeu en dict léger compatible avec le solveur interne.

        Les cartes du board complet sont tirées aléatoirement pour les
        positions non encore révélées (nécessaire pour les simulations).
        """
        # ── Phase ─────────────────────────────────────────────────────────
        phase_map = {
            Phase.PREFLOP  : _H_PREFLOP,
            Phase.FLOP     : _H_FLOP,
            Phase.TURN     : _H_TURN,
            Phase.RIVER    : _H_RIVER,
            Phase.SHOWDOWN : _H_RIVER,
            Phase.TERMINEE : _H_RIVER,
        }
        phase_idx = phase_map.get(etat_jeu.phase, _H_PREFLOP)

        # ── Cartes des joueurs ─────────────────────────────────────────────
        cartes = [list(j.cartes) if j.cartes else [] for j in etat_jeu.joueurs]

        # ── Board visible ──────────────────────────────────────────────────
        board_visible = list(etat_jeu.board)

        # ── Board complet (tirer les cartes manquantes) ────────────────────
        board_complet = self._completer_board_depuis_etat(etat_jeu)

        # ── Stacks, contributions, mises ──────────────────────────────────
        stacks        = [j.stack             for j in etat_jeu.joueurs]
        contributions = [j.contribution_main for j in etat_jeu.joueurs]
        mises_tour    = [j.mise_tour         for j in etat_jeu.joueurs]

        # ── Statuts ────────────────────────────────────────────────────────
        statut_map = {
            StatutJoueur.ACTIF   : _H_ACTIF,
            StatutJoueur.FOLD    : _H_FOLD,
            StatutJoueur.ALL_IN  : _H_ALLIN,
            StatutJoueur.ELIMINE : _H_FOLD,
        }
        statuts = [statut_map.get(j.statut, _H_FOLD) for j in etat_jeu.joueurs]

        # ── File d'attente ─────────────────────────────────────────────────
        # Reconstruit l'ordre à partir du joueur actif courant
        if phase_idx == _H_PREFLOP:
            ordre = _ORDRE_PREFLOP
        else:
            ordre = _ORDRE_POSTFLOP

        idx_actif_moteur = etat_jeu.index_actif
        joueur_actif_courant = etat_jeu.joueurs.index(
            etat_jeu.joueur_actif()
        )
        # File : du joueur courant à la fin de l'ordre
        file = []
        idx_dans_ordre = None
        for k, j in enumerate(ordre):
            if j == joueur_actif_courant:
                idx_dans_ordre = k
                break
        if idx_dans_ordre is not None:
            depuis = idx_dans_ordre
            ordre_rotation = ordre[depuis:] + ordre[:depuis]
            file = [j for j in ordre_rotation if statuts[j] == _H_ACTIF]
        else:
            file = [j for j in ordre if statuts[j] == _H_ACTIF]

        # ── Historique par phase ───────────────────────────────────────────
        hist_phases = self._encoder_historique(etat_jeu, phase_idx)

        # ── Buckets + Équités ──────────────────────────────────────────────
        boards_phases = [
            [],
            board_complet[:3],
            board_complet[:4],
            board_complet[:5],
        ]
        buckets, equites = [], []
        for j in range(len(etat_jeu.joueurs)):
            bkts_j, eqs_j = [], []
            for p in range(4):
                bk, eq = self._abs_cartes.bucket_et_equite(cartes[j], boards_phases[p])
                bkts_j.append(bk); eqs_j.append(eq)
            buckets.append(bkts_j); equites.append(eqs_j)

        # raise_fracs : fraction mise_courante/pot au moment courant (phase courante)
        raise_fracs = [0.0, 0.0, 0.0, 0.0]
        raise_fracs[phase_idx] = etat_jeu.mise_courante / max(etat_jeu.pot, 1)

        return {
            'cartes'             : cartes,
            'board_complet'      : board_complet,
            'board_visible'      : board_visible,
            'stacks'             : stacks,
            'contributions'      : contributions,
            'mises_tour'         : mises_tour,
            'mise_courante'      : etat_jeu.mise_courante,
            'pot'                : etat_jeu.pot,
            'statuts'            : statuts,
            'phase'              : phase_idx,
            'joueurs_en_attente' : file,
            'hist_phases'        : hist_phases,
            'grande_blinde'      : max(etat_jeu.grande_blinde, 1),
            'buckets'            : buckets,
            'equites'            : equites,
            'raise_fracs'        : raise_fracs,
        }

    def _completer_board_depuis_etat(self, etat_jeu: EtatJeu) -> list:
        """
        Retourne un board complet de 5 cartes.
        Les cartes déjà visibles sont conservées, les manquantes sont tirées.
        """
        board_visible = list(etat_jeu.board)
        nb_manquantes = 5 - len(board_visible)
        if nb_manquantes == 0:
            return board_visible

        # Collecter toutes les cartes déjà utilisées
        utilisees = set(board_visible)
        for j in etat_jeu.joueurs:
            for c in (j.cartes or []):
                utilisees.add(c)

        # Tirer les cartes manquantes
        deck_treys = _TreysDeck()
        disponibles = [c for c in deck_treys.cards if c not in utilisees]
        _rng_sol.shuffle(disponibles)
        return board_visible + disponibles[:nb_manquantes]

    def _cartes_connues(self, etat_jeu: EtatJeu) -> set:
        """Retourne l'ensemble de toutes les cartes déjà distribuées."""
        connues = set(etat_jeu.board)
        for j in etat_jeu.joueurs:
            for c in (j.cartes or []):
                connues.add(c)
        return connues

    def _completer_board(self, etat_dict: dict, cartes_connues: set) -> dict:
        """
        Pour chaque itération CFR, tire un nouveau board_complet aléatoire
        pour les cartes non encore révélées.
        """
        board_visible  = etat_dict['board_visible']
        nb_manquantes  = 5 - len(board_visible)

        if nb_manquantes == 0:
            return etat_dict  # rien à changer

        # Nouvelle copie avec un board différent
        deck_treys = _TreysDeck()
        # Exclure toutes les cartes connues Y COMPRIS les mains adverses
        # (dans un contexte réel, les mains adverses sont inconnues →
        #  on ne les exclut pas du tirage, ce qui est correct)
        toutes_utilisees = set(board_visible)
        for cartes_j in etat_dict['cartes']:
            toutes_utilisees.update(cartes_j)
        disponibles = [c for c in deck_treys.cards if c not in toutes_utilisees]
        _rng_sol.shuffle(disponibles)

        nouveau_board = list(board_visible) + disponibles[:nb_manquantes]

        # Mettre à jour les buckets ET équités pour le nouveau board
        boards_phases = [[], nouveau_board[:3], nouveau_board[:4], nouveau_board[:5]]
        nb_joueurs = len(etat_dict['cartes'])
        nouveaux_buckets, nouvelles_equites = [], []
        for j in range(nb_joueurs):
            bkts_j, eqs_j = [], []
            for p in range(4):
                bk, eq = self._abs_cartes.bucket_et_equite(etat_dict['cartes'][j], boards_phases[p])
                bkts_j.append(bk); eqs_j.append(eq)
            nouveaux_buckets.append(bkts_j); nouvelles_equites.append(eqs_j)

        etat_iter = dict(etat_dict)  # copie superficielle (board_complet change)
        etat_iter['board_complet'] = nouveau_board
        etat_iter['buckets']       = nouveaux_buckets
        etat_iter['equites']       = nouvelles_equites
        return etat_iter

    def _encoder_historique(self, etat_jeu: EtatJeu, phase_idx: int) -> list:
        """Encode l'historique en 4 séquences de lettres (une par street).

        Utilise directement etat_jeu.historique_phases qui est déjà segmenté
        par street par le moteur de jeu — identique au format d'entraînement
        MCCFR et à ce que fait agent.py:_convertir_etat.

        L'ancienne version itérait sur etat_jeu.historique et attribuait
        TOUTES les actions à phase_idx (phase courante), corrompant les 4
        slots : preflop/flop/turn actions terminaient toutes dans hist[river].
        """
        return list(etat_jeu.historique_phases)

    # ==================================================================
    # OPÉRATIONS SUR L'ÉTAT DICT (reprises et adaptées de mccfr.py)
    # ==================================================================

    def _est_terminal(self, etat: dict) -> bool:
        """État terminal si 0 ou 1 joueur actif/allin, ou phase > RIVER."""
        actifs = [i for i in range(3)
                  if etat['statuts'][i] in (_H_ACTIF, _H_ALLIN)]
        if len(actifs) <= 1:
            return True
        if etat['phase'] > _H_RIVER:
            return True
        return False

    def _gain_terminal(self, etat: dict, joueur_traversant: int) -> float:
        """Gain net du joueur_traversant dans un nœud terminal."""
        actifs = [i for i in range(3)
                  if etat['statuts'][i] in (_H_ACTIF, _H_ALLIN)]
        if not actifs:
            return -etat['contributions'][joueur_traversant]
        if len(actifs) == 1:
            gagnant = actifs[0]
            if joueur_traversant == gagnant:
                return etat['pot'] - etat['contributions'][joueur_traversant]
            return -etat['contributions'][joueur_traversant]

        # Showdown
        board = etat['board_visible'] if len(etat['board_visible']) == 5 \
                else etat['board_complet']
        meilleur_score, gagnants = None, []
        for i in actifs:
            try:
                score = _eval_h.evaluate(board, etat['cartes'][i])
            except Exception:
                score = 9999
            if meilleur_score is None or score < meilleur_score:
                meilleur_score = score; gagnants = [i]
            elif score == meilleur_score:
                gagnants.append(i)

        if joueur_traversant in gagnants:
            return (etat['pot'] / len(gagnants)) - etat['contributions'][joueur_traversant]
        return -etat['contributions'][joueur_traversant]

    def _actions_abstraites(self, etat: dict, joueur_idx: int) -> list:
        """Retourne les actions légales abstraites (reprises de mccfr.py)."""
        stack         = etat['stacks'][joueur_idx]
        mise_tour     = etat['mises_tour'][joueur_idx]
        mise_courante = etat['mise_courante']
        pot           = etat['pot']
        gb            = etat['grande_blinde']
        a_payer       = mise_courante - mise_tour

        actions = []
        if a_payer > 0:
            actions.append(_ActionL(_T.FOLD))
        if a_payer == 0:
            actions.append(_ActionL(_T.CHECK))
        if 0 < a_payer < stack:
            actions.append(_ActionL(_T.CALL, montant=mise_courante))

        tailles_phase = (TAILLES_MISE_PREFLOP if etat['phase'] == _H_PREFLOP
                         else TAILLES_MISE_POSTFLOP)
        for fraction in tailles_phase:
            montant_raise = mise_courante + int(pot * fraction)
            montant_raise = max(montant_raise, mise_courante + gb)
            if montant_raise < mise_tour + stack:
                actions.append(_ActionL(_T.RAISE, montant=montant_raise))

        if ALL_IN and stack > 0:
            montant_allin = mise_tour + stack
            if not any(a.type == _T.RAISE and a.montant == montant_allin
                       for a in actions):
                actions.append(_ActionL(_T.ALLIN, montant=montant_allin))

        # Dédoublonnage
        vus, result = set(), []
        for a in actions:
            cle = (a.type, a.montant)
            if cle not in vus:
                vus.add(cle); result.append(a)
        return result

    def _appliquer_action(self, etat: dict, joueur_idx: int,
                           action: _ActionL) -> None:
        """Applique une action sur l'état (en place)."""
        phase     = etat['phase']
        stack     = etat['stacks'][joueur_idx]
        mise_tour = etat['mises_tour'][joueur_idx]

        if action.type == _T.FOLD:
            etat['statuts'][joueur_idx] = _H_FOLD
            etat['hist_phases'][phase] += 'f'

        elif action.type == _T.CHECK:
            etat['hist_phases'][phase] += 'x'

        elif action.type == _T.CALL:
            a_payer_reel = min(etat['mise_courante'] - mise_tour, stack)
            etat['stacks'][joueur_idx]        -= a_payer_reel
            etat['mises_tour'][joueur_idx]    += a_payer_reel
            etat['contributions'][joueur_idx] += a_payer_reel
            etat['pot']                       += a_payer_reel
            if etat['stacks'][joueur_idx] == 0:
                etat['statuts'][joueur_idx] = _H_ALLIN
            etat['hist_phases'][phase] += 'c'

        elif action.type in (_T.RAISE, _T.ALLIN):
            # Point 3 : bucket de sizing AVANT de modifier le pot
            frac_raise   = action.montant / max(etat['pot'], 1)
            raise_bucket = _discretiser_raise_frac(frac_raise)

            a_payer_reel = min(action.montant - mise_tour, stack)
            etat['stacks'][joueur_idx]        -= a_payer_reel
            etat['mises_tour'][joueur_idx]    += a_payer_reel
            etat['contributions'][joueur_idx] += a_payer_reel
            etat['pot']                       += a_payer_reel
            nouvelle_mise = etat['mises_tour'][joueur_idx]
            if nouvelle_mise > etat['mise_courante']:
                etat['mise_courante'] = nouvelle_mise
                etat['raise_fracs'][phase] = etat['mise_courante'] / max(etat['pot'], 1)
                self._reinserer_apres_raise(etat, joueur_idx)
            if etat['stacks'][joueur_idx] == 0:
                etat['statuts'][joueur_idx] = _H_ALLIN
            if action.type == _T.RAISE:
                code = f'r{raise_bucket}'
            else:
                code = 'a'
            etat['hist_phases'][phase] += code

    def _reinserer_apres_raise(self, etat: dict, raiser_idx: int) -> None:
        """Reconstruit la file d'attente après un raise."""
        ordre = (_ORDRE_PREFLOP if etat['phase'] == _H_PREFLOP
                 else _ORDRE_POSTFLOP)
        pos = ordre.index(raiser_idx) if raiser_idx in ordre else 0
        depuis_apres = ordre[pos + 1:] + ordre[:pos + 1]
        etat['joueurs_en_attente'] = [
            j for j in depuis_apres
            if j != raiser_idx and etat['statuts'][j] == _H_ACTIF
        ]

    def _passer_street(self, etat: dict) -> None:
        """Passe à la street suivante (en place)."""
        etat['mises_tour']    = [0, 0, 0]
        etat['mise_courante'] = 0
        etat['phase']        += 1
        phase = etat['phase']
        if phase == _H_FLOP:
            etat['board_visible'] = etat['board_complet'][:3]
        elif phase == _H_TURN:
            etat['board_visible'] = etat['board_complet'][:4]
        elif phase == _H_RIVER:
            etat['board_visible'] = etat['board_complet'][:5]
        etat['joueurs_en_attente'] = [
            j for j in _ORDRE_POSTFLOP
            if etat['statuts'][j] == _H_ACTIF
        ]

    def _copier_etat(self, etat: dict) -> dict:
        """Copie profonde légère de l'état."""
        return {
            'cartes'             : etat['cartes'],
            'board_complet'      : etat['board_complet'],
            'board_visible'      : list(etat['board_visible']),
            'stacks'             : list(etat['stacks']),
            'contributions'      : list(etat['contributions']),
            'mises_tour'         : list(etat['mises_tour']),
            'mise_courante'      : etat['mise_courante'],
            'pot'                : etat['pot'],
            'statuts'            : list(etat['statuts']),
            'phase'              : etat['phase'],
            'joueurs_en_attente' : list(etat['joueurs_en_attente']),
            'hist_phases'        : list(etat['hist_phases']),
            'grande_blinde'      : etat['grande_blinde'],
            'buckets'            : etat['buckets'],
            'equites'            : etat.get('equites'),
            'raise_fracs'        : list(etat.get('raise_fracs', [0.0, 0.0, 0.0, 0.0])),
        }

    def _cle_infoset(self, etat: dict, joueur_idx: int) -> str:
        """Construit la clé d'infoset (format compatible blueprint MCCFR)."""
        phase = etat['phase']
        gb    = max(etat['grande_blinde'], 1)
        # P7 : stacks bucketises Spin & Rush, hist abstrait S/M/L + cap 6
        pot_norm   = _normaliser(etat['pot'] / gb, PALIERS_POT)
        stacks_str = ','.join(
            str(_normaliser(etat['stacks'][i] / gb, PALIERS_STACK_SPIN_RUSH))
            for i in range(3)
        )
        raise_bucket = _discretiser_raise_frac(
            etat['mise_courante'] / max(etat['pot'], 1)
        )
        return (f"{_H_NOM_PHASE[phase]}"
                f"|pos={joueur_idx}"
                f"|bucket={etat['buckets'][joueur_idx][phase]}"
                f"|pot={pot_norm}"
                f"|stacks=({stacks_str})"
                f"|hist={_format_hist_avec_cap(etat['hist_phases'][phase])}"
                f"|raise={raise_bucket}")

    def _index_global(self, action: _ActionL, phase: int = _H_PREFLOP) -> int:
        """Mappe une _ActionL vers son index abstrait global (0..NB_ACTIONS_MAX-1)."""
        if action.type == _T.FOLD:  return 0
        if action.type == _T.CHECK: return 1
        if action.type == _T.CALL:  return 2
        if action.type == _T.ALLIN:
            tailles = (TAILLES_MISE_PREFLOP if phase == _H_PREFLOP
                       else TAILLES_MISE_POSTFLOP)
            return 3 + len(tailles)
        if action.type == _T.RAISE:
            return 3   # toutes les raises se cumulent à l'index de base (agent gère la répartition)
        return -1

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================

    def __repr__(self) -> str:
        return (f"SolveurProfondeurLimitee("
                f"profondeur={self.profondeur}, "
                f"nb_iter={self.nb_iterations}, "
                f"temps_max={self.temps_max}s)")


# =============================================================================
# TEST RAPIDE
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  AXIOM — Test depth_limited.py")
    print("="*60)

    from engine.player import Joueur, TypeJoueur
    from engine.game_state import EtatJeu
    from ai.agent import AgentAXIOM

    j0 = Joueur("AXIOM",    TypeJoueur.AXIOM,  1500, 0)
    j1 = Joueur("Humain-1", TypeJoueur.HUMAIN, 1500, 1)
    j2 = Joueur("Humain-2", TypeJoueur.HUMAIN, 1500, 2)

    etat = EtatJeu([j0, j1, j2], petite_blinde=10, grande_blinde=20)
    etat.nouvelle_main()

    agent   = AgentAXIOM(mode_deterministe=False)
    solveur = SolveurProfondeurLimitee(
        profondeur=1, nb_iterations=20, temps_max=5.0, nb_simul_equite=10
    )
    print(f"\n  {solveur}")

    print(f"\n  Lancement real-time search (PREFLOP, 20 itérations)...")
    strat = solveur.resoudre(etat, j0, agent, verbose=True)

    assert strat.shape == (NB_ACTIONS_MAX,), f"Shape incorrect : {strat.shape}"
    assert abs(strat.sum() - 1.0) < 1e-4, f"Non normalisé : {strat.sum()}"
    print(f"\n  Stratégie affinée : {np.round(strat, 3)}")
    print(f"  Stats : {solveur.stats}")

    print("\n  ✅ Test depth_limited.py passé !")
    print("="*60 + "\n")
