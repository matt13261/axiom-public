# =============================================================================
# AXIOM — train_hu.py
# Entraînement MCCFR Heads-Up (2 joueurs) pour AXIOM.
#
# Quand un joueur est éliminé en tournoi, AXIOM passe en mode 1v1.
# Ce script entraîne un blueprint MCCFR spécifiquement pour le jeu heads-up,
# bien plus rapide à calculer (2 joueurs → espace de jeu ~3× plus petit).
#
# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS EN HEADS-UP
# ─────────────────────────────────────────────────────────────────────────────
# BTN/SB = joueur 0 : dealer, poste la petite blinde, PARLE EN PREMIER préflop
# BB     = joueur 1 : poste la grande blinde, répond préflop, PARLE EN PREMIER postflop
#
# Règle HU (conforme au poker officiel) :
#   Préflop  : [0, 1]  → BTN/SB act first, BB second (avec l'option)
#   Postflop : [1, 0]  → BB first, BTN/SB second
#
# ─────────────────────────────────────────────────────────────────────────────
# FORMAT DE CLÉ D'INFOSET HU
# ─────────────────────────────────────────────────────────────────────────────
# "HU_PREFLOP|pos=P|bucket=B|pot=N|stacks=(A,B)|hist=H"
#
# Le préfixe "HU_" + 2 stacks (au lieu de 3) garantissent
# zéro collision avec les clés du blueprint 3-joueurs.
#
# ─────────────────────────────────────────────────────────────────────────────
# UTILISATION
# ─────────────────────────────────────────────────────────────────────────────
# Depuis le dossier axiom/ :
#
#   python train_hu.py                  # entraînement complet (recommandé)
#   python train_hu.py --iterations 50000 --verbose  # test rapide
#
# Génère : data/strategy/blueprint_hu.pkl
# =============================================================================

import argparse
import random
import time
import os
import pickle
import sys
import multiprocessing as mp

from treys import Deck as _TreysDeck, Evaluator as _EvalHU

from config.settings import (
    TAILLES_MISE_PREFLOP  as _TAILLES_PREFLOP,
    TAILLES_MISE_POSTFLOP as _TAILLES_POSTFLOP,
    ALL_IN                as _ALL_IN,
    NIVEAUX_BLINDES,
    STACK_DEPART,
    NB_WORKERS_MAX,
)
from abstraction.card_abstraction import AbstractionCartes as _AbsCartes
from abstraction.info_set import (
    _normaliser, PALIERS_POT, PALIERS_STACK, PALIERS_STACK_SPIN_RUSH,
    _discretiser_raise_frac, _format_hist_avec_cap,
)
from engine.actions import Action as _Action, TypeAction as _TypeAction
from ai.mccfr import NoeudCFR
from ai.strategy import sauvegarder_blueprint, charger_blueprint


# =============================================================================
# CONSTANTES INTERNES
# =============================================================================

# Statuts joueur dans l'état léger
_HU_ACTIF = 0    # encore dans la main, peut agir
_HU_FOLD  = 1    # s'est couché
_HU_ALLIN = 2    # all-in, ne peut plus agir

# Phases de jeu
_HU_PREFLOP = 0
_HU_FLOP    = 1
_HU_TURN    = 2
_HU_RIVER   = 3
_HU_NOM_PHASE = ['HU_PREFLOP', 'HU_FLOP', 'HU_TURN', 'HU_RIVER']

# Ordres d'action heads-up
# Préflop  : BTN/SB=0 parle en premier, BB=1 en second (avec l'option)
# Postflop : BB=1 parle en premier, BTN/SB=0 en second
_HU_ORDRE_PREFLOP  = [0, 1]
_HU_ORDRE_POSTFLOP = [1, 0]

# Point 7 — Élagage epsilon (Pluribus)
_EPSILON_ELAGAGE_HU = 0.05   # seuil de probabilité minimale
_PROBA_ELAGAGE_HU   = 0.95   # proportion des traversées adverses avec élagage

# Chemin de sauvegarde du blueprint HU
CHEMIN_HU = "data/strategy/blueprint_hu.pkl"

# Évaluateur treys (singleton)
_eval_hu = _EvalHU()


# =============================================================================
# CLASSE PRINCIPALE : MCCFRHeadsUp
# =============================================================================

class MCCFRHeadsUp:
    """
    External Sampling MCCFR pour Texas Hold'em No Limit 2 joueurs (Heads-Up).

    Construit un blueprint HU par auto-jeu : les 2 joueurs jouent contre eux-mêmes
    en accumulant des regrets, jusqu'à converger vers un Équilibre de Nash.

    Les clés d'infoset utilisent le préfixe "HU_" et un format à 2 stacks,
    garantissant une compatibilité avec le chargement sélectif dans agent.py.

    Usage :
        mccfr_hu = MCCFRHeadsUp()
        mccfr_hu.entrainer(
            nb_iterations=100_000, pb=10, gb=20,
            verbose=True, save_every=25_000,
            chemin='data/strategy/blueprint_hu.pkl'
        )

    Reprendre un entraînement interrompu :
        mccfr_hu.noeuds = charger_blueprint('data/strategy/blueprint_hu.pkl')
        mccfr_hu.entrainer(...)
    """

    def __init__(self):
        self.noeuds: dict    = {}   # cle_infoset (str) → NoeudCFR
        self.iterations: int = 0    # nombre d'itérations complètes effectuées

        # Point 1 — Linear CFR : numéro global d'itération (mis à jour dans entrainer())
        self._iteration_courante: int = 0

        # Abstraction des cartes en mode HU : 1 adversaire + seuils et table
        # preflop calibrés pour la distribution d'équité heads-up. Indispensable
        # pour que le blueprint HU utilise les mêmes buckets que l'agent à
        # l'exécution en endgame (via abstraction_cartes_hu dans agent.py).
        self._abs_cartes = _AbsCartes(nb_simulations=100, mode='hu')

    # ------------------------------------------------------------------
    # ENTRAÎNEMENT PRINCIPAL
    # ------------------------------------------------------------------

    def entrainer(self, nb_iterations: int, stacks: int = 500,
                  pb: int = 10, gb: int = 20, verbose: bool = False,
                  save_every: int = 0, chemin: str = None) -> None:
        """
        Lance nb_iterations d'External Sampling MCCFR heads-up.

        Chaque itération comprend :
          1. Un deal aléatoire (2 mains + 5 cartes board précalculées)
          2. 2 traversées de l'arbre (une par joueur traversant)

        nb_iterations : nombre d'itérations à effectuer
        stacks        : stack de départ de chaque joueur
        pb, gb        : petite et grande blinde
        verbose       : affiche la progression toutes les 10%
        save_every    : sauvegarde automatique toutes les N itérations
        chemin        : chemin du fichier .pkl de sauvegarde
        """
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  AXIOM — MCCFR Heads-Up | {nb_iterations:,} itérations")
            print(f"  Stacks={stacks} | Blindes {pb}/{gb}")
            print(f"  Infosets existants : {len(self.noeuds):,}")
            print(f"{'─'*60}")

        for it in range(1, nb_iterations + 1):
            # Point 1 — Linear CFR : numéro global de l'itération en cours
            self._iteration_courante = self.iterations + it

            # Deal aléatoire : cartes + board complet précalculé
            etat_base = self._dealer_aleatoire(stacks, pb, gb)

            # 2 traversées : une par joueur traversant
            for joueur_traversant in range(2):
                etat = self._copier_etat(etat_base)
                self._es_mccfr(etat, joueur_traversant)

            self.iterations += 1

            # Progression toutes les 10%
            if verbose:
                pas = max(1, nb_iterations // 10)
                if it % pas == 0:
                    print(f"  It. {it:8,}/{nb_iterations:,} | "
                          f"Infosets : {len(self.noeuds):,}")

            # Sauvegarde automatique
            if save_every > 0 and chemin and it % save_every == 0:
                sauvegarder_blueprint(self.noeuds, chemin)
                if verbose:
                    print(f"  💾 Sauvegarde intermédiaire → {chemin}")

        if verbose:
            print(f"{'─'*60}")
            print(f"  ✅ Entraînement terminé | {len(self.noeuds):,} infosets")
            print(f"{'─'*60}\n")

    # ------------------------------------------------------------------
    # TRAVERSÉE RÉCURSIVE ES-MCCFR
    # ------------------------------------------------------------------

    def _es_mccfr(self, etat: dict, joueur_traversant: int) -> float:
        """
        Traversée récursive External Sampling MCCFR (2 joueurs).

        Retourne l'utilité (gain net en jetons) pour le joueur_traversant.
        """
        # ── Joueurs encore dans la main ─────────────────────────────────
        actifs_allin = [i for i in range(2)
                        if etat['statuts'][i] in (_HU_ACTIF, _HU_ALLIN)]
        actifs_pouvant_agir = [i for i in actifs_allin
                               if etat['statuts'][i] == _HU_ACTIF]

        # ── Nœud terminal : un seul joueur non-fold ─────────────────────
        if len(actifs_allin) <= 1:
            return self._gain_fold(etat, joueur_traversant)

        # ── File vide : fin du tour de parole ────────────────────────────
        if not etat['joueurs_en_attente']:
            if etat['phase'] == _HU_RIVER:
                return self._gain_showdown(etat, joueur_traversant)

            # Les deux en all-in → distribuer les cartes restantes et showdown
            if not actifs_pouvant_agir:
                etat = self._copier_etat(etat)
                while etat['phase'] < _HU_RIVER:
                    self._passer_street(etat)
                return self._gain_showdown(etat, joueur_traversant)

            # Street suivante normale
            etat = self._copier_etat(etat)
            self._passer_street(etat)
            return self._es_mccfr(etat, joueur_traversant)

        # ── Joueur courant dans la file ──────────────────────────────────
        joueur_idx = etat['joueurs_en_attente'][0]

        # Passer les joueurs fold / all-in présents dans la file
        if etat['statuts'][joueur_idx] != _HU_ACTIF:
            etat = self._copier_etat(etat)
            etat['joueurs_en_attente'] = etat['joueurs_en_attente'][1:]
            return self._es_mccfr(etat, joueur_traversant)

        # ── Actions abstraites et nœud CFR ──────────────────────────────
        actions = self._actions_abstraites(etat, joueur_idx)
        if not actions:
            etat = self._copier_etat(etat)
            etat['joueurs_en_attente'] = etat['joueurs_en_attente'][1:]
            return self._es_mccfr(etat, joueur_traversant)

        cle   = self._cle_infoset(etat, joueur_idx)
        noeud = self._obtenir_noeud(cle, len(actions))

        # ── Joueur traversant : explorer TOUTES ses actions ──────────────
        if joueur_idx == joueur_traversant:
            # Point 1 — Linear CFR : pondération par t (itérations récentes = plus lourd)
            strat_full = noeud.strategie_courante(float(self._iteration_courante))
            strategie  = strat_full[:len(actions)]
            somme_s    = sum(strategie)
            if somme_s > 0:
                strategie = [s / somme_s for s in strategie]
            else:
                strategie = [1.0 / len(actions)] * len(actions)

            valeurs = []
            for action in actions:
                etat_copie = self._copier_etat(etat)
                etat_copie['joueurs_en_attente'] = list(
                    etat['joueurs_en_attente'][1:])
                self._appliquer_action(etat_copie, joueur_idx, action)
                v = self._es_mccfr(etat_copie, joueur_traversant)
                valeurs.append(v)

            valeur_noeud = sum(strategie[i] * valeurs[i]
                               for i in range(len(actions)))

            for i in range(len(actions)):
                noeud.regrets_cumules[i] += valeurs[i] - valeur_noeud

            return valeur_noeud

        # ── Adversaire : échantillonner UNE action ───────────────────────
        else:
            strat_full = self._regret_matching(noeud)
            strategie  = strat_full[:len(actions)]
            somme_s    = sum(strategie)
            if somme_s > 0:
                strategie = [s / somme_s for s in strategie]
            else:
                strategie = [1.0 / len(actions)] * len(actions)

            # Point 1 — Linear CFR : pondérer la stratégie somme par t
            t = float(self._iteration_courante)
            for i in range(len(actions)):
                noeud.strategie_somme[i] += t * strategie[i]
            noeud.nb_visites += 1

            # Point 7 — Élagage epsilon (même logique que MCCFRHoldEm)
            if random.random() < _PROBA_ELAGAGE_HU:
                actions_retenues  = [a for a, p in zip(actions, strategie)
                                     if p >= _EPSILON_ELAGAGE_HU]
                strategie_retenue = [p for p in strategie
                                     if p >= _EPSILON_ELAGAGE_HU]
                if actions_retenues:
                    somme_r = sum(strategie_retenue)
                    strategie_retenue = [p / somme_r for p in strategie_retenue]
                    idx_action = self._echantillonner(strategie_retenue)
                    action = actions_retenues[idx_action]
                else:
                    idx_action = self._echantillonner(strategie)
                    action = actions[idx_action]
            else:
                idx_action = self._echantillonner(strategie)
                action     = actions[idx_action]

            etat_copie = self._copier_etat(etat)
            etat_copie['joueurs_en_attente'] = list(
                etat['joueurs_en_attente'][1:])
            self._appliquer_action(etat_copie, joueur_idx, action)
            return self._es_mccfr(etat_copie, joueur_traversant)

    # ------------------------------------------------------------------
    # DEAL ALÉATOIRE (nœud chance)
    # ------------------------------------------------------------------

    def _dealer_aleatoire(self, stacks: int, pb: int, gb: int) -> dict:
        """
        Crée un état de départ heads-up avec un deal aléatoire.

        BTN/SB = joueur 0 : poste la petite blinde
        BB     = joueur 1 : poste la grande blinde
        Distribue 2 cartes par joueur + 5 cartes pour le board.
        """
        deck = list(_TreysDeck().cards)
        random.shuffle(deck)

        cartes        = [deck[0:2], deck[2:4]]   # [BTN/SB, BB]
        board_complet = deck[4:9]                 # 5 cartes précalculées

        stacks_liste  = [stacks, stacks]
        contributions = [0, 0]
        mises_tour    = [0, 0]

        # BTN/SB (joueur 0) poste la petite blinde
        mise_sb = min(pb, stacks_liste[0])
        stacks_liste[0]  -= mise_sb
        contributions[0] += mise_sb
        mises_tour[0]     = mise_sb

        # BB (joueur 1) poste la grande blinde
        mise_bb = min(gb, stacks_liste[1])
        stacks_liste[1]  -= mise_bb
        contributions[1] += mise_bb
        mises_tour[1]     = mise_bb

        # Statuts (ALL_IN si stack épuisé par les blindes)
        statuts = [
            _HU_ALLIN if stacks_liste[i] == 0 else _HU_ACTIF
            for i in range(2)
        ]

        # Précalcul des buckets ET équités (une fois par deal, partagé)
        buckets, equites = self._precomputer_buckets_et_equites(cartes, board_complet)

        return {
            'cartes'             : cartes,
            'board_complet'      : board_complet,  # partagé, read-only
            'board_visible'      : [],
            'stacks'             : stacks_liste,
            'contributions'      : contributions,
            'mises_tour'         : mises_tour,
            'mise_courante'      : gb,
            'pot'                : mise_sb + mise_bb,
            'statuts'            : statuts,
            'phase'              : _HU_PREFLOP,
            'joueurs_en_attente' : list(_HU_ORDRE_PREFLOP),  # [0, 1]
            'hist_phases'        : ['', '', '', ''],
            'grande_blinde'      : gb,
            'buckets'            : buckets,         # partagé, read-only
            'equites'            : equites,         # partagé, read-only
            'raise_fracs'        : [0.0, 0.0, 0.0, 0.0],
        }

    def _precomputer_buckets_et_equites(self, cartes: list,
                                        board_complet: list) -> tuple:
        """
        Précalcule les buckets ET équités pour 2 joueurs × 4 phases.

        Retourne (buckets, equites) :
          buckets[joueur][phase] → int (0 à 7)
          equites[joueur][phase] → float (0.0 à 1.0)
        """
        boards_par_phase = [
            [],                    # PREFLOP (lookup table)
            board_complet[:3],     # FLOP    (Monte Carlo 100 simulations)
            board_complet[:4],     # TURN
            board_complet[:5],     # RIVER
        ]
        buckets, equites = [], []
        for j in range(2):
            bkts_j, eqs_j = [], []
            for board in boards_par_phase:
                bk, eq = self._abs_cartes.bucket_et_equite(cartes[j], board)
                bkts_j.append(bk); eqs_j.append(eq)
            buckets.append(bkts_j); equites.append(eqs_j)
        return buckets, equites

    # ------------------------------------------------------------------
    # CLÉ D'INFOSET HU
    # ------------------------------------------------------------------

    def _cle_infoset(self, etat: dict, joueur_idx: int) -> str:
        """
        Construit la clé d'infoset heads-up.

        Format : "HU_PREFLOP|pos=P|bucket=B|pot=N|stacks=(A,B)|hist=H"

        Le préfixe "HU_" et le format à 2 stacks garantissent
        zéro collision avec les clés du blueprint 3-joueurs.
        """
        phase = etat['phase']
        gb    = max(etat['grande_blinde'], 1)

        # P7 : stacks bucketises Spin & Rush (7 niveaux), hist abstrait S/M/L + cap 6
        pot_norm   = _normaliser(etat['pot'] / gb, PALIERS_POT)
        stacks_str = ','.join(
            str(_normaliser(etat['stacks'][i] / gb, PALIERS_STACK_SPIN_RUSH))
            for i in range(2)
        )

        raise_bucket = _discretiser_raise_frac(
            etat['mise_courante'] / max(etat['pot'], 1)
        )

        return (
            f"{_HU_NOM_PHASE[phase]}"
            f"|pos={joueur_idx}"
            f"|bucket={etat['buckets'][joueur_idx][phase]}"
            f"|pot={pot_norm}"
            f"|stacks=({stacks_str})"
            f"|hist={_format_hist_avec_cap(etat['hist_phases'][phase])}"
            f"|raise={raise_bucket}"
        )

    # ------------------------------------------------------------------
    # ACTIONS ABSTRAITES HU
    # ------------------------------------------------------------------

    def _actions_abstraites(self, etat: dict, joueur_idx: int) -> list:
        """
        Retourne les actions légales abstraites pour un joueur en HU.

        FOLD   : si une mise est à suivre
        CHECK  : si rien à suivre
        CALL   : si mise à suivre et stack suffisant
        RAISE  : aux fractions du pot définies dans TAILLES_MISE_PREFLOP/POSTFLOP
        ALL_IN : toujours si stack > 0
        """
        stack         = etat['stacks'][joueur_idx]
        mise_tour     = etat['mises_tour'][joueur_idx]
        mise_courante = etat['mise_courante']
        pot           = etat['pot']
        gb            = etat['grande_blinde']
        a_payer       = mise_courante - mise_tour

        actions = []

        if a_payer > 0:
            actions.append(_Action(_TypeAction.FOLD))

        if a_payer == 0:
            actions.append(_Action(_TypeAction.CHECK))

        if 0 < a_payer < stack:
            actions.append(_Action(_TypeAction.CALL, montant=mise_courante))

        # Tailles de mise selon la street
        tailles = _TAILLES_PREFLOP if etat['phase'] == _HU_PREFLOP else _TAILLES_POSTFLOP
        for fraction in tailles:
            montant_raise = mise_courante + int(pot * fraction)
            montant_raise = max(montant_raise, mise_courante + gb)
            if montant_raise < mise_tour + stack:
                actions.append(_Action(_TypeAction.RAISE, montant=montant_raise))

        if _ALL_IN and stack > 0:
            montant_allin = mise_tour + stack
            if not any(a.type == _TypeAction.RAISE and a.montant == montant_allin
                       for a in actions):
                actions.append(_Action(_TypeAction.ALL_IN, montant=montant_allin))

        # Dédoublonnage
        vus = set()
        actions_uniques = []
        for a in actions:
            cle = (a.type, a.montant)
            if cle not in vus:
                vus.add(cle)
                actions_uniques.append(a)

        return actions_uniques

    # ------------------------------------------------------------------
    # APPLIQUER UNE ACTION
    # ------------------------------------------------------------------

    def _appliquer_action(self, etat: dict, joueur_idx: int,
                          action: _Action) -> None:
        """
        Applique une action sur l'état (en place — l'état a déjà été copié).
        """
        phase     = etat['phase']
        stack     = etat['stacks'][joueur_idx]
        mise_tour = etat['mises_tour'][joueur_idx]

        if action.type == _TypeAction.FOLD:
            etat['statuts'][joueur_idx] = _HU_FOLD
            etat['hist_phases'][phase] += 'f'

        elif action.type == _TypeAction.CHECK:
            etat['hist_phases'][phase] += 'x'

        elif action.type == _TypeAction.CALL:
            a_payer      = etat['mise_courante'] - mise_tour
            a_payer_reel = min(a_payer, stack)
            etat['stacks'][joueur_idx]        -= a_payer_reel
            etat['mises_tour'][joueur_idx]    += a_payer_reel
            etat['contributions'][joueur_idx] += a_payer_reel
            etat['pot']                       += a_payer_reel
            if etat['stacks'][joueur_idx] == 0:
                etat['statuts'][joueur_idx] = _HU_ALLIN
            etat['hist_phases'][phase] += 'c'

        elif action.type in (_TypeAction.RAISE, _TypeAction.ALL_IN):
            a_payer      = action.montant - mise_tour
            a_payer_reel = min(a_payer, stack)

            # Calculer le bucket AVANT la mise à jour du pot
            if action.type == _TypeAction.RAISE:
                frac_raise   = action.montant / max(etat['pot'], 1)
                raise_bucket = _discretiser_raise_frac(frac_raise)
                code = f'r{raise_bucket}'
            else:
                code = 'a'

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
                etat['statuts'][joueur_idx] = _HU_ALLIN

            etat['hist_phases'][phase] += code

    def _reinserer_apres_raise(self, etat: dict, raiser_idx: int) -> None:
        """
        En HU, après un raise, l'unique adversaire doit répondre.
        On le réinsère dans la file s'il est encore ACTIF.
        """
        autre = 1 - raiser_idx
        if etat['statuts'][autre] == _HU_ACTIF:
            etat['joueurs_en_attente'] = [autre]
        else:
            etat['joueurs_en_attente'] = []

    # ------------------------------------------------------------------
    # TRANSITION DE STREET
    # ------------------------------------------------------------------

    def _passer_street(self, etat: dict) -> None:
        """
        Passe à la street suivante (modification en place).

        - Remet les mises du tour à 0
        - Avance la phase (PREFLOP→FLOP→TURN→RIVER)
        - Met à jour le board visible
        - Reconstruit la file en ordre postflop HU : BB(1) → BTN(0)
        """
        etat['mises_tour']    = [0, 0]
        etat['mise_courante'] = 0
        etat['phase']        += 1
        phase = etat['phase']

        if phase == _HU_FLOP:
            etat['board_visible'] = etat['board_complet'][:3]
        elif phase == _HU_TURN:
            etat['board_visible'] = etat['board_complet'][:4]
        elif phase == _HU_RIVER:
            etat['board_visible'] = etat['board_complet'][:5]

        # File postflop HU : BB(1) en premier, BTN/SB(0) en second
        etat['joueurs_en_attente'] = [
            j for j in _HU_ORDRE_POSTFLOP
            if etat['statuts'][j] == _HU_ACTIF
        ]

    # ------------------------------------------------------------------
    # GAINS AUX NŒUDS TERMINAUX
    # ------------------------------------------------------------------

    def _gain_fold(self, etat: dict, joueur_traversant: int) -> float:
        """
        Gain net quand la main se termine par fold.
        Le seul joueur non-fold remporte le pot.
        """
        actifs = [i for i in range(2)
                  if etat['statuts'][i] in (_HU_ACTIF, _HU_ALLIN)]
        if not actifs:
            return -etat['contributions'][joueur_traversant]

        gagnant = actifs[0]
        if joueur_traversant == gagnant:
            return etat['pot'] - etat['contributions'][joueur_traversant]
        else:
            return -etat['contributions'][joueur_traversant]

    def _gain_showdown(self, etat: dict, joueur_traversant: int) -> float:
        """
        Gain net au showdown HU.

        Évalue les 2 mains avec treys (score le plus bas = gagnant).
        En cas d'égalité, le pot est partagé.
        """
        board = (etat['board_visible']
                 if len(etat['board_visible']) == 5
                 else etat['board_complet'])

        actifs = [i for i in range(2)
                  if etat['statuts'][i] in (_HU_ACTIF, _HU_ALLIN)]

        if len(actifs) == 0:
            return 0.0
        if len(actifs) == 1:
            return self._gain_fold(etat, joueur_traversant)

        # Évaluer les 2 mains (treys : score bas = main forte)
        meilleur_score = None
        gagnants       = []
        for i in actifs:
            try:
                score = _eval_hu.evaluate(board, etat['cartes'][i])
            except Exception:
                score = 9999
            if meilleur_score is None or score < meilleur_score:
                meilleur_score = score
                gagnants = [i]
            elif score == meilleur_score:
                gagnants.append(i)

        if joueur_traversant in gagnants:
            return (etat['pot'] / len(gagnants)) - etat['contributions'][joueur_traversant]
        else:
            return -etat['contributions'][joueur_traversant]

    # ------------------------------------------------------------------
    # UTILITAIRES INTERNES
    # ------------------------------------------------------------------

    def _obtenir_noeud(self, cle: str, nb_actions: int) -> NoeudCFR:
        """Récupère ou crée un NoeudCFR pour cette clé."""
        if cle not in self.noeuds:
            self.noeuds[cle] = NoeudCFR(nb_actions)
        noeud = self.noeuds[cle]
        # Étendre si le nombre d'actions a augmenté (stacks variables)
        if noeud.nb_actions < nb_actions:
            diff = nb_actions - noeud.nb_actions
            noeud.regrets_cumules.extend([0.0] * diff)
            noeud.strategie_somme.extend([0.0] * diff)
            noeud.nb_actions = nb_actions
        return noeud

    def _copier_etat(self, etat: dict) -> dict:
        """
        Copie légère de l'état pour la traversée de l'arbre.

        'board_complet' et 'buckets' sont partagés (read-only).
        Les listes mutables sont copiées (shallow copy).
        """
        return {
            'cartes'             : etat['cartes'],           # read-only
            'board_complet'      : etat['board_complet'],    # read-only
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
            'buckets'            : etat['buckets'],          # read-only
            'equites'            : etat['equites'],          # read-only
            'raise_fracs'        : list(etat['raise_fracs']),
        }

    def _regret_matching(self, noeud: NoeudCFR) -> list:
        """Regret matching sans effets de bord (pour les nœuds adversaires)."""
        regrets_pos = [max(r, 0.0) for r in noeud.regrets_cumules]
        somme_pos   = sum(regrets_pos)
        if somme_pos > 0:
            return [r / somme_pos for r in regrets_pos]
        return [1.0 / noeud.nb_actions] * noeud.nb_actions

    def _echantillonner(self, strategie: list) -> int:
        """Roulette wheel sampling — retourne un index selon la distribution."""
        r     = random.random()
        cumul = 0.0
        for i, p in enumerate(strategie):
            cumul += p
            if r <= cumul:
                return i
        return len(strategie) - 1   # sécurité numérique

    # ------------------------------------------------------------------
    # STATISTIQUES ET AFFICHAGE
    # ------------------------------------------------------------------

    def afficher_stats(self) -> None:
        """Affiche un résumé du blueprint HU : répartition par phase, visites."""
        if not self.noeuds:
            print("  Aucun infoset — lancez d'abord entrainer().")
            return

        comptage        = {nom: 0 for nom in _HU_NOM_PHASE}
        visites_totales = 0

        for cle, noeud in self.noeuds.items():
            for nom in _HU_NOM_PHASE:
                if cle.startswith(nom + '|'):
                    comptage[nom] += 1
                    break
            visites_totales += noeud.nb_visites

        print(f"\n{'─'*55}")
        print(f"  AXIOM Blueprint HU | {self.iterations:,} itérations")
        print(f"  Infosets total : {len(self.noeuds):,}")
        for nom in _HU_NOM_PHASE:
            print(f"    {nom:12} : {comptage[nom]:,}")
        moy = visites_totales / len(self.noeuds) if self.noeuds else 0
        print(f"  Visites moy/infoset : {moy:.1f}")
        print(f"{'─'*55}\n")

    def reinitialiser(self) -> None:
        """Remet tous les nœuds et le compteur d'itérations à zéro."""
        self.noeuds.clear()
        self.iterations = 0

    def __repr__(self) -> str:
        return (f"MCCFRHeadsUp(iterations={self.iterations:,}, "
                f"infosets={len(self.noeuds):,})")


# =============================================================================
# SCRIPT D'ENTRAÎNEMENT
# =============================================================================

# =============================================================================
# MULTIPROCESSING — worker, fusion, barre de progression
# =============================================================================

def _worker_hu_mp(pb, gb, nb_iterations, stacks, it_offset,
                  compteur_partage, chemin_tmp, worker_id):
    """
    Worker multiprocessing : effectue nb_iterations de MCCFR HU.
    Incrémente compteur_partage (mp.Value) tous les rapport_chaque itérations.
    Sauvegarde le dict de noeuds dans un fichier temp (évite le deadlock queue).
    it_offset : décalage Linear CFR pour que chaque worker ait des t distincts.
    """
    import os as _os, sys as _sys, pickle as _pickle
    dossier = _os.path.dirname(_os.path.abspath(__file__))
    if dossier not in _sys.path:
        _sys.path.insert(0, dossier)

    mccfr          = MCCFRHeadsUp()
    rapport_chaque = max(10, nb_iterations // 50)   # ~50 updates par worker

    for it in range(1, nb_iterations + 1):
        mccfr._iteration_courante = it_offset + it
        etat_base = mccfr._dealer_aleatoire(stacks, pb, gb)
        for joueur in range(2):
            etat = mccfr._copier_etat(etat_base)
            mccfr._es_mccfr(etat, joueur)
        mccfr.iterations += 1

        if it % rapport_chaque == 0:
            with compteur_partage.get_lock():
                compteur_partage.value += rapport_chaque

    # Incrémenter le reste
    reste = nb_iterations % rapport_chaque
    if reste:
        with compteur_partage.get_lock():
            compteur_partage.value += reste

    # Sauvegarder dans un fichier temp (pas de queue → pas de deadlock)
    with open(chemin_tmp, 'wb') as f:
        _pickle.dump(mccfr.noeuds, f, protocol=4)


def _fusionner_noeuds(liste_noeuds: list) -> dict:
    """
    Fusionne plusieurs dicts {cle: NoeudCFR} en additionnant
    regret_somme, strategie_somme et nb_visites noeud par noeud.
    """
    fusionnes = {}
    for noeuds in liste_noeuds:
        for cle, noeud in noeuds.items():
            if cle not in fusionnes:
                nb_a  = len(noeud.regrets_cumules)
                copie = NoeudCFR(nb_a)
                copie.regrets_cumules = list(noeud.regrets_cumules)
                copie.strategie_somme = list(noeud.strategie_somme)
                copie.nb_visites      = noeud.nb_visites
                fusionnes[cle] = copie
            else:
                existant = fusionnes[cle]
                n = min(len(noeud.regrets_cumules), len(existant.regrets_cumules))
                for i in range(n):
                    existant.regrets_cumules[i] += noeud.regrets_cumules[i]
                    existant.strategie_somme[i] += noeud.strategie_somme[i]
                existant.nb_visites += noeud.nb_visites
    return fusionnes


def _barre_progression(it_actuel: int, it_total: int,
                       nb_infosets: int, t_debut: float,
                       largeur: int = 38) -> None:
    """
    Affiche une barre de progression sur la ligne courante (overwrite).

    Exemple :
      [████████████████████░░░░░░░░░░░░░░░░░░]  52.3%  |  652,500/1,250,000  |  17,901 infosets  |  ETA 4m 12s
    """
    pct    = min(it_actuel / max(it_total, 1), 1.0)
    rempli = int(largeur * pct)
    barre  = '\u2588' * rempli + '\u2591' * (largeur - rempli)

    elapsed = time.time() - t_debut
    if pct > 0.001:
        eta_sec = elapsed / pct * (1.0 - pct)
        if eta_sec < 60:
            eta_str = f"{int(eta_sec)}s"
        elif eta_sec < 3600:
            eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"
        else:
            eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60):02d}m"
    else:
        eta_str = "--"

    w     = len(f"{it_total:,}")
    ligne = (
        f"\r  [{barre}] {pct * 100:5.1f}%"
        f"  |  {it_actuel:{w},}/{it_total:,}"
        f"  |  {nb_infosets:,} infosets"
        f"  |  ETA {eta_str}   "
    )
    sys.stdout.write(ligne)
    sys.stdout.flush()


def entrainer_blueprint_hu(nb_iterations_par_niveau: int = 75_000,
                            chemin: str = CHEMIN_HU,
                            verbose: bool = True,
                            reprendre: bool = True,
                            nb_workers: int = None) -> MCCFRHeadsUp:
    """
    Lance l'entraînement complet du blueprint HU sur les 8 niveaux de blindes.
    Utilise le multiprocessing (NB_WORKERS_MAX cœurs) et affiche une barre de
    progression en temps réel pour chaque niveau.

    nb_iterations_par_niveau : itérations MCCFR par niveau de blindes
    chemin                   : chemin de sauvegarde du fichier .pkl
    verbose                  : affiche les stats détaillées après chaque niveau
    reprendre                : reprend depuis le fichier existant si disponible
    nb_workers               : nombre de cœurs (défaut : NB_WORKERS_MAX)
    """
    mccfr = MCCFRHeadsUp()

    # ── Nombre de workers ───────────────────────────────────────────────
    cpu_dispo = mp.cpu_count()
    if nb_workers is None:
        nb_workers = min(NB_WORKERS_MAX, cpu_dispo)
    nb_workers = max(1, min(nb_workers, cpu_dispo))

    # ── Reprendre un entraînement existant ─────────────────────────────
    if reprendre and os.path.exists(chemin):
        try:
            mccfr.noeuds = charger_blueprint(chemin)
            print(f"  Reprise depuis le fichier existant "
                  f"({len(mccfr.noeuds):,} infosets)")
        except Exception as e:
            print(f"  Impossible de charger le fichier existant : {e}")
            print(f"  Entrainement depuis zero")

    t_debut_global = time.time()
    total_iterations = nb_iterations_par_niveau * len(NIVEAUX_BLINDES)

    print(f"\n{'='*62}")
    print(f"  AXIOM -- ENTRAINEMENT BLUEPRINT HEADS-UP")
    print(f"  {nb_iterations_par_niveau:,} iterations x {len(NIVEAUX_BLINDES)} niveaux"
          f" = {total_iterations:,} total")
    print(f"  Workers CPU : {nb_workers} / {cpu_dispo} coeurs")
    print(f"  Sauvegarde  : {chemin}")
    print(f"{'='*62}")

    for niveau_idx, (pb, gb, _) in enumerate(NIVEAUX_BLINDES):
        bb_ratio = STACK_DEPART // gb
        print(f"\n  -- Niveau {niveau_idx+1}/{len(NIVEAUX_BLINDES)} : "
              f"Blindes {pb}/{gb}  ({bb_ratio} BB)  --  {nb_workers} workers")

        t_niveau = time.time()

        # ── Répartir les itérations entre workers ───────────────────────
        its_par_worker = [nb_iterations_par_niveau // nb_workers] * nb_workers
        for i in range(nb_iterations_par_niveau % nb_workers):
            its_par_worker[i] += 1

        # Offsets Linear CFR : chaque worker a des numéros d'itération distincts
        offsets = [sum(its_par_worker[:i]) for i in range(nb_workers)]

        # Fichiers temporaires : chaque worker écrit ses noeuds sur disque
        # → évite le deadlock pipe/queue sur les gros dicts
        dossier_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'data', 'tmp_hu')
        os.makedirs(dossier_tmp, exist_ok=True)
        chemins_tmp = [os.path.join(dossier_tmp, f'worker_{niveau_idx}_{w}.pkl')
                       for w in range(nb_workers)]

        # Nettoyer les fichiers temp d'un run précédent
        for c in chemins_tmp:
            if os.path.exists(c):
                os.remove(c)

        compteur_it  = mp.Value('i', 0)

        workers_proc = []
        for w_id, nb_it_w in enumerate(its_par_worker):
            p = mp.Process(
                target = _worker_hu_mp,
                args   = (pb, gb, nb_it_w, STACK_DEPART, offsets[w_id],
                          compteur_it, chemins_tmp[w_id], w_id),
            )
            p.daemon = True
            workers_proc.append(p)

        for p in workers_proc:
            p.start()

        # ── Barre de progression ────────────────────────────────────────
        derniere_val = -1
        while any(p.is_alive() for p in workers_proc):
            val_actuelle = compteur_it.value
            if val_actuelle != derniere_val:
                _barre_progression(
                    val_actuelle, nb_iterations_par_niveau,
                    0, t_niveau,
                )
                derniere_val = val_actuelle
            time.sleep(5)

        # Affichage 100% final
        _barre_progression(
            nb_iterations_par_niveau, nb_iterations_par_niveau,
            0, t_niveau,
        )
        print()  # saut de ligne après la barre

        for p in workers_proc:
            p.join()

        # ── Lire et fusionner les fichiers temp ─────────────────────────
        liste_noeuds = []
        if mccfr.noeuds:
            liste_noeuds.append(mccfr.noeuds)
        for c in chemins_tmp:
            if os.path.exists(c):
                with open(c, 'rb') as f:
                    liste_noeuds.append(pickle.load(f))
                os.remove(c)  # nettoyer après lecture

        mccfr.noeuds     = _fusionner_noeuds(liste_noeuds)
        mccfr.iterations += nb_iterations_par_niveau

        duree_niv  = time.time() - t_niveau
        it_per_sec = nb_iterations_par_niveau / max(duree_niv, 1)
        sauvegarder_blueprint(mccfr.noeuds, chemin)
        print(f"  Sauvegarde : {len(mccfr.noeuds):,} infosets | "
              f"{it_per_sec:.0f} it/s | {duree_niv:.0f}s")

    # ── Sauvegarde finale ───────────────────────────────────────────────
    sauvegarder_blueprint(mccfr.noeuds, chemin)
    duree_totale = time.time() - t_debut_global

    print(f"\n{'='*62}")
    print(f"  Blueprint HU entraine avec succes !")
    print(f"  Infosets   : {len(mccfr.noeuds):,}")
    print(f"  Duree      : {duree_totale/60:.1f} min")
    print(f"  Sauvegarde : {chemin}")
    print(f"{'='*62}\n")

    if verbose:
        mccfr.afficher_stats()

    return mccfr


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == '__main__':
    mp.freeze_support()   # obligatoire sur Windows (spawn)

    parser = argparse.ArgumentParser(
        description="Entraine le blueprint MCCFR Heads-Up d'AXIOM"
    )
    parser.add_argument(
        '--iterations',
        type    = int,
        default = 75_000,
        help    = "Iterations MCCFR par niveau de blindes (defaut: 75000 x 8 = 600k total)"
    )
    parser.add_argument(
        '--chemin',
        type    = str,
        default = CHEMIN_HU,
        help    = f"Chemin de sauvegarde du blueprint (defaut: {CHEMIN_HU})"
    )
    parser.add_argument(
        '--verbose',
        action  = 'store_true',
        default = True,
        help    = "Afficher les stats apres chaque niveau (active par defaut)"
    )
    parser.add_argument(
        '--no-verbose',
        dest    = 'verbose',
        action  = 'store_false',
        help    = "Desactiver l'affichage detaille"
    )
    parser.add_argument(
        '--from-scratch',
        action  = 'store_true',
        default = False,
        help    = "Reentrainer depuis zero (ignore le fichier existant)"
    )
    parser.add_argument(
        '--workers',
        type    = int,
        default = None,
        help    = f"Nombre de coeurs CPU (defaut: {NB_WORKERS_MAX})"
    )
    args = parser.parse_args()

    entrainer_blueprint_hu(
        nb_iterations_par_niveau = args.iterations,
        chemin                   = args.chemin,
        verbose                  = args.verbose,
        reprendre                = not args.from_scratch,
        nb_workers               = args.workers,
    )
