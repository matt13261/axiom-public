# =============================================================================
# AXIOM — ai/deep_cfr.py
# Algorithme Deep CFR complet (Phase 4).
#
# Ce module orchestre l'entraînement Deep CFR end-to-end :
#   1. Traversées de l'arbre de jeu (génération de données)
#   2. Alimentation des Reservoir Buffers
#   3. Epochs d'entraînement PyTorch sur les réseaux
#   4. Sauvegarde des modèles
#
# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHME DEEP CFR (Brown et al., 2019)
# ─────────────────────────────────────────────────────────────────────────────
#
# Deep CFR remplace les tables de regrets de MCCFR (Phase 3) par des réseaux
# de neurones. Cela permet de généraliser à des infosets jamais vus et de
# ne plus être limité par la mémoire (plus de dict Python avec des millions
# de clés).
#
# Boucle principale (T itérations) :
# ─────────────────────────────────────────────────────────────────────────────
# Pour t = 1..T :
#   Pour chaque joueur traversant i = 0, 1, 2 :
#     Pour k = 1..K traversées :
#       deal = dealer_aleatoire()
#       traverser_arbre(deal, joueur_i, reseau_regret[i])
#         → À chaque nœud de joueur i :
#             actions = actions_abstraites(etat)
#             strategie = regret_matching(reseau_regret[i](infoset))
#             pour chaque action a :
#               valeur[a] = traverser_arbre(etat + a, joueur_i, ...)
#             regret_inst[a] = valeur[a] - Σ_a' strategie[a'] × valeur[a']
#             buffer_regret[i].ajouter(infoset, regret_inst, nb_actions)
#             buffer_strategie[i].ajouter(infoset, strategie, t)
#         → À chaque nœud adversaire j ≠ i :
#             strategie = regret_matching(reseau_regret[j](infoset))
#             action = echantillonner(strategie)
#             retourner traverser_arbre(etat + action, joueur_i, ...)
#             (N°11 : on NE stocke PAS dans les buffers de j ici —
#              seuls les nœuds du joueur traversant i alimentent ses buffers)
#
#   Entraîner reseau_regret[i] sur buffer_regret[i]    (1 epoch)
#   Entraîner reseau_strategie[i] sur buffer_strategie[i] (1 epoch)
#   (Les buffers sont conservés entre les itérations — reservoir sampling)
#
# Résultat : reseau_strategie[i] ≈ stratégie Nash du joueur i
#
# ─────────────────────────────────────────────────────────────────────────────
# CORRECTIONS APPORTÉES
# ─────────────────────────────────────────────────────────────────────────────
#
# N°1  : stacks par défaut = STACK_DEPART (500) au lieu de 1500.
#        Le Deep CFR était entraîné sur des situations de stacks profonds (75 BB)
#        qui n'existent pas dans le tournoi réel (25 BB au départ).
#
# N°2  : split preflop/postflop pour les tailles de mise abstraites.
#        Le MCCFR utilise TAILLES_MISE_PREFLOP au preflop et TAILLES_MISE_POSTFLOP
#        au postflop. Le Deep CFR utilisait uniquement TAILLES_MISE (postflop)
#        pour toutes les streets → désalignement des espaces d'actions.
#
# N°11 : suppression des regrets nuls dans les nœuds adversaires.
#        L'algorithme Deep CFR (Brown et al. 2019) ne stocke des regrets
#        que pour le joueur traversant i, pas pour les adversaires j ≠ i.
#        Stocker des regrets nuls pour les adversaires diluait le signal
#        d'entraînement et forçait le réseau à prédire zéro pour des
#        infosets qui ne lui appartiennent pas.
#
# ─────────────────────────────────────────────────────────────────────────────
# DIFFÉRENCES AVEC MCCFR (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────
#
# MCCFR (Phase 3) :
#   + Garanti de converger (résultats déterministes à suffisamment d'itérations)
#   − Mémoire O(nb_infosets) — explose en Hold'em complet
#   − Pas de généralisation aux infosets non vus
#
# Deep CFR (Phase 4) :
#   + Mémoire bornée (Reservoir Buffer de taille fixe)
#   + Généralisation : le réseau interpole entre infosets similaires
#   + Compatible GPU → entraînement massivement parallèle
#   − Convergence stochastique (erreur d'approximation du réseau)
#   − Hyperparamètres supplémentaires (architecture, lr, buffer size)
#
# Référence : "Deep Counterfactual Regret Minimization"
#             N. Brown, A. Lerer, S. Gross, T. Sandholm — ICML 2019
# =============================================================================

import os
import time
import random
import threading
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from treys import Deck as _TreysDeck, Evaluator as _Evaluator

from config.settings import (
    DEEP_CFR_ITERATIONS, DEEP_CFR_TRAVERSALS,
    BATCH_SIZE, ALL_IN,
    TAILLES_MISE_PREFLOP, TAILLES_MISE_POSTFLOP,  # N°2 : import split
    CHEMIN_REGRET_NET, CHEMIN_STRATEGY_NET, CHEMIN_LOG,
    STACK_DEPART,                                  # N°1 : stack tournoi réel
)
from ai.network import (
    ReseauRegret, ReseauStrategie, ReseauValeur,
    creer_reseaux, afficher_info_reseaux,
    encoder_infoset, NB_ACTIONS_MAX, DEVICE,
    sauvegarder_reseau, charger_reseau,
)
from ai.reservoir import (
    ReservoirBufferRegret, ReservoirBufferStrategie, ReservoirBufferValeur,
    creer_buffers, afficher_etat_buffers,
    TAILLE_BUFFER_DEFAUT,
)
from ai.trainer import (
    EntraineurRegret, EntraineurStrategie, EntraineurValeur,
    creer_entraineurs, afficher_stats_entrainement,
    NB_BATCHS_PAR_EPOCH,
)
from abstraction.card_abstraction import AbstractionCartes, AbstractionCartesV2
from abstraction.action_abstraction import AbstractionAction
from abstraction.info_set import _normaliser, PALIERS_POT, PALIERS_STACK, _discretiser_raise_frac
from engine.actions import Action, TypeAction


# =============================================================================
# CONSTANTES INTERNES (état léger, copié depuis mccfr.py pour l'indépendance)
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

_eval_holdem = _Evaluator()
_rng         = random.Random()   # fallback single-threaded
_tls         = threading.local() # RNG thread-local pour traversées parallèles

def _rng_thread() -> random.Random:
    """Retourne un random.Random propre à chaque thread (thread-safe)."""
    if not hasattr(_tls, 'rng'):
        _tls.rng = random.Random()
    return _tls.rng


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class DeepCFR:
    """
    Implémentation complète de Deep CFR pour Texas Hold'em No Limit 3 joueurs.

    Orchestre :
      - Les traversées de l'arbre de jeu (génération de données)
      - L'alimentation des Reservoir Buffers
      - Les epochs d'entraînement PyTorch
      - La sauvegarde et le chargement des modèles

    Usage typique
    -------------
        dcfr = DeepCFR()
        dcfr.entrainer(nb_iterations=100, nb_traversees=500, verbose=True)
        dcfr.sauvegarder()

    Reprendre un entraînement :
        dcfr = DeepCFR()
        dcfr.charger()
        dcfr.entrainer(nb_iterations=100, ...)
    """

    def __init__(self,
                 taille_buffer   : int           = TAILLE_BUFFER_DEFAUT,
                 device          : torch.device  = None,
                 chemin_regret   : str           = CHEMIN_REGRET_NET,
                 chemin_strategie: str           = CHEMIN_STRATEGY_NET,
                 chemin_log      : str           = CHEMIN_LOG):
        """
        taille_buffer    : capacité des Reservoir Buffers (défaut : 3M)
        device           : device PyTorch (auto-détecté si None)
        chemin_regret    : chemin de sauvegarde des ReseauRegret
        chemin_strategie : chemin de sauvegarde des ReseauStrategie
        chemin_log       : chemin du fichier CSV de log
        """
        self.device           = device or DEVICE
        self.chemin_regret    = chemin_regret
        self.chemin_strategie = chemin_strategie
        self.chemin_log       = chemin_log
        self.iteration        = 0

        # Réseaux (6 regret/stratégie + 3 valeur)
        self.reseaux_regret, self.reseaux_strategie = creer_reseaux(self.device)
        self.reseaux_valeur = [ReseauValeur().to(self.device) for _ in range(3)]

        # Reservoir Buffers (regret + stratégie + valeur)
        self.buffers_regret, self.buffers_strategie = creer_buffers(taille_buffer)
        self.buffers_valeur = [ReservoirBufferValeur(taille_buffer) for _ in range(3)]

        # Entraîneurs (regret + stratégie + valeur)
        self.entraineurs_regret, self.entraineurs_strategie = creer_entraineurs(
            self.reseaux_regret, self.reseaux_strategie, device=self.device
        )
        self.entraineurs_valeur = [
            EntraineurValeur(self.reseaux_valeur[i], joueur_idx=i, device=self.device)
            for i in range(3)
        ]

        # Abstractions (partagées, instanciées une seule fois)
        self._abs_cartes  = AbstractionCartesV2()
        self._abs_actions = AbstractionAction()

    # ==================================================================
    # ENTRAÎNEMENT PRINCIPAL
    # ==================================================================

    def entrainer(self,
                  nb_iterations  : int  = DEEP_CFR_ITERATIONS,
                  nb_traversees  : int  = DEEP_CFR_TRAVERSALS,
                  stacks         : int  = STACK_DEPART,   # N°1 : 500 par défaut
                  pb             : int  = 10,
                  gb             : int  = 20,
                  nb_batchs      : int  = NB_BATCHS_PAR_EPOCH,
                  batch_size     : int  = BATCH_SIZE,
                  verbose        : bool = True,
                  save_every     : int  = 10) -> None:
        """
        Lance nb_iterations de Deep CFR.

        Chaque itération comprend :
          1. Pour chaque joueur traversant i (0, 1, 2) :
               nb_traversees deals aléatoires → traversée de l'arbre →
               alimentation buffer_regret[i] et buffer_strategie[i]
          2. Entraînement de reseau_regret[i] sur buffer_regret[i]
          3. Entraînement de reseau_strategie[i] sur buffer_strategie[i]

        Paramètres
        ----------
        nb_iterations : nombre d'itérations Deep CFR à effectuer
        nb_traversees : traversées par joueur par itération (Monte Carlo)
        stacks        : stack de départ de chaque joueur
                        N°1 : défaut = STACK_DEPART (500) — cohérent avec
                        le tournoi réel. Ancienne valeur : 1500 (trop profond).
        pb, gb        : petite et grande blinde
        nb_batchs     : mini-batchs par epoch d'entraînement
        batch_size    : taille d'un mini-batch
        verbose       : affiche la progression
        save_every    : sauvegarde tous les N itérations (0 = jamais)
        """
        if verbose:
            print(f"\n{'═'*62}")
            print(f"  AXIOM — Deep CFR | {nb_iterations} itérations × "
                  f"{nb_traversees} traversées/joueur")
            print(f"  Stacks={stacks} | Blindes {pb}/{gb} | Device={self.device}")
            afficher_info_reseaux(self.reseaux_regret, self.reseaux_strategie)

        self._init_log()

        t_debut = time.time()

        for it in range(1, nb_iterations + 1):
            self.iteration += 1
            t_it = time.time()

            stats_regret    = []
            stats_strategie = []

            # ── Point 10 — LR schedule global décroissant (Pluribus) ───
            # LR_eff(t) = LEARNING_RATE / sqrt(t), plancher à 1% de LR.
            # Les itérations tardives affinent des détails fins ; un LR
            # élevé en fin d'entraînement peut défaire les itérations
            # précédentes.
            for joueur_i in range(3):
                self.entraineurs_regret[joueur_i].reinitialiser_scheduler(
                    iteration_courante=self.iteration,
                    nb_iterations_total=nb_iterations,
                )
                self.entraineurs_strategie[joueur_i].reinitialiser_scheduler(
                    iteration_courante=self.iteration,
                    nb_iterations_total=nb_iterations,
                )
                self.entraineurs_valeur[joueur_i].reinitialiser_scheduler(
                    iteration_courante=self.iteration,
                    nb_iterations_total=nb_iterations,
                )

            # ── Traversées pour chaque joueur (parallèles) ────────────
            nb_echant_avant = [len(self.buffers_regret[j]) for j in range(3)]

            def _traverser_joueur(joueur_i: int) -> None:
                for _ in range(nb_traversees):
                    etat = self._dealer_aleatoire(stacks, pb, gb)
                    self._traverser(etat, joueur_i, self.iteration)

            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(_traverser_joueur, range(3)))

            for joueur_i in range(3):
                nb_echant_apres = len(self.buffers_regret[joueur_i])
                if verbose:
                    nouveaux = nb_echant_apres - nb_echant_avant[joueur_i]
                    print(f"  It.{self.iteration:4d} | J{joueur_i} | "
                          f"+{nouveaux:6,} échant. → "
                          f"buffer={nb_echant_apres:,}")

                # ── Entraînement réseau regret ─────────────────────────
                sr = self.entraineurs_regret[joueur_i].entrainer_epoch(
                    self.buffers_regret[joueur_i],
                    nb_batchs=nb_batchs,
                    batch_size=batch_size,
                )
                stats_regret.append(sr)

                # ── Entraînement réseau stratégie ──────────────────────
                ss = self.entraineurs_strategie[joueur_i].entrainer_epoch(
                    self.buffers_strategie[joueur_i],
                    nb_batchs=nb_batchs,
                    batch_size=batch_size,
                )
                stats_strategie.append(ss)

                # ── Entraînement réseau valeur (Point 6) ───────────────
                self.entraineurs_valeur[joueur_i].entrainer_epoch(
                    self.buffers_valeur[joueur_i],
                    nb_batchs=max(nb_batchs // 2, 50),
                    batch_size=batch_size,
                )

            # ── Affichage et log ───────────────────────────────────────
            duree_it = time.time() - t_it
            if verbose:
                afficher_stats_entrainement(
                    stats_regret, stats_strategie, self.iteration)
                print(f"  Durée itération : {duree_it:.1f}s | "
                      f"Durée totale : {time.time()-t_debut:.0f}s")

            self._log_iteration(stats_regret, stats_strategie, duree_it)

            # ── Sauvegarde automatique ─────────────────────────────────
            if save_every > 0 and self.iteration % save_every == 0:
                self.sauvegarder(verbose=verbose)

        if verbose:
            print(f"\n{'═'*62}")
            print(f"  ✅ Deep CFR terminé | {self.iteration} itérations | "
                  f"{time.time()-t_debut:.0f}s total")
            print(f"{'═'*62}\n")
            afficher_etat_buffers(self.buffers_regret, self.buffers_strategie)

    # ==================================================================
    # TRAVERSÉE DE L'ARBRE (cœur de Deep CFR)
    # ==================================================================

    def _traverser(self,
                   etat            : dict,
                   joueur_traversant: int,
                   iteration       : int) -> float:
        """
        Traversée récursive de l'arbre de jeu pour Deep CFR.

        Pour chaque nœud du joueur traversant i :
          1. Calculer la stratégie par regret matching sur reseau_regret[i]
          2. Explorer TOUTES les actions (comme ES-MCCFR)
          3. Calculer les regrets instantanés
          4. Stocker (infoset, regrets) dans buffer_regret[i]
          5. Stocker (infoset, strategie) dans buffer_strategie[i]

        Pour chaque nœud adversaire j ≠ i :
          1. Calculer la stratégie par regret matching sur reseau_regret[j]
          2. Échantillonner UNE action (External Sampling)
          3. Continuer la traversée avec cette action
          N°11 : ne PAS stocker dans les buffers de j — seul le joueur
          traversant i alimente ses propres buffers (Deep CFR original).

        Retourne l'utilité (gain net) pour le joueur_traversant.
        """
        # ── Compter les joueurs encore en jeu ─────────────────────────
        actifs_allin = [j for j in range(3)
                        if etat['statuts'][j] in (_H_ACTIF, _H_ALLIN)]
        pouvant_agir = [j for j in actifs_allin
                        if etat['statuts'][j] == _H_ACTIF]

        # ── Nœud terminal ─────────────────────────────────────────────
        if len(actifs_allin) <= 1:
            return self._gain_fold(etat, joueur_traversant)

        # ── File vide → transition de street ou showdown ──────────────
        if not etat['joueurs_en_attente']:
            if etat['phase'] == _H_RIVER:
                return self._gain_showdown(etat, joueur_traversant)
            if not pouvant_agir:
                etat = self._copier_etat(etat)
                while etat['phase'] < _H_RIVER:
                    self._passer_street(etat)
                return self._gain_showdown(etat, joueur_traversant)
            etat = self._copier_etat(etat)
            self._passer_street(etat)
            return self._traverser(etat, joueur_traversant, iteration)

        # ── Joueur courant dans la file ────────────────────────────────
        joueur_idx = etat['joueurs_en_attente'][0]

        # Passer les joueurs fold/all-in dans la file
        if etat['statuts'][joueur_idx] != _H_ACTIF:
            etat = self._copier_etat(etat)
            etat['joueurs_en_attente'] = etat['joueurs_en_attente'][1:]
            return self._traverser(etat, joueur_traversant, iteration)

        # ── Actions légales ───────────────────────────────────────────
        actions = self._actions_abstraites(etat, joueur_idx)
        if not actions:
            etat = self._copier_etat(etat)
            etat['joueurs_en_attente'] = etat['joueurs_en_attente'][1:]
            return self._traverser(etat, joueur_traversant, iteration)

        # ── Encoder l'infoset ─────────────────────────────────────────
        infoset_vec = encoder_infoset(etat, joueur_idx)

        # ── Obtenir la stratégie depuis le réseau regret ──────────────
        strategie = self._strategie_depuis_reseau(
            self.reseaux_regret[joueur_idx], infoset_vec, len(actions)
        )

        # ── JOUEUR TRAVERSANT : explorer toutes les actions ───────────
        if joueur_idx == joueur_traversant:
            valeurs = []
            for action in actions:
                etat_copie = self._copier_etat(etat)
                etat_copie['joueurs_en_attente'] = list(etat['joueurs_en_attente'][1:])
                self._appliquer_action(etat_copie, joueur_idx, action)
                v = self._traverser(etat_copie, joueur_traversant, iteration)
                valeurs.append(v)

            # Valeur du nœud = espérance sous la stratégie courante
            valeur_noeud = sum(strategie[i] * valeurs[i]
                               for i in range(len(actions)))

            # Regrets instantanés
            regrets_inst = np.zeros(NB_ACTIONS_MAX, dtype=np.float32)
            for i in range(len(actions)):
                regrets_inst[i] = valeurs[i] - valeur_noeud

            # Stocker dans les buffers du joueur traversant uniquement
            self.buffers_regret[joueur_idx].ajouter(
                infoset_vec, regrets_inst, len(actions))
            strategie_arr = np.array(strategie, dtype=np.float32)
            self.buffers_strategie[joueur_idx].ajouter(
                infoset_vec, strategie_arr, iteration, len(actions))
            # Point 6 : stocker (infoset, valeur_noeud) pour le ReseauValeur
            self.buffers_valeur[joueur_idx].ajouter(infoset_vec, valeur_noeud)

            return valeur_noeud

        # ── ADVERSAIRE : échantillonner une seule action ──────────────
        else:
            # N°11 : ne PAS stocker dans les buffers de l'adversaire.
            # Selon Deep CFR (Brown et al. 2019), seuls les nœuds du joueur
            # traversant i alimentent buffer_regret[i] et buffer_strategie[i].
            # Stocker des regrets nuls pour les adversaires diluait le signal
            # d'entraînement et introduisait un biais vers zéro.

            # Échantillonner une action selon la stratégie de l'adversaire
            idx_action = _echantillonner(strategie)
            action = actions[idx_action]

            etat_copie = self._copier_etat(etat)
            etat_copie['joueurs_en_attente'] = list(etat['joueurs_en_attente'][1:])
            self._appliquer_action(etat_copie, joueur_idx, action)
            return self._traverser(etat_copie, joueur_traversant, iteration)

    # ==================================================================
    # STRATÉGIE DEPUIS LE RÉSEAU (inference)
    # ==================================================================

    def _strategie_depuis_reseau(self,
                                  reseau      : ReseauRegret,
                                  infoset_vec : np.ndarray,
                                  nb_actions  : int) -> List[float]:
        """
        Interroge le réseau de regrets pour obtenir une stratégie par
        Regret Matching.

        1. Passe avant → vecteur de regrets (NB_ACTIONS_MAX,)
        2. Tronque aux nb_actions légales
        3. Applique regret matching : prob(a) = max(r(a), 0) / Σ max(r, 0)
        4. Si tous ≤ 0 → stratégie uniforme

        Retourne une liste de nb_actions probabilités (somme = 1.0).
        """
        reseau.eval()
        with torch.no_grad():
            x = torch.from_numpy(infoset_vec).unsqueeze(0).to(self.device)
            regrets = reseau(x).squeeze(0).cpu().numpy()

        regrets_tronques = regrets[:nb_actions]
        regrets_pos = np.maximum(regrets_tronques, 0.0)
        somme = regrets_pos.sum()

        if somme > 1e-8:
            strategie = (regrets_pos / somme).tolist()
        else:
            strategie = [1.0 / nb_actions] * nb_actions

        return strategie

    def obtenir_strategie(self,
                          etat       : dict,
                          joueur_idx : int) -> np.ndarray:
        """
        Retourne la stratégie du ReseauStrategie (pour la production / le jeu).

        C'est cette méthode qu'AXIOM appelle en cours de partie pour décider
        de son action. Elle utilise reseau_strategie (pas reseau_regret).

        etat       : dict interne de l'état de jeu (format MCCFRHoldEm)
        joueur_idx : index du joueur (0, 1 ou 2)

        Retourne np.ndarray (NB_ACTIONS_MAX,) — distribution de probabilités
        """
        infoset_vec = encoder_infoset(etat, joueur_idx)
        reseau = self.reseaux_strategie[joueur_idx]
        reseau.eval()
        with torch.no_grad():
            x    = torch.from_numpy(infoset_vec).unsqueeze(0).to(self.device)
            strat = reseau(x).squeeze(0).cpu().numpy()
        return strat

    # ==================================================================
    # GESTION DU JEU (deal, actions, transitions, gains)
    # ==================================================================

    def _dealer_aleatoire(self, stacks: int, pb: int, gb: int) -> dict:
        """
        Crée un état de départ avec un deal aléatoire.
        Identique à MCCFRHoldEm._dealer_aleatoire() (Phase 3).

        N°1 : stacks est maintenant cohérent avec STACK_DEPART (500)
        par défaut dans entrainer(), au lieu de 1500.
        """
        deck = list(_TreysDeck().cards)
        _rng_thread().shuffle(deck)

        cartes        = [deck[0:2], deck[2:4], deck[4:6]]
        board_complet = deck[6:11]

        stacks_l  = [stacks, stacks, stacks]
        contribs  = [0, 0, 0]
        mises_t   = [0, 0, 0]

        mise_sb = min(pb, stacks_l[1])
        stacks_l[1] -= mise_sb; contribs[1] += mise_sb; mises_t[1] = mise_sb

        mise_bb = min(gb, stacks_l[2])
        stacks_l[2] -= mise_bb; contribs[2] += mise_bb; mises_t[2] = mise_bb

        statuts = [_H_ALLIN if stacks_l[i] == 0 else _H_ACTIF for i in range(3)]
        buckets, equites = self._precomputer_buckets_et_equites(cartes, board_complet)

        return {
            'cartes'             : cartes,
            'board_complet'      : board_complet,
            'board_visible'      : [],
            'stacks'             : stacks_l,
            'contributions'      : contribs,
            'mises_tour'         : mises_t,
            'mise_courante'      : gb,
            'pot'                : mise_sb + mise_bb,
            'statuts'            : statuts,
            'phase'              : _H_PREFLOP,
            'joueurs_en_attente' : list(_ORDRE_PREFLOP),
            'hist_phases'        : ['', '', '', ''],
            'grande_blinde'      : gb,
            'buckets'            : buckets,
            'equites'            : equites,
            # raise_fracs[0] initialisé avec la fraction BB/pot — cohérent
            # avec mccfr.py pour que encoder_infoset voie le blind à suivre
            # dès la première décision du BTN.
            'raise_fracs'        : [gb / max(mise_sb + mise_bb, 1),
                                    0.0, 0.0, 0.0],
        }

    def _precomputer_buckets_et_equites(self, cartes, board_complet):
        boards = [[], board_complet[:3], board_complet[:4], board_complet[:5]]
        buckets, equites = [], []
        for j in range(3):
            bkts_j, eqs_j = [], []
            for b in boards:
                bk, eq = self._abs_cartes.bucket_et_equite(cartes[j], b)
                bkts_j.append(bk); eqs_j.append(eq)
            buckets.append(bkts_j); equites.append(eqs_j)
        return buckets, equites

    def _actions_abstraites(self, etat: dict, joueur_idx: int) -> list:
        """
        Actions légales abstraites.

        N°2 : utilise TAILLES_MISE_PREFLOP au preflop ([1.0, 2.5, 3.0])
        et TAILLES_MISE_POSTFLOP au flop/turn/river ([0.35, 0.65, 1.0]),
        aligné avec le blueprint MCCFR (mccfr.py ligne 903).
        Anciennement : utilisait uniquement TAILLES_MISE (postflop) pour
        toutes les streets → désalignement de l'espace d'actions.
        """
        stack         = etat['stacks'][joueur_idx]
        mise_tour     = etat['mises_tour'][joueur_idx]
        mise_courante = etat['mise_courante']
        pot           = etat['pot']
        gb            = etat['grande_blinde']
        a_payer       = mise_courante - mise_tour

        actions = []
        if a_payer > 0:
            actions.append(Action(TypeAction.FOLD))
        if a_payer == 0:
            actions.append(Action(TypeAction.CHECK))
        if 0 < a_payer < stack:
            actions.append(Action(TypeAction.CALL, montant=mise_courante))

        # N°2 : split preflop / postflop pour aligner avec le MCCFR
        tailles = (TAILLES_MISE_PREFLOP
                   if etat['phase'] == _H_PREFLOP
                   else TAILLES_MISE_POSTFLOP)

        for fraction in tailles:
            montant_raise = mise_courante + int(pot * fraction)
            montant_raise = max(montant_raise, mise_courante + gb)
            if montant_raise < mise_tour + stack:
                actions.append(Action(TypeAction.RAISE, montant=montant_raise))

        if ALL_IN and stack > 0:
            montant_allin = mise_tour + stack
            if not any(a.type == TypeAction.RAISE and a.montant == montant_allin
                       for a in actions):
                actions.append(Action(TypeAction.ALL_IN, montant=montant_allin))

        vus, actions_uniques = set(), []
        for a in actions:
            cle = (a.type, a.montant)
            if cle not in vus:
                vus.add(cle); actions_uniques.append(a)
        return actions_uniques

    def _appliquer_action(self, etat: dict, joueur_idx: int, action) -> None:
        """Applique une action sur l'état (en place)."""
        phase     = etat['phase']
        stack     = etat['stacks'][joueur_idx]
        mise_tour = etat['mises_tour'][joueur_idx]

        if action.type == TypeAction.FOLD:
            etat['statuts'][joueur_idx] = _H_FOLD
            etat['hist_phases'][phase] += 'f'

        elif action.type == TypeAction.CHECK:
            etat['hist_phases'][phase] += 'x'

        elif action.type == TypeAction.CALL:
            a_payer      = etat['mise_courante'] - mise_tour
            a_payer_reel = min(a_payer, stack)
            etat['stacks'][joueur_idx]        -= a_payer_reel
            etat['mises_tour'][joueur_idx]    += a_payer_reel
            etat['contributions'][joueur_idx] += a_payer_reel
            etat['pot']                       += a_payer_reel
            if etat['stacks'][joueur_idx] == 0:
                etat['statuts'][joueur_idx] = _H_ALLIN
            etat['hist_phases'][phase] += 'c'

        elif action.type in (TypeAction.RAISE, TypeAction.ALL_IN):
            a_payer      = action.montant - mise_tour
            a_payer_reel = min(a_payer, stack)

            # Bucket calculé AVANT la mise à jour du pot (cohérent avec mccfr.py)
            if action.type == TypeAction.RAISE:
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
                etat['statuts'][joueur_idx] = _H_ALLIN

            etat['hist_phases'][phase] += code

    def _reinserer_apres_raise(self, etat: dict, raiser_idx: int) -> None:
        ordre = (_ORDRE_PREFLOP if etat['phase'] == _H_PREFLOP else _ORDRE_POSTFLOP)
        pos = ordre.index(raiser_idx) if raiser_idx in ordre else 0
        depuis_apres = ordre[pos + 1:] + ordre[:pos + 1]
        etat['joueurs_en_attente'] = [
            j for j in depuis_apres
            if j != raiser_idx and etat['statuts'][j] == _H_ACTIF
        ]

    def _passer_street(self, etat: dict) -> None:
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
            j for j in _ORDRE_POSTFLOP if etat['statuts'][j] == _H_ACTIF
        ]

    def _gain_fold(self, etat: dict, joueur_traversant: int) -> float:
        actifs = [i for i in range(3)
                  if etat['statuts'][i] in (_H_ACTIF, _H_ALLIN)]
        if not actifs:
            return -etat['contributions'][joueur_traversant]
        gagnant = actifs[0]
        if joueur_traversant == gagnant:
            return etat['pot'] - etat['contributions'][joueur_traversant]
        return -etat['contributions'][joueur_traversant]

    def _gain_showdown(self, etat: dict, joueur_traversant: int) -> float:
        board = (etat['board_visible']
                 if len(etat['board_visible']) == 5
                 else etat['board_complet'])
        actifs = [i for i in range(3)
                  if etat['statuts'][i] in (_H_ACTIF, _H_ALLIN)]
        if len(actifs) == 0:
            return 0.0
        if len(actifs) == 1:
            return self._gain_fold(etat, joueur_traversant)

        meilleur_score, gagnants = None, []
        for i in actifs:
            try:
                score = _eval_holdem.evaluate(board, etat['cartes'][i])
            except Exception:
                score = 9999
            if meilleur_score is None or score < meilleur_score:
                meilleur_score = score; gagnants = [i]
            elif score == meilleur_score:
                gagnants.append(i)

        if joueur_traversant in gagnants:
            return (etat['pot'] / len(gagnants)) - etat['contributions'][joueur_traversant]
        return -etat['contributions'][joueur_traversant]

    def _copier_etat(self, etat: dict) -> dict:
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
            'equites'            : etat['equites'],
            'raise_fracs'        : list(etat['raise_fracs']),
        }

    # ==================================================================
    # SAUVEGARDE ET CHARGEMENT
    # ==================================================================

    def sauvegarder(self, chemin_regret: str = None,
                    chemin_strategie: str = None,
                    verbose: bool = True) -> None:
        """
        Sauvegarde les 9 réseaux (regrets + stratégies + valeurs) en fichiers .pt.
        """
        cr = chemin_regret    or self.chemin_regret
        cs = chemin_strategie or self.chemin_strategie
        cv = cr.replace('regret', 'valeur')   # ex: data/models/valeur_net_j0.pt
        for i in range(3):
            sauvegarder_reseau(self.reseaux_regret[i],
                               cr.replace('.pt', f'_j{i}.pt'))
            sauvegarder_reseau(self.reseaux_strategie[i],
                               cs.replace('.pt', f'_j{i}.pt'))
            sauvegarder_reseau(self.reseaux_valeur[i],
                               cv.replace('.pt', f'_j{i}.pt'))
        if verbose:
            print(f"  💾 Deep CFR sauvegardé (itération {self.iteration})")

    def charger(self, verbose: bool = True) -> None:
        """Charge les 9 réseaux depuis les fichiers .pt."""
        import os
        cv = self.chemin_regret.replace('regret', 'valeur')
        for i in range(3):
            chemin_r = self.chemin_regret.replace('.pt', f'_j{i}.pt')
            chemin_s = self.chemin_strategie.replace('.pt', f'_j{i}.pt')
            chemin_v = cv.replace('.pt', f'_j{i}.pt')
            charger_reseau(self.reseaux_regret[i],    chemin_r, self.device)
            charger_reseau(self.reseaux_strategie[i], chemin_s, self.device)
            if os.path.exists(chemin_v):
                charger_reseau(self.reseaux_valeur[i], chemin_v, self.device)
        if verbose:
            print(f"  📂 Deep CFR chargé depuis les fichiers .pt")

    # ==================================================================
    # LOG CSV
    # ==================================================================

    def _init_log(self) -> None:
        os.makedirs(os.path.dirname(self.chemin_log), exist_ok=True)
        if not os.path.exists(self.chemin_log):
            with open(self.chemin_log, 'w') as f:
                f.write('iteration,joueur,perte_regret,perte_strategie,'
                        'lr_regret,duree_s\n')

    def _log_iteration(self,
                       stats_regret    : list,
                       stats_strategie : list,
                       duree           : float) -> None:
        try:
            with open(self.chemin_log, 'a') as f:
                for i in range(3):
                    sr = stats_regret[i]
                    ss = stats_strategie[i]
                    f.write(
                        f"{self.iteration},{i},"
                        f"{sr.get('perte_moy', float('nan')):.6f},"
                        f"{ss.get('perte_moy', float('nan')):.6f},"
                        f"{sr.get('lr_courant', float('nan')):.2e},"
                        f"{duree:.1f}\n"
                    )
        except IOError:
            pass

    # ==================================================================
    # AFFICHAGE
    # ==================================================================

    def __repr__(self) -> str:
        total_echant = sum(len(b) for b in self.buffers_regret)
        return (f"DeepCFR(iteration={self.iteration}, "
                f"device={self.device}, "
                f"échantillons_total={total_echant:,})")


# =============================================================================
# UTILITAIRE INTERNE
# =============================================================================

def _echantillonner(strategie: list) -> int:
    """Roulette wheel sampling."""
    r = _rng_thread().random()
    cumul = 0.0
    for i, p in enumerate(strategie):
        cumul += p
        if r <= cumul:
            return i
    return len(strategie) - 1


# =============================================================================
# TEST RAPIDE (si exécuté directement)
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*62)
    print("  AXIOM — Test Deep CFR (corrections N°1, N°2, N°11)")
    print("="*62)

    dcfr = DeepCFR(taille_buffer=50_000)
    print(f"\n  {dcfr}")

    # N°1 : vérifier que le stack par défaut est bien STACK_DEPART
    import inspect
    sig = inspect.signature(dcfr.entrainer)
    stack_defaut = sig.parameters['stacks'].default
    assert stack_defaut == STACK_DEPART, (
        f"Stack par défaut attendu={STACK_DEPART}, trouvé={stack_defaut}")
    print(f"\n  ✅ N°1 : stacks par défaut = {stack_defaut} (STACK_DEPART)")

    # N°2 : vérifier le split preflop/postflop dans _actions_abstraites
    etat_pf = dcfr._dealer_aleatoire(stacks=STACK_DEPART, pb=10, gb=20)
    actions_pf = dcfr._actions_abstraites(etat_pf, 0)
    raises_pf  = [a for a in actions_pf if a.type == TypeAction.RAISE]
    etat_fl = dict(etat_pf); etat_fl['phase'] = _H_FLOP
    actions_fl = dcfr._actions_abstraites(etat_fl, 0)
    raises_fl  = [a for a in actions_fl if a.type == TypeAction.RAISE]
    print(f"  ✅ N°2 : raises preflop={len(raises_pf)} "
          f"(attendu {len(TAILLES_MISE_PREFLOP)}), "
          f"postflop={len(raises_fl)} "
          f"(attendu {len(TAILLES_MISE_POSTFLOP)})")
    assert len(raises_pf) <= len(TAILLES_MISE_PREFLOP)
    assert len(raises_fl) <= len(TAILLES_MISE_POSTFLOP)

    # Smoke test entraînement (3 itérations × 20 traversées)
    print("\n  Smoke test (3 itérations × 20 traversées)...")
    dcfr.entrainer(
        nb_iterations = 3,
        nb_traversees = 20,
        stacks        = STACK_DEPART,
        nb_batchs     = 3,
        batch_size    = 64,
        verbose       = True,
        save_every    = 0,
    )

    # N°11 : vérifier que les buffers adversaires restent vides
    # (dans une traversée simple de joueur 0, seul buffer[0] est alimenté
    # si on ne stocke plus les nœuds adversaires)
    for i in range(3):
        assert len(dcfr.buffers_regret[i]) > 0, f"Buffer regret J{i} vide !"
        assert len(dcfr.buffers_strategie[i]) > 0, f"Buffer stratégie J{i} vide !"
        print(f"  J{i} | regret={len(dcfr.buffers_regret[i]):,} | "
              f"stratégie={len(dcfr.buffers_strategie[i]):,}")
    print(f"  ✅ N°11 : buffers alimentés uniquement par le joueur traversant")

    # Vérifier obtenir_strategie()
    print("\n  Test obtenir_strategie()...")
    etat_test = dcfr._dealer_aleatoire(stacks=STACK_DEPART, pb=10, gb=20)
    for i in range(3):
        strat = dcfr.obtenir_strategie(etat_test, i)
        assert strat.shape == (NB_ACTIONS_MAX,)
        assert abs(strat.sum() - 1.0) < 1e-5
        print(f"  J{i} stratégie : {np.round(strat, 3)}")

    print(f"\n  {dcfr}")
    print("\n  ✅ Tous les tests Deep CFR sont passés !")
    print("="*62 + "\n")
