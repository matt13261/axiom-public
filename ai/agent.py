# =============================================================================
# AXIOM — ai/agent.py
# Agent AXIOM : interface entre le moteur de jeu et les IA (Phase 5).
#
# Ce module est le "cerveau en production" d'AXIOM.
# Il assemble les sources de stratégie construites en phases 3, 4 et 11 :
#
#   1. BLUEPRINT HU (MCCFRHeadsUp, Phase 11)
#      → priorité absolue quand il reste 2 joueurs (mode heads-up)
#      → clés format "HU_PREFLOP|pos=P|bucket=B|pot=N|stacks=(A,B)|hist=H"
#      → chargé depuis data/strategy/blueprint_hu.pkl
#
#   2. BLUEPRINT 3-JOUEURS (MCCFR, Phase 3b)
#      → utilisé en mode normal (3 joueurs actifs)
#      → clés format "PREFLOP|pos=P|bucket=B|pot=N|stacks=(a,b,c)|hist=H"
#      → rapide (lookup O(1)), mais généralise mal aux infosets non vus
#
#   3. DEEP CFR (Phase 4)
#      → réseaux PyTorch — généralisent à tout état jamais vu
#      → consultés si le blueprint n'a pas l'infoset courant
#      → légèrement plus lent (forward pass GPU/CPU)
#
# Ordre de priorité en production :
#   2 joueurs non éliminés :
#     → Blueprint HU (si disponible et infoset connu)
#     → Deep CFR
#     → Fallback heuristique
#   3 joueurs non éliminés :
#     → Blueprint 3-joueurs (si disponible et infoset connu)
#     → Deep CFR
#     → Fallback heuristique
#
# ─────────────────────────────────────────────────────────────────────────────
# CONVERSION ÉTAT DE JEU → INFOSET
# ─────────────────────────────────────────────────────────────────────────────
#
# Le moteur de jeu utilise EtatJeu (engine/game_state.py).
# Les IA utilisent deux formats différents :
#
#   Blueprint MCCFR    : clé string via construire_cle_infoset(etat, joueur)
#   Blueprint HU       : clé string via _construire_cle_hu(etat, joueur)
#   Deep CFR           : vecteur float32 via encoder_infoset(etat_dict, joueur_idx)
#
# ─────────────────────────────────────────────────────────────────────────────
# MAPPING ACTIONS ABSTRAITES → ACTIONS CONCRÈTES
# ─────────────────────────────────────────────────────────────────────────────
#
# La stratégie retournée est un vecteur de NB_ACTIONS_MAX probabilités
# sur les actions abstraites :
#   index 0   → FOLD
#   index 1   → CHECK
#   index 2   → CALL
#   index 3–5 → RAISE (3 tailles postflop : 35%, 65%, 100%)
#   index 6   → ALL_IN
#
# L'agent mappe ces probabilités sur les Actions légales concrètes disponibles.
# Si une action abstraite n'est pas légale, sa probabilité est redistribuée.
#
# =============================================================================

import random
import math
import numpy as np
import torch
from typing import Optional, List

from engine.game_state import EtatJeu, Phase, _PHASE_IDX
from engine.player import Joueur, StatutJoueur
from engine.actions import Action, TypeAction
from abstraction.info_set import construire_cle_infoset
from abstraction.card_abstraction import AbstractionCartes, AbstractionCartesV2
from abstraction.info_set import (
    _normaliser, PALIERS_POT, PALIERS_STACK, _discretiser_raise_frac,
    buckets_pseudo_harmonic,
)
from abstraction.action_abstraction import traduire_fraction
from ai.network import encoder_infoset, NB_ACTIONS_MAX, DEVICE
from config.settings import (
    TAILLES_MISE, TAILLES_MISE_PREFLOP, CHEMIN_BLUEPRINT, CHEMIN_BLUEPRINT_HU,
    CHEMIN_REGRET_NET, CHEMIN_STRATEGY_NET,
    PERTURBATION_SIZING,
)
from ai.opponent_tracker import OpponentTracker
from ai.exploit_mixer import ExploitMixer

# Mapping action string → index entier (pour enregistrer_action)
_ACTION_STR_VERS_INT = {
    'FOLD'  : 0,
    'CHECK' : 1,
    'CALL'  : 2,
    'RAISE' : 3,
    'ALLIN' : 8,
    'ALL_IN': 8,
}


# =============================================================================
# CONSTANTES INTERNES
# =============================================================================

# Mapping phase (enum) → index entier (pour encoder_infoset)
_PHASE_VERS_IDX = {
    Phase.PREFLOP  : 0,
    Phase.FLOP     : 1,
    Phase.TURN     : 2,
    Phase.RIVER    : 3,
    Phase.SHOWDOWN : 3,
    Phase.TERMINEE : 3,
}

# Noms de phases pour la clé HU (doit correspondre à _HU_NOM_PHASE de train_hu.py)
_PHASE_NOM_HU = ['HU_PREFLOP', 'HU_FLOP', 'HU_TURN', 'HU_RIVER']

# Mapping TypeAction → code lettre (pour encoder l'historique)
_TYPE_VERS_CODE = {
    TypeAction.FOLD   : 'f',
    TypeAction.CHECK  : 'x',
    TypeAction.CALL   : 'c',
    TypeAction.RAISE  : 'r',
    TypeAction.ALL_IN : 'a',
}

# Index abstraits des familles d'actions (même ordre que AbstractionAction)
_IDX_FOLD       = 0
_IDX_CHECK      = 1
_IDX_CALL       = 2
_IDX_RAISE_BASE = 3                       # indices 3..3+len(TAILLES_MISE)-1
_IDX_ALLIN      = 3 + len(TAILLES_MISE)  # = 8 (TAILLES_MISE = postflop, 5 tailles)

# Tolérance pour considérer qu'une probabilité est nulle
_PROBA_MIN = 1e-9

# Graine RNG globale (reproductible si besoin)
_rng = random.Random()


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class AgentAXIOM:
    """
    Agent de décision AXIOM pour une partie réelle.

    Encapsule trois sources de stratégie :
      - Blueprint HU (MCCFRHeadsUp)   → utilisé en 1v1 (2 joueurs non éliminés)
      - Blueprint 3-joueurs (MCCFR)   → utilisé en partie normale (3 joueurs)
      - Deep CFR (réseaux PyTorch)    → fallback si l'infoset n'est pas dans le blueprint

    Usage typique
    -------------
        agent = AgentAXIOM()
        agent.charger_blueprint('data/strategy/blueprint_v1.pkl')
        agent.charger_blueprint_hu('data/strategy/blueprint_hu.pkl')
        agent.charger_deep_cfr('data/models/strategy_net.pt')
        action = agent.choisir_action(etat, joueur, legales)

    Paramètres de construction
    --------------------------
    mode_deterministe : bool
        True  → prend toujours l'action de plus haute probabilité (argmax)
                 Utile pour les tests et l'évaluation.
        False → échantillonne selon la distribution (comportement par défaut)
                 Rend AXIOM imprévisible — nécessaire en tournoi.
    device : torch.device
        Device PyTorch pour Deep CFR. Auto-détecté si None.
    """

    def __init__(self,
                 mode_deterministe: bool = False,
                 device: torch.device = None):

        self.mode_deterministe = mode_deterministe
        self.device = device or DEVICE

        # ── Sources de stratégie ──────────────────────────────────────────
        self._blueprint:    Optional[dict] = None   # blueprint 3-joueurs (actif)
        self._blueprint_hu: Optional[dict] = None   # blueprint heads-up (2 joueurs)

        # ── Point 2 — Continuation Strategies (Pluribus k=4) ──────────────
        # Dict des variantes chargées : 'baseline', 'fold', 'call', 'raise'.
        # Si > 1 variante chargée, self._blueprint pointe sur celle tirée au
        # hasard au début de chaque nouvelle main (détectée via fingerprint).
        self._blueprints_k : dict = {}
        self._variante_courante : str = 'baseline'
        self._fingerprint_main  : tuple = None
        self._reseaux_strategie: Optional[list] = None  # [ReseauStrategie × 3]
        self._reseaux_valeur:    Optional[list] = None  # [ReseauValeur × 3]
        self._solveur         = None  # SolveurProfondeurLimitee — FLOP
        self._solveur_subgame = None  # SolveurSousJeu — TURN/RIVER (ranges)

        # ── Statistiques de décision (pour diagnostic) ────────────────────
        self.stats = {
            'total'       : 0,
            'blueprint_hu': 0,
            'blueprint'   : 0,
            'deep_cfr'    : 0,
            'solveur_dl'  : 0,   # depth-limited (FLOP)
            'solveur_sg'  : 0,   # subgame solver (TURN/RIVER)
            'heurist'     : 0,
            # Répartition des variantes Pluribus k=4 (par main)
            'variante_baseline': 0,
            'variante_fold'    : 0,
            'variante_call'    : 0,
            'variante_raise'   : 0,
        }

        # ── Abstractions des cartes ───────────────────────────────────────
        # 3-max : équités vs 2 adversaires, seuils calibrés 3-way. Utilisée
        #   pour le Deep CFR (réseaux entraînés en 3-max) et l'heuristique.
        # HU    : équités vs 1 adversaire, seuils calibrés HU. Utilisée pour
        #   la clé du blueprint HU (endgame 2 joueurs) qui a été entraîné
        #   avec la même abstraction.
        self._abs_cartes    = AbstractionCartesV2()
        self._abs_cartes_hu = AbstractionCartesV2()

        # ── OFT — Opponent Frequency Tracker + Exploit Mixer ─────────────
        self.tracker = OpponentTracker()
        self.mixer   = ExploitMixer(self.tracker)

        print(f"  🤖 AgentAXIOM initialisé "
              f"({'déterministe' if mode_deterministe else 'stochastique'}, "
              f"device={self.device})")

    # ==================================================================
    # CHARGEMENT DES SOURCES DE STRATÉGIE
    # ==================================================================

    def charger_blueprint(self, chemin: str = CHEMIN_BLUEPRINT) -> None:
        """Charge la stratégie blueprint MCCFR 3-joueurs depuis un fichier pickle."""
        from ai.strategy import charger_blueprint as _charger
        self._blueprint = _charger(chemin)
        # Enregistrer comme variante 'baseline' pour compatibilité k=4
        self._blueprints_k['baseline'] = self._blueprint
        print(f"  📋 Blueprint 3-joueurs chargé : {len(self._blueprint):,} infosets")

    def charger_blueprints_continuations(self,
                                          dossier: str = 'data/strategy',
                                          base_nom: str = 'blueprint_v1') -> None:
        """
        Charge les 4 variantes Pluribus (baseline + fold + call + raise) pour
        mélange k=4 en temps réel. Variantes manquantes ignorées silencieusement.

        Fichiers attendus dans {dossier}/ :
          - {base_nom}.pkl          (baseline, obligatoire si pas déjà chargé)
          - {base_nom}_fold.pkl
          - {base_nom}_call.pkl
          - {base_nom}_raise.pkl

        À chaque nouvelle main (détectée via fingerprint des cartes), une
        variante est tirée au hasard parmi celles chargées. L'adversaire qui
        apprend à exploiter une variante s'expose aux autres → moins
        exploitable (cf. Pluribus, Brown & Sandholm 2019).
        """
        import os as _os
        from ai.strategy import charger_blueprint as _charger

        variantes_a_charger = [
            ('baseline', f"{dossier}/{base_nom}.pkl"),
            ('fold',     f"{dossier}/{base_nom}_fold.pkl"),
            ('call',     f"{dossier}/{base_nom}_call.pkl"),
            ('raise',    f"{dossier}/{base_nom}_raise.pkl"),
        ]

        for nom, chemin in variantes_a_charger:
            if _os.path.exists(chemin):
                try:
                    self._blueprints_k[nom] = _charger(chemin)
                    print(f"  📋 Blueprint k=4 [{nom:>8}] chargé : "
                          f"{len(self._blueprints_k[nom]):,} infosets")
                except Exception as e:
                    print(f"  ⚠️ Chargement [{nom}] échoué : {e}")
            elif nom != 'baseline':
                # Les biais sont optionnels, seul baseline est critique
                pass

        # Baseline devient le blueprint actif par défaut (comportement
        # identique à l'ancien self.charger_blueprint() si aucune variante
        # biaisée n'existe).
        if 'baseline' in self._blueprints_k:
            self._blueprint = self._blueprints_k['baseline']
            self._variante_courante = 'baseline'

        nb_variantes = len(self._blueprints_k)
        if nb_variantes >= 2:
            print(f"  🎲 Continuations k={nb_variantes} activées "
                  f"({', '.join(sorted(self._blueprints_k))})")

    def _rafraichir_variante_si_nouvelle_main(self, etat: EtatJeu) -> None:
        """
        Détecte une nouvelle main via le fingerprint des cartes de tous les
        joueurs + pot=={sb+bb} préflop. Si la main a changé, tire une
        nouvelle variante parmi self._blueprints_k et l'installe comme
        blueprint actif.

        Ne fait rien si < 2 variantes chargées (k=1, comportement standard).
        """
        if len(self._blueprints_k) < 2:
            return   # pas de mélange k=4

        # Fingerprint = tuple des cartes de chaque joueur (change à chaque
        # nouvelle donne). On inclut le nombre de joueurs pour robustesse.
        fp = tuple(tuple(j.cartes) if j.cartes else () for j in etat.joueurs)
        if fp == self._fingerprint_main:
            return   # même main, on garde la variante courante

        # Nouvelle main → tirer une variante aléatoirement (uniforme)
        self._fingerprint_main   = fp
        self._variante_courante  = _rng.choice(sorted(self._blueprints_k))
        self._blueprint          = self._blueprints_k[self._variante_courante]
        cle_stat = f'variante_{self._variante_courante}'
        if cle_stat in self.stats:
            self.stats[cle_stat] += 1

    def charger_blueprint_hu(self, chemin: str = CHEMIN_BLUEPRINT_HU) -> None:
        """Charge la stratégie blueprint Heads-Up depuis un fichier pickle."""
        from ai.strategy import charger_blueprint as _charger
        self._blueprint_hu = _charger(chemin)
        print(f"  🃏 Blueprint HU chargé : {len(self._blueprint_hu):,} infosets")

    def charger_deep_cfr(self,
                         chemin_strategie: str = CHEMIN_STRATEGY_NET) -> None:
        """
        Charge les 3 réseaux de stratégie Deep CFR (un par joueur).

        chemin_strategie : chemin de base → suffixe _j{0,1,2}.pt ajouté auto.
        """
        from ai.network import ReseauStrategie, charger_reseau

        reseaux = []
        for i in range(3):
            chemin_ji = chemin_strategie.replace('.pt', f'_j{i}.pt')
            net = ReseauStrategie().to(self.device)
            charger_reseau(net, chemin_ji, self.device)
            net.eval()
            reseaux.append(net)

        self._reseaux_strategie = reseaux
        print(f"  🧠 Deep CFR chargé : 3 réseaux de stratégie (device={self.device})")

    def charger_reseaux_valeur(self, chemin_valeur: str = None) -> None:
        """
        Charge les 3 réseaux de valeur Deep CFR (un par joueur).
        Utilisés par le solveur depth-limited comme oracle de valeur aux feuilles.

        chemin_valeur : chemin de base → suffixe _j{0,1,2}.pt ajouté auto.
                        Par défaut : data/models/valeur_net_j{i}.pt
        """
        from ai.network import ReseauValeur, charger_reseau

        base = chemin_valeur or "data/models/valeur_net.pt"
        reseaux = []
        for i in range(3):
            chemin_ji = base.replace('.pt', f'_j{i}.pt')
            net = ReseauValeur().to(self.device)
            charger_reseau(net, chemin_ji, self.device)
            net.eval()
            reseaux.append(net)

        self._reseaux_valeur = reseaux
        print(f"  💎 Réseaux de valeur chargés (device={self.device})")

    def activer_solveur(self,
                        profondeur    : int   = 2,
                        nb_iterations : int   = 50,
                        nb_simul      : int   = 6,
                        nb_scenarios  : int   = 15,
                        temps_max     : float = 3.0) -> None:
        """
        Active les solveurs real-time (Point 2) :
          - SolveurProfondeurLimitee → FLOP (mini-CFR sans range)
          - SolveurSousJeu           → TURN/RIVER (CFR pondéré par ranges adverses)

        profondeur    : profondeur de recherche en streets (FLOP : 2, TURN/RIVER : 1)
        nb_iterations : itérations CFR par solveur
        nb_simul      : simulations Monte Carlo d'équité par feuille
        nb_scenarios  : scénarios de mains adverses tirés par le subgame solver
        temps_max     : budget temps max en secondes par décision
        """
        from solver.depth_limited import SolveurProfondeurLimitee
        from solver.subgame_solver import SolveurSousJeu

        self._solveur = SolveurProfondeurLimitee(
            profondeur      = profondeur,
            nb_iterations   = nb_iterations,
            temps_max       = temps_max,
            nb_simul_equite = nb_simul,
        )
        self._solveur_subgame = SolveurSousJeu(
            nb_scenarios    = nb_scenarios,
            nb_iterations   = nb_iterations,
            temps_max       = temps_max,
            profondeur      = 1,          # TURN/RIVER : une seule street restante
            nb_simul_equite = nb_simul,
        )
        print(f"  🔍 Solveur FLOP activé (profondeur={profondeur}, iter={nb_iterations})")
        print(f"  🎯 Solveur TURN/RIVER activé (scenarios={nb_scenarios}, iter={nb_iterations})")

    # ==================================================================
    # MÉTHODE PRINCIPALE : choisir une action
    # ==================================================================

    def choisir_action(self,
                       etat    : EtatJeu,
                       joueur  : Joueur,
                       legales : List[Action]) -> Action:
        """Choisit la meilleure action pour AXIOM étant donné l'état de jeu."""
        if not legales:
            raise ValueError("Liste d'actions légales vide — impossible de décider.")

        self.stats['total'] += 1
        # Point 2 — Continuation Strategies : à chaque nouvelle main, tirer
        # une variante k=4 (si plusieurs sont chargées). Silencieux sinon.
        self._rafraichir_variante_si_nouvelle_main(etat)
        distribution   = self._obtenir_distribution(etat, joueur)
        # OFT — blend exploit si adversaire actif identifiable
        adv = self._identifier_adversaire_actif(etat, joueur)
        if adv is not None:
            game_type    = self._detecter_game_type(etat)
            distribution = self.mixer.ajuster(distribution, adv, game_type)
        probas_legales = self._mapper_sur_legales(distribution, legales, etat)
        action         = self._selectionner(legales, probas_legales)
        # Point 9 — randomisation du sizing pour éviter le pattern matching humain
        action = self._perturber_sizing(action, etat, joueur)
        return action

    # ==================================================================
    # OBTENIR LA DISTRIBUTION ABSTRAITE
    # ==================================================================

    def _obtenir_distribution(self,
                               etat   : EtatJeu,
                               joueur : Joueur) -> np.ndarray:
        """
        Retourne un vecteur de NB_ACTIONS_MAX probabilités.

        Ordre de priorité :
          Mode HU  → Blueprint HU → Deep CFR → Heuristique
          Mode 3J  → Blueprint 3J → Deep CFR → Heuristique
        """
        joueur_idx = etat.joueurs.index(joueur)
        non_elim   = etat.joueurs_non_elimines()
        mode_hu    = (len(non_elim) == 2)

        # ── Point 2 : Solveurs real-time postflop ─────────────────────────
        # FLOP  → SolveurProfondeurLimitee (mini-CFR, rapide)
        # TURN/RIVER → SolveurSousJeu (CFR pondéré par ranges adverses)
        phase_nom = etat.phase.name
        if phase_nom == 'FLOP' and self._solveur is not None:
            try:
                distribution = self._solveur.resoudre(etat, joueur, self)
                if distribution is not None:
                    self.stats['solveur_dl'] += 1
                    return distribution
            except Exception:
                pass   # fallback si erreur
        elif phase_nom in ('TURN', 'RIVER') and self._solveur_subgame is not None:
            try:
                distribution = self._solveur_subgame.resoudre(etat, joueur, self)
                if distribution is not None:
                    self.stats['solveur_sg'] += 1
                    return distribution
            except Exception:
                pass   # fallback si erreur

        # ── Tentative 1a : Blueprint Heads-Up ─────────────────────────────
        # Point 4 : double lookup pseudo-harmonic sur le bucket raise —
        # si la fraction observée tombe près d'une frontière, on interroge
        # les 2 buckets voisins et on blend les stratégies.
        if mode_hu and self._blueprint_hu is not None:
            cle  = self._construire_cle_hu(etat, joueur)
            frac = etat.mise_courante / max(etat.pot, 1)
            vec, _ = self._lookup_blueprint_blende(self._blueprint_hu, cle, frac)
            if vec is not None:
                self.stats['blueprint_hu'] += 1
                return vec

        # ── Tentative 1b : Blueprint 3-joueurs ────────────────────────────
        if not mode_hu and self._blueprint is not None:
            cle  = construire_cle_infoset(etat, joueur)
            frac = etat.mise_courante / max(etat.pot, 1)
            vec, _ = self._lookup_blueprint_blende(self._blueprint, cle, frac)
            if vec is not None:
                self.stats['blueprint'] += 1
                return vec

        # ── Tentative 2 : Deep CFR ─────────────────────────────────────────
        if self._reseaux_strategie is not None:
            try:
                etat_dict = self._convertir_etat(etat, joueur_idx)
                vec_input = encoder_infoset(etat_dict, joueur_idx)
                x = torch.tensor(vec_input, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self._reseaux_strategie[joueur_idx](x)
                strat = output.squeeze(0).cpu().numpy().astype(np.float32)
                self.stats['deep_cfr'] += 1
                return strat
            except Exception:
                pass

        # ── Fallback heuristique ───────────────────────────────────────────
        self.stats['heurist'] += 1
        return self._heuristique(etat, joueur)

    # ==================================================================
    # OFT — Opponent Frequency Tracking & Exploit Mixing
    # ==================================================================

    def enregistrer_action(self,
                            seat,
                            action,
                            street=None,
                            vpip_action=False,
                            contexte=None):
        """Enregistre une action adversaire dans l'OpponentTracker.

        Paramètres
        ----------
        seat        : int  — index de seat (0-2)
        action      : int ou str — code action (int) ou nom ('FOLD','CALL', …)
        street      : str ou None — 'preflop' / 'flop' / 'turn' / 'river'
        vpip_action : bool — True si l'action compte comme entrée volontaire
        contexte    : dict ou None — contexte additionnel (est_cbet_opp, etc.)
        """
        # Normaliser l'action en entier
        if isinstance(action, str):
            action_int = _ACTION_STR_VERS_INT.get(action.upper(), 0)
        else:
            action_int = int(action)

        # Construire le contexte à partir des paramètres nommés
        ctx = {}
        if street is not None:
            ctx['phase'] = street.lower()
        if vpip_action:
            ctx['vpip_action'] = True
        if contexte:
            ctx.update(contexte)

        self.tracker.observer_action(seat, action_int, ctx)

    def obtenir_distribution(self,
                              etat,
                              joueur,
                              adversaire_actif=None):
        """Distribution de stratégie publique avec fallback uniforme.

        Utilisé par les tests (et potentiellement par les solveurs externes).
        Garantit un vecteur float32 de taille 9, même sur un _MockEtat.
        """
        try:
            dist = self._obtenir_distribution(etat, joueur)
        except (AttributeError, TypeError):
            dist = np.full(9, 1.0 / 9, dtype=np.float32)

        if adversaire_actif is not None:
            game_type = self._detecter_game_type(etat)
            dist = self.mixer.ajuster(dist, adversaire_actif, game_type)

        return dist

    def _detecter_game_type(self, etat) -> str:
        """Détecte le type de jeu depuis l'état de jeu.

        Priorité :
          1. etat.game_type s'il existe
          2. len(joueurs_non_elimines()) == 2  → 'NLHE_HU'
          3. Sinon                             → 'NLHE_3MAX'
        """
        gt = getattr(etat, 'game_type', None)
        if gt is not None:
            return gt
        try:
            non_elim = etat.joueurs_non_elimines()
            return 'NLHE_HU' if len(non_elim) == 2 else 'NLHE_3MAX'
        except (AttributeError, TypeError):
            return 'NLHE_3MAX'

    def _identifier_adversaire_actif(self, etat, joueur) -> Optional[int]:
        """Retourne l'index de seat de l'adversaire principal, ou None.

        En HU  : le seul autre joueur non éliminé.
        En 3-max : le joueur en position BTN/IP (simplifié : le prochain actif).
        Retourne None si impossible à déterminer (MockEtat, etc.).
        """
        try:
            non_elim = etat.joueurs_non_elimines()
            autres = [j for j in non_elim if j is not joueur]
            if not autres:
                return None
            # Retourne l'index du premier adversaire actif dans la liste joueurs
            adv = autres[0]
            return etat.joueurs.index(adv)
        except (AttributeError, TypeError):
            return None

    # ==================================================================
    # CLÉ D'INFOSET HEADS-UP
    # ==================================================================

    def _construire_cle_hu(self, etat: EtatJeu, joueur: Joueur) -> str:
        """
        Construit la clé d'infoset au format HU.
        Format : "HU_PREFLOP|pos=P|bucket=B|pot=N|stacks=(A,B)|hist=H"
        """
        phase_idx = _PHASE_VERS_IDX.get(etat.phase, 0)
        nom_phase = _PHASE_NOM_HU[phase_idx]
        gb        = max(etat.grande_blinde, 1)

        non_elim = etat.joueurs_non_elimines()
        hu_pos   = non_elim.index(joueur) if joueur in non_elim else 0
        # Bucket HU (1 adversaire) pour matcher les clés du blueprint HU.
        bucket   = self._abs_cartes_hu.bucket(joueur.cartes, etat.board)
        pot_norm = _normaliser(etat.pot / gb, PALIERS_POT)
        stacks_str = ','.join(
            str(_normaliser(j.stack / gb, PALIERS_STACK))
            for j in non_elim
        )

        # N°4 : historique_phases[phase_idx] est maintenu par EtatJeu —
        # format identique à l'entraînement MCCFRHeadsUp.
        hist = etat.historique_phases[phase_idx]

        raise_bucket = _discretiser_raise_frac(etat.mise_courante / max(etat.pot, 1))

        return (
            f"{nom_phase}"
            f"|pos={hu_pos}"
            f"|bucket={bucket}"
            f"|pot={pot_norm}"
            f"|stacks=({stacks_str})"
            f"|hist={hist}"
            f"|raise={raise_bucket}"
        )

    # ==================================================================
    # LOOKUP BLUEPRINT AVEC BLENDING PSEUDO-HARMONIQUE (Point 4)
    # ==================================================================

    def _lookup_blueprint_blende(self, blueprint: dict,
                                  cle_originale: str,
                                  frac: float):
        """
        Consulte le blueprint avec un éventuel double-lookup pseudo-harmonique
        sur le bucket raise. Retourne (vec_strategie, poids_cumule) ou
        (None, 0.0) si aucun infoset n'est trouvé.

        Principe (Ganzfried-Sandholm 2013) :
          - On calcule la distribution pseudo-harmonique sur les buckets
            voisins (au plus 2) via buckets_pseudo_harmonic(frac).
          - On interroge le blueprint pour chaque bucket (en remplaçant
            le suffixe "|raise=X" dans la clé).
          - On blend les stratégies pondérées par leurs poids.
          - Si un seul bucket est retourné (cas hors-bornes ou frac==0),
            le comportement est strictement identique à l'ancien lookup.

        Renormalisation finale : vec.sum() = 1.0.
        """
        idx_raise = cle_originale.rfind('|raise=')
        if idx_raise < 0:
            # Format inattendu : fallback direct sans blending
            noeud = blueprint.get(cle_originale)
            if noeud is None:
                return None, 0.0
            strat = noeud.strategie_moyenne()
            if not strat or len(strat) == 0:
                return None, 0.0
            vec   = np.zeros(NB_ACTIONS_MAX, dtype=np.float32)
            n     = min(len(strat), NB_ACTIONS_MAX)
            vec[:n] = np.array(strat[:n], dtype=np.float32)
            total = vec.sum()
            if total <= _PROBA_MIN:
                return None, 0.0
            return vec / total, 1.0

        base          = cle_originale[:idx_raise]
        buckets_dist  = buckets_pseudo_harmonic(frac)
        vec_total     = np.zeros(NB_ACTIONS_MAX, dtype=np.float32)
        poids_trouve  = 0.0

        for bucket, poids in buckets_dist:
            cle   = f"{base}|raise={bucket}"
            noeud = blueprint.get(cle)
            if noeud is None:
                continue
            strat = noeud.strategie_moyenne()
            if not strat or len(strat) == 0:
                continue
            vec_b   = np.zeros(NB_ACTIONS_MAX, dtype=np.float32)
            n       = min(len(strat), NB_ACTIONS_MAX)
            vec_b[:n] = np.array(strat[:n], dtype=np.float32)
            s = vec_b.sum()
            if s <= _PROBA_MIN:
                continue
            vec_b /= s
            vec_total    += poids * vec_b
            poids_trouve += poids

        if poids_trouve <= _PROBA_MIN:
            return None, 0.0

        # Renormalisation (un seul bucket trouvé sur deux → on ne perd pas la masse)
        total = vec_total.sum()
        if total <= _PROBA_MIN:
            return None, 0.0
        return vec_total / total, poids_trouve

    # ==================================================================
    # CONVERSION EtatJeu → dict léger (compatible encoder_infoset)
    # ==================================================================

    def _convertir_etat(self, etat: EtatJeu, joueur_idx: int) -> dict:
        """
        Convertit un EtatJeu en dict léger compatible avec encoder_infoset().
        """
        phase_idx = _PHASE_VERS_IDX.get(etat.phase, 0)

        boards_par_phase = [
            [],
            etat.board[:3],
            etat.board[:4],
            etat.board[:5],
        ]

        # Calculer buckets ET équités en une passe (bucket_et_equite évite la double MC)
        buckets, equites = [], []
        for j in etat.joueurs:
            bkts_j, eqs_j = [], []
            for p in range(4):
                bk, eq = self._abs_cartes.bucket_et_equite(j.cartes, boards_par_phase[p])
                bkts_j.append(bk); eqs_j.append(eq)
            buckets.append(bkts_j); equites.append(eqs_j)

        stacks = [j.stack for j in etat.joueurs]

        # N°4 : utiliser directement etat.historique_phases — déjà segmenté
        # par street par le moteur de jeu, identique au format d'entraînement
        # MCCFR. Évite le bug de l'ancienne version qui assignait toutes les
        # actions à la phase courante uniquement.
        hist_phases = list(etat.historique_phases)  # copie défensive

        # raise_fracs : fraction mise_courante/pot pour la phase courante
        raise_fracs = [0.0, 0.0, 0.0, 0.0]
        raise_fracs[phase_idx] = etat.mise_courante / max(etat.pot, 1)

        return {
            'phase'         : phase_idx,
            'buckets'       : buckets,
            'equites'       : equites,
            'raise_fracs'   : raise_fracs,
            'pot'           : etat.pot,
            'grande_blinde' : max(etat.grande_blinde, 1),
            'stacks'        : stacks,
            'hist_phases'   : hist_phases,
        }

    # ==================================================================
    # MAPPING DISTRIBUTION ABSTRAITE → ACTIONS LÉGALES
    # ==================================================================

    def _mapper_sur_legales(self,
                             distribution : np.ndarray,
                             legales      : List[Action],
                             etat         : EtatJeu) -> np.ndarray:
        """
        Projette le vecteur de stratégie abstraite sur les actions légales.

        Pour FOLD / CHECK / CALL / ALL_IN : mapping direct (1 abstrait → 1 concret).
        Pour RAISE : le moteur de jeu ne fournit souvent qu'un seul raise (le min légal),
        alors que le blueprint contient N tailles abstraites. On somme TOUTES les
        probabilités de raise abstraites sur le(s) raise(s) concret(s) disponible(s),
        au prorata de leur proximité (softmax inverse de la distance).
        """
        probas = np.zeros(len(legales), dtype=np.float32)

        # Indices des raises concrets dans la liste legales
        indices_raises = [i for i, a in enumerate(legales)
                          if a.type == TypeAction.RAISE]

        # Tailles de raise selon la phase (preflop vs postflop)
        tailles_phase = (TAILLES_MISE_PREFLOP if etat.phase == Phase.PREFLOP
                         else TAILLES_MISE)

        # Masse totale des raises abstraits dans la distribution
        # On n'itère que sur les N tailles de la phase courante pour ne pas
        # mordre sur l'index ALL_IN qui suit immédiatement (ex: index 6 au preflop).
        prob_raises_abstraits = np.array([
            max(distribution[_IDX_RAISE_BASE + k], 0.0)
            for k in range(len(tailles_phase))
            if _IDX_RAISE_BASE + k < len(distribution)
        ], dtype=np.float32)
        masse_raise_totale = prob_raises_abstraits.sum()

        for i, action in enumerate(legales):
            if action.type == TypeAction.RAISE:
                continue   # traité séparément ci-dessous
            idx_abstrait = self._index_abstrait(action, etat)
            if 0 <= idx_abstrait < len(distribution):
                probas[i] = max(distribution[idx_abstrait], 0.0)

        # Répartir la masse raise totale sur les raises concrets disponibles
        if indices_raises and masse_raise_totale > _PROBA_MIN:
            if len(indices_raises) == 1:
                # Cas le plus fréquent : un seul raise concret → il reçoit tout
                probas[indices_raises[0]] = masse_raise_totale
            else:
                # Plusieurs raises concrets : répartir via pseudo-harmonic
                # action translation (Ganzfried & Sandholm 2013, utilisé par
                # Libratus et Pluribus). Chaque raise concret reçoit la masse
                # issue des tailles abstraites qui l'encadrent, proportionnelle
                # à sa fraction réelle du pot.
                pot = max(etat.pot, 1)
                # prob_par_taille[k] : masse attribuée à la taille abstraite k
                prob_par_taille = np.zeros(len(tailles_phase), dtype=np.float32)
                if len(prob_raises_abstraits) > 0:
                    prob_par_taille[:len(prob_raises_abstraits)] = prob_raises_abstraits

                # On distribue prob_par_taille[k] sur les raises concrets dont
                # la fraction réelle "colle" à la taille k selon pseudo-harmonic.
                poids = np.zeros(len(indices_raises), dtype=np.float32)
                for j, i in enumerate(indices_raises):
                    frac = (legales[i].montant - etat.mise_courante) / pot
                    # Quelles tailles abstraites ce concret "incarne-t-il" ?
                    # (au plus 2 — les deux voisines pseudo-harmoniques)
                    for k_idx, p_k in traduire_fraction(frac, tailles_phase):
                        poids[j] += p_k * prob_par_taille[k_idx]

                somme = poids.sum()
                if somme > _PROBA_MIN:
                    poids /= somme
                    for j, i in enumerate(indices_raises):
                        probas[i] = masse_raise_totale * poids[j]
                else:
                    # Fallback : aucune masse attribuable → uniforme sur les raises
                    for i in indices_raises:
                        probas[i] = masse_raise_totale / len(indices_raises)

        total = probas.sum()
        if total > _PROBA_MIN:
            return probas / total

        return np.ones(len(legales), dtype=np.float32) / len(legales)

    def _index_abstrait(self, action: Action, etat: EtatJeu) -> int:
        """
        Retourne l'index abstrait (0..NB_ACTIONS_MAX-1) d'une action concrète.

        RAISE et ALL_IN sont phase-dépendants :
          - Préflop  : TAILLES_MISE_PREFLOP (3 tailles) → ALL_IN à index 6
          - Postflop : TAILLES_MISE          (5 tailles) → ALL_IN à index 8
        Le blueprint MCCFR est entraîné avec ces deux espaces distincts.
        """
        if action.type == TypeAction.FOLD:
            return _IDX_FOLD
        if action.type == TypeAction.CHECK:
            return _IDX_CHECK
        if action.type == TypeAction.CALL:
            return _IDX_CALL

        tailles = (TAILLES_MISE_PREFLOP if etat.phase == Phase.PREFLOP
                   else TAILLES_MISE)
        idx_allin = _IDX_RAISE_BASE + len(tailles)

        if action.type == TypeAction.ALL_IN:
            return idx_allin
        if action.type == TypeAction.RAISE:
            # Reste compatible : on renvoie l'index dominant de la distribution
            # pseudo-harmonique (équivalent à l'ancien argmin sur des frontières
            # convexes, mais cohérent avec _indices_abstraits_distribution).
            dist = self._indices_abstraits_distribution(action, etat)
            if not dist:
                return -1
            return max(dist, key=lambda t: t[1])[0]
        return -1

    def _indices_abstraits_distribution(self,
                                         action: Action,
                                         etat:   EtatJeu) -> list:
        """
        Variante distributionnelle de `_index_abstrait` : pour un raise observé
        qui ne colle à aucune taille abstraite, répartit la masse sur les deux
        tailles voisines via pseudo-harmonic mapping (Ganzfried-Sandholm 2013).

        Retourne une liste [(idx_abstrait, poids)] telle que somme(poids)=1.
          - FOLD/CHECK/CALL/ALL_IN : [(idx, 1.0)]
          - RAISE                  : 1 ou 2 couples selon la fraction.

        Utile lorsqu'on infère la stratégie d'un adversaire à partir d'une
        raise observée (mode observation) — évite les discontinuités aux
        frontières de buckets que pourraient exploiter un adversaire conscient
        de l'abstraction.
        """
        if action.type == TypeAction.FOLD:
            return [(_IDX_FOLD,  1.0)]
        if action.type == TypeAction.CHECK:
            return [(_IDX_CHECK, 1.0)]
        if action.type == TypeAction.CALL:
            return [(_IDX_CALL,  1.0)]

        tailles = (TAILLES_MISE_PREFLOP if etat.phase == Phase.PREFLOP
                   else TAILLES_MISE)
        idx_allin = _IDX_RAISE_BASE + len(tailles)

        if action.type == TypeAction.ALL_IN:
            return [(idx_allin, 1.0)]

        if action.type == TypeAction.RAISE:
            pot = max(etat.pot, 1)
            fraction_reelle = (action.montant - etat.mise_courante) / pot
            dist = traduire_fraction(fraction_reelle, list(tailles))
            if not dist:
                return []
            # Remap indices locaux (tailles) → indices globaux (_IDX_RAISE_BASE+k).
            return [(_IDX_RAISE_BASE + k, p) for k, p in dist]

        return []

    # ==================================================================
    # SÉLECTION DE L'ACTION
    # ==================================================================

    def _selectionner(self,
                       legales : List[Action],
                       probas  : np.ndarray) -> Action:
        """Sélectionne une action selon les probabilités calculées."""
        if self.mode_deterministe:
            idx = int(np.argmax(probas))
        else:
            r = _rng.random()
            cumul = 0.0
            idx = len(legales) - 1
            for i, p in enumerate(probas):
                cumul += float(p)
                if r <= cumul:
                    idx = i
                    break
        return legales[idx]

    # ==================================================================
    # POINT 9 — RANDOMISATION DU SIZING (Libratus / Pluribus)
    # ==================================================================

    def _perturber_sizing(self, action: Action, etat: EtatJeu,
                           joueur: Joueur = None) -> Action:
        """
        Applique une perturbation aléatoire ±PERTURBATION_SIZING% au montant
        d'un RAISE pour éviter que les adversaires identifient des patterns.

        - Ne perturbe pas les ALL_IN (montant fixé par le stack).
        - Ne perturbe pas les CALL (montant imposé par la mise adverse).
        - Ne perturbe pas les CHECK/FOLD (montant = 0).
        - Le montant perturbé est clampé dans la plage légale :
            min = mise_courante + mise_min_raise   (raise minimal officiel)
            max = joueur.mise_tour + joueur.stack - 1  (juste en-dessous de l'all-in)
        - Si joueur est None (ancienne API), on garde seulement la borne min.
        """
        if action.type != TypeAction.RAISE:
            return action
        if PERTURBATION_SIZING <= 0.0:
            return action

        facteur = 1.0 + _rng.uniform(-PERTURBATION_SIZING, PERTURBATION_SIZING)
        nouveau_montant = int(round(action.montant * facteur))

        # Borne inférieure : raise minimal officiel (pas juste +1 jeton)
        mise_min = etat.mise_courante + max(etat.mise_min_raise, 1)
        nouveau_montant = max(nouveau_montant, mise_min)

        # Borne supérieure : strictement en-dessous de l'all-in pour éviter
        # de transformer par accident un RAISE perturbé en ALL_IN.
        if joueur is not None:
            mise_max_avant_allin = joueur.mise_tour + joueur.stack - 1
            if mise_max_avant_allin >= mise_min:
                nouveau_montant = min(nouveau_montant, mise_max_avant_allin)
            else:
                # Stack trop court pour un raise non-allin → pas de perturbation
                return action

        if nouveau_montant == action.montant:
            return action

        return Action(TypeAction.RAISE, nouveau_montant)

    # ==================================================================
    # HEURISTIQUE DE FALLBACK — N°5 : context-aware selon position/bucket
    # ==================================================================

    def _heuristique(self, etat: EtatJeu, joueur: Joueur) -> np.ndarray:
        """
        Stratégie de fallback context-aware quand aucune source n'est disponible.

        N°5 — Logique différenciée par position et force de la main :

        SB face à une mise (blind défense, position OOP) :
          bucket ≥ 5 : 40% raise agressif, 40% call, 20% fold
          bucket 3-4 : 20% raise, 45% call, 35% fold
          bucket 0-2 : 5% raise (bluff), 30% call, 65% fold

        BB face à une mise (option, dernière position preflop) :
          bucket ≥ 5 : 35% raise, 50% call, 15% fold
          bucket 3-4 : 15% raise, 55% call, 30% fold
          bucket 0-2 : 5% raise, 35% call, 60% fold

        BTN / situation sans mise (check/bet) :
          60% check/call, 30% raise moyen, 10% fold
        """
        vec = np.zeros(NB_ACTIONS_MAX, dtype=np.float32)

        position     = joueur.position                        # 0=BTN, 1=SB, 2=BB
        a_payer      = etat.mise_courante - joueur.mise_tour
        face_a_raise = (a_payer > 0)

        bucket = self._abs_cartes.bucket(joueur.cartes, etat.board)  # 0-7

        idx_raise_fort   = _IDX_ALLIN - 1
        idx_raise_milieu = _IDX_RAISE_BASE + len(TAILLES_MISE) // 2
        idx_raise_petit  = _IDX_RAISE_BASE

        if face_a_raise and position == 1:
            # SB face à une mise : défense de blind agressive
            if bucket >= 5:
                vec[_IDX_FOLD]      = 0.20
                vec[_IDX_CALL]      = 0.40
                vec[idx_raise_fort] = 0.40
            elif bucket >= 3:
                vec[_IDX_FOLD]        = 0.35
                vec[_IDX_CALL]        = 0.45
                vec[idx_raise_milieu] = 0.20
            else:
                vec[_IDX_FOLD]       = 0.65
                vec[_IDX_CALL]       = 0.30
                vec[idx_raise_petit] = 0.05

        elif face_a_raise and position == 2:
            # BB face à une mise : défense de blind avec option
            if bucket >= 5:
                vec[_IDX_FOLD]        = 0.15
                vec[_IDX_CALL]        = 0.50
                vec[idx_raise_fort]   = 0.35
            elif bucket >= 3:
                vec[_IDX_FOLD]        = 0.30
                vec[_IDX_CALL]        = 0.55
                vec[idx_raise_milieu] = 0.15
            else:
                vec[_IDX_FOLD]       = 0.60
                vec[_IDX_CALL]       = 0.35
                vec[idx_raise_petit] = 0.05

        else:
            # BTN ou situation sans mise : logique standard
            vec[_IDX_FOLD]        = 0.10
            vec[_IDX_CHECK]       = 0.60
            vec[_IDX_CALL]        = 0.60
            vec[idx_raise_milieu] = 0.30

        total = vec.sum()
        if total > _PROBA_MIN:
            return vec / total
        return np.ones(NB_ACTIONS_MAX, dtype=np.float32) / NB_ACTIONS_MAX

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================

    def afficher_stats(self) -> None:
        """Affiche les statistiques de décision depuis le début de la partie."""
        total = max(self.stats['total'], 1)
        print(f"\n{'─'*50}")
        print(f"  AgentAXIOM — Statistiques de décision")
        print(f"{'─'*50}")
        print(f"  Total décisions   : {self.stats['total']:,}")
        print(f"  Blueprint HU      : {self.stats['blueprint_hu']:,} "
              f"({100*self.stats['blueprint_hu']//total}%)")
        print(f"  Blueprint 3J      : {self.stats['blueprint']:,} "
              f"({100*self.stats['blueprint']//total}%)")
        print(f"  Solveur FLOP      : {self.stats['solveur_dl']:,} "
              f"({100*self.stats['solveur_dl']//total}%)")
        print(f"  Solveur TURN/RIV  : {self.stats['solveur_sg']:,} "
              f"({100*self.stats['solveur_sg']//total}%)")
        print(f"  Deep CFR          : {self.stats['deep_cfr']:,} "
              f"({100*self.stats['deep_cfr']//total}%)")
        print(f"  Heuristique       : {self.stats['heurist']:,} "
              f"({100*self.stats['heurist']//total}%)")
        print(f"{'─'*50}\n")

    def reinitialiser_stats(self) -> None:
        """Réinitialise les statistiques (utile entre deux parties)."""
        for k in self.stats:
            self.stats[k] = 0

    def __repr__(self) -> str:
        sources = []
        if self._blueprint_hu is not None:
            sources.append(f"blueprint_hu({len(self._blueprint_hu):,} infosets)")
        if self._blueprint is not None:
            sources.append(f"blueprint({len(self._blueprint):,} infosets)")
        if self._reseaux_strategie is not None:
            sources.append("deep_cfr(3 réseaux)")
        if not sources:
            sources.append("heuristique")
        return (f"AgentAXIOM("
                f"mode={'déterministe' if self.mode_deterministe else 'stochastique'}, "
                f"sources=[{', '.join(sources)}])")


# =============================================================================
# FONCTION DE CRÉATION RAPIDE
# =============================================================================

def creer_agent(chemin_blueprint    : str  = None,
                chemin_blueprint_hu : str  = None,
                chemin_strategie    : str  = None,
                mode_deterministe   : bool = False,
                verbose             : bool = True,
                continuations       : bool = True,
                solveur_realtime    : bool = True,
                solveur_iterations  : int  = 50,
                solveur_temps_max   : float = 0.3) -> AgentAXIOM:
    """
    Crée et initialise un AgentAXIOM en chargeant toutes les sources disponibles.
    Les fichiers manquants sont ignorés sans lever d'exception.

    Pluribus complet (TA1 fix 2026-04-30) :
      - continuations=True       : charge blueprint_*_call/fold/raise (k=4)
      - solveur_realtime=True    : active SolveurProfondeurLimitee (FLOP)
                                    et SolveurSousJeu (TURN/RIVER)
      - solveur_iterations=50    : config "rapide" pour éval batch
                                    (mode prod screen/jouer.py utilise 50 + temps_max=3s)
      - solveur_temps_max=0.3s   : budget par décision (vs 3s en prod)
                                    → 60K mains × ~10 décisions × 0.3s = ~50h max
                                    en pratique solveurs ne se déclenchent que postflop
    """
    import os

    agent = AgentAXIOM(mode_deterministe=mode_deterministe)

    chemin_hu = chemin_blueprint_hu or CHEMIN_BLUEPRINT_HU
    if chemin_hu and os.path.exists(chemin_hu):
        try:
            agent.charger_blueprint_hu(chemin_hu)
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Blueprint HU non chargé : {e}")
    elif verbose:
        print(f"  ℹ️  Blueprint HU introuvable ({chemin_hu})"
              f" — mode HU utilisera Deep CFR ou heuristique")

    chemin_bp = chemin_blueprint or CHEMIN_BLUEPRINT
    if chemin_bp and os.path.exists(chemin_bp):
        try:
            agent.charger_blueprint(chemin_bp)
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Blueprint 3J non chargé : {e}")

        # ── TA1 fix : continuation strategies Pluribus k=4 ────────────────
        # Tente de charger les variantes biaisées blueprint_*_call/fold/raise
        # à partir du même nom de base que chemin_bp. Silencieux si absent.
        if continuations:
            try:
                dossier  = os.path.dirname(chemin_bp) or '.'
                base_nom = os.path.basename(chemin_bp).replace('.pkl', '')
                agent.charger_blueprints_continuations(dossier=dossier, base_nom=base_nom)
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Continuations non chargées : {e}")
    elif verbose:
        print(f"  ℹ️  Blueprint 3J introuvable ({chemin_bp}) — ignoré")

    chemin_s = chemin_strategie or CHEMIN_STRATEGY_NET
    if chemin_s:
        tous_presents = all(
            os.path.exists(chemin_s.replace('.pt', f'_j{i}.pt'))
            for i in range(3)
        )
        if tous_presents:
            try:
                agent.charger_deep_cfr(chemin_s)
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Deep CFR non chargé : {e}")
        elif verbose:
            print(f"  ℹ️  Deep CFR introuvable ({chemin_s}) — ignoré")

    # Charger les réseaux de valeur (optionnel — utilisés par le solveur depth-limited)
    base_valeur = "data/models/valeur_net.pt"
    tous_valeur = all(
        os.path.exists(base_valeur.replace('.pt', f'_j{i}.pt'))
        for i in range(3)
    )
    if tous_valeur:
        try:
            agent.charger_reseaux_valeur(base_valeur)
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Réseaux valeur non chargés : {e}")
    elif verbose:
        print(f"  ℹ️  Réseaux valeur introuvables — oracle = équité brute")

    # ── TA1 fix : real-time solveur (FLOP + TURN/RIVER) ───────────────────
    # Mode "rapide" pour les évals batch (50 iter, 0.3s budget).
    # screen/jouer.py reste libre d'appeler activer_solveur(...) avec params
    # plus larges (e.g. temps_max=3s) pour le mode prod.
    if solveur_realtime:
        try:
            agent.activer_solveur(
                profondeur    = 2,
                nb_iterations = solveur_iterations,
                nb_simul      = 6,
                nb_scenarios  = 15,
                temps_max     = solveur_temps_max,
            )
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Solveur real-time non activé : {e}")

    if verbose:
        print(f"  ✅ Agent prêt : {agent}")

    return agent


# =============================================================================
# TEST RAPIDE (si exécuté directement)
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*65)
    print("  AXIOM — Test agent.py (corrections N°3, N°4, N°5)")
    print("="*65)

    from engine.game_state import EtatJeu, Phase
    from engine.player import Joueur, TypeJoueur, StatutJoueur
    from engine.actions import actions_legales
    from engine.card import DeckAXIOM

    j0 = Joueur("AXIOM",    TypeJoueur.AXIOM,  1500, 0)
    j1 = Joueur("Humain-1", TypeJoueur.HUMAIN, 1500, 1)
    j2 = Joueur("Humain-2", TypeJoueur.HUMAIN, 1500, 2)
    joueurs = [j0, j1, j2]

    etat_3j = EtatJeu(joueurs, petite_blinde=10, grande_blinde=20)
    etat_3j.nouvelle_main()

    print(f"\n  État 3 joueurs :")
    print(f"    Joueurs non éliminés : {len(etat_3j.joueurs_non_elimines())}")

    agent = AgentAXIOM(mode_deterministe=False)
    print(f"\n  {agent}")

    legales = actions_legales(
        j0,
        mise_a_suivre  = etat_3j.mise_courante,
        pot            = etat_3j.pot,
        mise_min_raise = etat_3j.mise_min_raise,
    )

    # ── Test mode 3-joueurs ─────────────────────────────────────────────────
    print(f"\n  ── Test mode 3-joueurs ──────────────────────────────────────")
    action_3j = agent.choisir_action(etat_3j, j0, legales)
    print(f"  Action choisie (heuristique) : {action_3j}")
    assert agent.stats['heurist'] == 1

    # ── Test mode HU ───────────────────────────────────────────────────────
    print(f"\n  ── Test mode Heads-Up ────────────────────────────────────────")
    j2.statut = StatutJoueur.ELIMINE
    non_elim_hu = etat_3j.joueurs_non_elimines()
    assert len(non_elim_hu) == 2, f"Attendu 2 joueurs, trouvé {len(non_elim_hu)}"

    cle_hu = agent._construire_cle_hu(etat_3j, j0)
    print(f"  Clé HU générée : {cle_hu}")
    assert cle_hu.startswith("HU_")
    partie_stacks = [p for p in cle_hu.split('|') if p.startswith('stacks=')]
    assert partie_stacks[0].count(',') == 1
    print(f"  ✅ Format clé HU correct (HU_, 2 stacks)")

    agent.reinitialiser_stats()
    action_hu = agent.choisir_action(etat_3j, j0, legales)
    print(f"  Action HU (sans blueprint HU) : {action_hu}")
    assert agent.stats['heurist'] == 1
    print(f"  ✅ Fallback heuristique HU correct")

    # ── Test heuristique context-aware N°5 ────────────────────────────────
    print(f"\n  ── Test heuristique context-aware (N°5) ────────────────────")
    j0_sb = Joueur("AXIOM-SB", TypeJoueur.AXIOM, 480, 1)
    j0_sb.mise_tour = 10
    j0_sb.cartes    = j0.cartes
    etat_3j.mise_courante = 40
    vec_sb = agent._heuristique(etat_3j, j0_sb)
    print(f"  Heuristique SB face à raise : "
          f"fold={vec_sb[0]:.2f}, call={vec_sb[2]:.2f}, "
          f"raise+={vec_sb[3:].sum():.2f}")
    assert abs(vec_sb.sum() - 1.0) < 1e-5
    print(f"  ✅ Heuristique SB context-aware correcte")

    # ── Test N°4 : historique_phases ──────────────────────────────────────
    print(f"\n  ── Test historique_phases (N°4) ─────────────────────────────")
    etat_dict = agent._convertir_etat(etat_3j, 0)
    assert 'hist_phases' in etat_dict
    assert isinstance(etat_dict['hist_phases'], list)
    assert len(etat_dict['hist_phases']) == 4
    print(f"  hist_phases : {etat_dict['hist_phases']}")
    print(f"  ✅ Historique par phase correct")

    j2.statut = StatutJoueur.ACTIF
    agent2 = creer_agent(verbose=True)
    agent.afficher_stats()

    print("  ✅ Tous les tests agent.py sont passés !")
    print("="*65 + "\n")
