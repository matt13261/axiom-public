# =============================================================================
# AXIOM — training/self_play.py
# Simulation de parties complètes pour évaluation et entraînement (Phase 5).
# Mise à jour Phase 10 : ajout AgentTAG, AgentLAG, AgentRegulier.
#
# ─────────────────────────────────────────────────────────────────────────────
# RÔLE DE CE MODULE
# ─────────────────────────────────────────────────────────────────────────────
#
# self_play.py simule des parties de poker entre agents sans interface
# graphique — uniquement le moteur de jeu et les agents IA.
#
# Deux usages principaux :
#
#   1. ÉVALUATION : faire jouer AXIOM contre lui-même ou contre des agents
#      de référence (random, call-only) pour mesurer ses performances.
#      → Utilisé par training/evaluator.py
#
#   2. GÉNÉRATION DE DONNÉES : produire des transitions (état, action, gain)
#      pour un éventuel fine-tuning futur des réseaux Deep CFR.
#      → Format compatible avec les Reservoir Buffers de ai/reservoir.py
#
# ─────────────────────────────────────────────────────────────────────────────
# AGENTS DISPONIBLES
# ─────────────────────────────────────────────────────────────────────────────
#
#   AgentAleatoire  : choisit une action légale au hasard (baseline)
#   AgentCallOnly   : appelle toujours, fold si tout-in requis (baseline)
#   AgentRaiseOnly  : relance au maximum (baseline agressif)
#   AgentTAG        : Tight Aggressive — range sélective, agressif postflop
#   AgentLAG        : Loose Aggressive — large range, bluffeur, overbet
#   AgentRegulier   : Régulier de casino — appelle trop, fort sur gros hands
#   AgentAXIOM      : notre agent principal (blueprint + Deep CFR)
#
# ─────────────────────────────────────────────────────────────────────────────
# FORMAT DES RÉSULTATS
# ─────────────────────────────────────────────────────────────────────────────
#
# Chaque partie retourne un ResultatPartie :
#   - gains_nets[joueur_idx] : gain net en jetons (positif = gagné)
#   - nb_mains               : nombre de mains jouées
#   - historique_mains       : liste de ResultatMain
#
# L'agrégation de N parties produit des statistiques :
#   - winrate en bb/100 (big blinds pour 100 mains)
#   - taux de victoire global
#
# =============================================================================

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

from engine.game_state import EtatJeu, Phase
from engine.player import Joueur, StatutJoueur, TypeJoueur
from engine.actions import Action, TypeAction, actions_legales
from engine.hand_evaluator import determiner_gagnants
from engine.blind_structure import StructureBlinde
from engine.card import cartes_en_texte
from config.settings import NB_JOUEURS, STACK_DEPART

# Import treys pour l'évaluation de force de main (bots semi-pro)
try:
    from treys import Card as TreysCard, Evaluator as TreysEvaluator
    _TREYS_DISPONIBLE = True
    _EVALUATEUR_TREYS = TreysEvaluator()
except ImportError:
    _TREYS_DISPONIBLE = False
    _EVALUATEUR_TREYS = None


# =============================================================================
# STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class ResultatMain:
    """Résultat d'une main unique."""
    numero         : int
    gains_nets     : List[int]          # gain net par joueur (peut être négatif)
    pot_final      : int
    nb_actions     : int
    phase_finale   : str                # "PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"
    gagnant_idx    : int                # -1 si égalité


@dataclass
class ResultatPartie:
    """Résultat agrégé d'une partie complète (plusieurs mains)."""
    gains_nets     : List[int]          # cumulés sur toutes les mains
    nb_mains       : int
    nb_mains_preflop : int              # mains terminées au preflop (fold)
    nb_showdowns   : int
    historique     : List[ResultatMain] = field(default_factory=list)
    duree_s        : float = 0.0

    def winrate_bb100(self, joueur_idx: int, grande_blinde: int) -> float:
        """
        Winrate en bb/100 pour un joueur.
        Métrique standard du poker en ligne :
            winrate = (gain_net / grande_blinde) / nb_mains * 100
        """
        if self.nb_mains == 0 or grande_blinde == 0:
            return 0.0
        return (self.gains_nets[joueur_idx] / grande_blinde) / self.nb_mains * 100


# =============================================================================
# AGENTS DE RÉFÉRENCE (baselines)
# =============================================================================

class AgentAleatoire:
    """
    Agent baseline : choisit une action légale uniformément au hasard.
    Utile comme plancher de référence — tout agent sérieux doit le battre.
    """
    def __init__(self, graine: int = None):
        self._rng = random.Random(graine)

    def choisir_action(self, etat: EtatJeu, joueur: Joueur,
                        legales: List[Action]) -> Action:
        return self._rng.choice(legales)

    def __repr__(self) -> str:
        return "AgentAleatoire"


class AgentCallOnly:
    """
    Agent baseline : appelle toujours la mise en cours.
    S'il ne peut pas appeler (all-in requis), il check ou fold.
    Représente un joueur passif/peureux.
    """
    def choisir_action(self, etat: EtatJeu, joueur: Joueur,
                        legales: List[Action]) -> Action:
        # Priorité : check > call > all-in > fold
        for type_voulu in (TypeAction.CHECK, TypeAction.CALL,
                           TypeAction.ALL_IN, TypeAction.FOLD):
            for a in legales:
                if a.type == type_voulu:
                    return a
        return legales[0]

    def __repr__(self) -> str:
        return "AgentCallOnly"


class AgentRaiseOnly:
    """
    Agent baseline : raise/all-in autant que possible.
    Représente un joueur ultra-agressif. Utile pour tester la résistance
    d'AXIOM face à une pression maximale.
    """
    def choisir_action(self, etat: EtatJeu, joueur: Joueur,
                        legales: List[Action]) -> Action:
        for type_voulu in (TypeAction.RAISE, TypeAction.ALL_IN,
                           TypeAction.CALL, TypeAction.CHECK, TypeAction.FOLD):
            for a in legales:
                if a.type == type_voulu:
                    return a
        return legales[0]

    def __repr__(self) -> str:
        return "AgentRaiseOnly"


# =============================================================================
# HELPERS PARTAGÉS ENTRE LES BOTS SEMI-PRO (Phase 10)
# =============================================================================

def _force_preflop(cartes: list) -> int:
    """
    Évalue la force d'une main préflop sur une échelle de 0 à 4.

    4 = premium  : AA, KK, QQ, JJ, TT, AK
    3 = fort     : 99/88, AQ/AJ/ATs, KQs, KJs, QJs, JTs, AQo
    2 = jouable  : 77/66/55/44, KTs/QTs, AJo, KQo, connecteurs suited ≥ T9s
    1 = spécul.  : 33/22, petits suited connectors, Axs
    0 = déchets  : tout le reste
    """
    if not _TREYS_DISPONIBLE or len(cartes) < 2:
        return 1  # valeur neutre par défaut

    r1 = TreysCard.get_rank_int(cartes[0])   # 0=2 … 12=As
    r2 = TreysCard.get_rank_int(cartes[1])
    s1 = TreysCard.get_suit_int(cartes[0])
    s2 = TreysCard.get_suit_int(cartes[1])

    if r1 < r2:                              # r1 toujours la carte la plus haute
        r1, r2 = r2, r1

    suited = (s1 == s2)
    paire  = (r1 == r2)

    # ── Paires ─────────────────────────────────────────────────────────────
    if paire:
        if r1 >= 8:  return 4   # TT+ (rang 8 = T dans treys : 2=0,3=1,...,T=8)
        if r1 >= 5:  return 3   # 77/88/99
        if r1 >= 2:  return 2   # 44/55/66
        return 1                 # 22/33

    # Indices treys : As=12, Roi=11, Dame=10, Valet=9, Dix=8
    AS, R, D, V, X = 12, 11, 10, 9, 8

    # ── Premium non-paire ──────────────────────────────────────────────────
    if r1 == AS and r2 == R:              return 4   # AK (s ou o)

    # ── Fort ──────────────────────────────────────────────────────────────
    if r1 == AS and r2 == D:              return 3   # AQ (s ou o)
    if r1 == AS and r2 >= V and suited:   return 3   # AJs
    if r1 == AS and r2 == V:             return 3    # AJo (fort)
    if r1 == R  and r2 == D and suited:  return 3    # KQs
    if r1 == R  and r2 >= V and suited:  return 3    # KJs
    if r1 == D  and r2 == V and suited:  return 3    # QJs
    if r1 == V  and r2 == X and suited:  return 3    # JTs

    # ── Jouable ───────────────────────────────────────────────────────────
    if r1 == AS and r2 == X and suited:  return 2    # ATs
    if r1 == R  and r2 == X and suited:  return 2    # KTs
    if r1 == D  and r2 == X and suited:  return 2    # QTs
    if r1 == R  and r2 == D:             return 2    # KQo
    if suited and (r1 - r2) == 1 and r1 >= X: return 2  # T9s+
    if suited and (r1 - r2) == 1 and r1 >= 7: return 2  # 87s/98s

    # ── Spéculatif ────────────────────────────────────────────────────────
    if r1 == AS and suited:               return 1   # Axs
    if suited and (r1 - r2) == 1:        return 1    # petits suited connectors
    if suited and (r1 - r2) == 2 and r1 >= 6: return 1  # suited one-gappers

    return 0   # déchets


def _force_postflop(cartes: list, board: list) -> float:
    """
    Évalue la force d'une main postflop via treys.
    Retourne un score [0.0, 1.0] : 1.0 = quinte flush royale.
    """
    if not _TREYS_DISPONIBLE or len(cartes) < 2 or not board:
        return 0.5   # valeur neutre si pas de board

    try:
        rang = _EVALUATEUR_TREYS.evaluate(board, cartes)
        # rang ∈ [1, 7462], 1 = meilleur
        return 1.0 - (rang - 1) / 7461.0
    except Exception:
        return 0.5


def _get_action(legales: List[Action], *types: TypeAction) -> Action:
    """
    Retourne la première action correspondant à l'un des TypeAction donnés.
    Fallback sur legales[0] si aucun match.
    """
    for t in types:
        for a in legales:
            if a.type == t:
                return a
    return legales[0]


def _construire_raise(legales: List[Action],
                       etat: EtatJeu,
                       joueur: Joueur,
                       fraction_pot: float) -> Optional[Action]:
    """
    Construit une Action RAISE avec un montant cible = mise_courante + fraction × pot.
    Le montant est borné par le stack disponible (→ ALL_IN si nécessaire).
    Retourne None si aucun raise n'est légalement possible.
    """
    has_raise = any(a.type == TypeAction.RAISE for a in legales)
    has_allin = any(a.type == TypeAction.ALL_IN for a in legales)

    if not has_raise and not has_allin:
        return None

    # Montant cible
    montant_cible = etat.mise_courante + max(
        etat.mise_min_raise,
        int(etat.pot * fraction_pot)
    )

    # Plafonner au stack total disponible
    stack_total = joueur.mise_tour + joueur.stack
    montant_cible = min(montant_cible, stack_total)

    # Si on atteint le stack → retourner ALL_IN
    if montant_cible >= stack_total and has_allin:
        for a in legales:
            if a.type == TypeAction.ALL_IN:
                return a

    # Construire le RAISE personnalisé
    if has_raise and montant_cible > etat.mise_courante:
        return Action(TypeAction.RAISE, montant=montant_cible)

    # Sinon ALL_IN si disponible
    if has_allin:
        for a in legales:
            if a.type == TypeAction.ALL_IN:
                return a

    return None


# =============================================================================
# BOT 1 — TAG : Tight Aggressive
# =============================================================================

class AgentTAG:
    """
    Simule un joueur TAG (Tight Aggressive) — style le plus commun
    chez les joueurs de casino réguliers et semi-pros.

    Caractéristiques :
    ─ Préflop : range sélective (~20%), open-raise 2.5× pot, 3-bet premiums
    ─ Postflop : c-bet 65 % pot sur bonnes mains, value-bet fort, fold sur air
    ─ Sizing   : 65-75 % pot en value, sizing similaire en bluff
    ─ Discipline : fold face à résistance répétée, appelle avec de bonnes mains
    """

    def __init__(self, graine: int = None):
        self._rng = random.Random(graine)

    def choisir_action(self, etat: EtatJeu, joueur: Joueur,
                        legales: List[Action]) -> Action:
        if etat.phase == Phase.PREFLOP:
            return self._preflop(etat, joueur, legales)
        return self._postflop(etat, joueur, legales)

    # ── Préflop ────────────────────────────────────────────────────────────

    def _preflop(self, etat: EtatJeu, joueur: Joueur,
                  legales: List[Action]) -> Action:
        force        = _force_preflop(joueur.cartes)
        face_a_raise = etat.mise_courante > etat.grande_blinde

        if force == 4:
            # Premium : toujours relancer (3-bet inclus)
            action = _construire_raise(legales, etat, joueur, 2.5)
            return action or _get_action(legales, TypeAction.CALL, TypeAction.CHECK)

        elif force == 3:
            # Fort : open-raise si pas de raise, sinon appeler
            if not face_a_raise:
                action = _construire_raise(legales, etat, joueur, 2.5)
                if action:
                    return action
            # Face à un raise : appeler avec bonnes mains, fold si 3-bet
            a_payer = etat.mise_courante - joueur.mise_tour
            if a_payer > joueur.stack * 0.35:
                return _get_action(legales, TypeAction.FOLD, TypeAction.CHECK)
            return _get_action(legales, TypeAction.CALL, TypeAction.CHECK, TypeAction.FOLD)

        elif force == 2:
            # Jouable : limp/check, fold face à une grosse mise
            if face_a_raise:
                a_payer = etat.mise_courante - joueur.mise_tour
                if a_payer > joueur.stack * 0.20:
                    return _get_action(legales, TypeAction.FOLD, TypeAction.CHECK)
            return _get_action(legales, TypeAction.CHECK, TypeAction.CALL, TypeAction.FOLD)

        elif force == 1:
            # Spéculatif : check gratuit, fold face à tout raise
            if face_a_raise:
                return _get_action(legales, TypeAction.FOLD, TypeAction.CHECK)
            return _get_action(legales, TypeAction.CHECK, TypeAction.FOLD)

        else:
            # Déchets : toujours fold (ou check gratuit)
            return _get_action(legales, TypeAction.CHECK, TypeAction.FOLD)

    # ── Postflop ───────────────────────────────────────────────────────────

    def _postflop(self, etat: EtatJeu, joueur: Joueur,
                   legales: List[Action]) -> Action:
        force = _force_postflop(joueur.cartes, etat.board)
        face_a_mise = any(a.type == TypeAction.CALL for a in legales)

        if force > 0.75:
            # Très forte main : value-bet 70 % pot
            action = _construire_raise(legales, etat, joueur, 0.70)
            return action or _get_action(legales, TypeAction.CALL, TypeAction.CHECK)

        elif force > 0.55:
            # Main correcte : c-bet 65 % des opportunités, sinon call/check
            if not face_a_mise and self._rng.random() < 0.65:
                action = _construire_raise(legales, etat, joueur, 0.65)
                if action:
                    return action
            if face_a_mise:
                a_payer = etat.mise_courante - joueur.mise_tour
                if a_payer <= etat.pot * 0.5:
                    return _get_action(legales, TypeAction.CALL)
                return _get_action(legales, TypeAction.FOLD, TypeAction.CHECK)
            return _get_action(legales, TypeAction.CHECK)

        elif force > 0.35:
            # Main faible : call avec bonnes pot-odds seulement
            if face_a_mise:
                a_payer = etat.mise_courante - joueur.mise_tour
                if a_payer <= etat.pot * 0.28:
                    return _get_action(legales, TypeAction.CALL)
                return _get_action(legales, TypeAction.FOLD)
            return _get_action(legales, TypeAction.CHECK)

        else:
            # Air : check ou fold
            return _get_action(legales, TypeAction.CHECK, TypeAction.FOLD)

    def __repr__(self) -> str:
        return "AgentTAG"


# =============================================================================
# BOT 2 — LAG : Loose Aggressive
# =============================================================================

class AgentLAG:
    """
    Simule un joueur LAG (Loose Aggressive) — style dangereux, type "tourelle".

    Caractéristiques :
    ─ Préflop : range large (~40%), 3-bet fréquent en position
    ─ Postflop : bluff sur bons runouts, float en position, overbet possible
    ─ Sizing   : overbet 120-150 % pot possible, sizing varié pour être illisible
    ─ Adaptation : ralentit face aux résistances répétées (pas kamikaze)
    """

    def __init__(self, graine: int = None):
        self._rng = random.Random(graine)

    def choisir_action(self, etat: EtatJeu, joueur: Joueur,
                        legales: List[Action]) -> Action:
        if etat.phase == Phase.PREFLOP:
            return self._preflop(etat, joueur, legales)
        return self._postflop(etat, joueur, legales)

    # ── Préflop ────────────────────────────────────────────────────────────

    def _preflop(self, etat: EtatJeu, joueur: Joueur,
                  legales: List[Action]) -> Action:
        force        = _force_preflop(joueur.cartes)
        face_a_raise = etat.mise_courante > etat.grande_blinde

        if force == 4:
            # Premium : toujours relancer fort (3-bet agressif)
            action = _construire_raise(legales, etat, joueur, 3.0)
            return action or _get_action(legales, TypeAction.CALL, TypeAction.CHECK)

        elif force == 3:
            # Fort : raise ou 3-bet
            action = _construire_raise(legales, etat, joueur, 2.5)
            return action or _get_action(legales, TypeAction.CALL, TypeAction.CHECK)

        elif force == 2:
            # Jouable : raise en open, call face à raise
            if not face_a_raise:
                action = _construire_raise(legales, etat, joueur, 2.0)
                if action:
                    return action
            return _get_action(legales, TypeAction.CALL, TypeAction.CHECK, TypeAction.FOLD)

        elif force == 1:
            # Spéculatif : limp ou call si pas trop cher, 3-bet bluff 15 %
            if face_a_raise:
                a_payer = etat.mise_courante - joueur.mise_tour
                # 3-bet bluff 15 % avec suited connectors
                if a_payer <= joueur.stack * 0.25 and self._rng.random() < 0.15:
                    action = _construire_raise(legales, etat, joueur, 3.0)
                    if action:
                        return action
                if a_payer <= joueur.stack * 0.12:
                    return _get_action(legales, TypeAction.CALL)
                return _get_action(legales, TypeAction.FOLD, TypeAction.CHECK)
            # Pas de raise : limp ou petit raise 25 % du temps
            if self._rng.random() < 0.25:
                action = _construire_raise(legales, etat, joueur, 1.5)
                if action:
                    return action
            return _get_action(legales, TypeAction.CHECK, TypeAction.CALL)

        else:
            # Déchets : fold (sauf check gratuit)
            return _get_action(legales, TypeAction.CHECK, TypeAction.FOLD)

    # ── Postflop ───────────────────────────────────────────────────────────

    def _postflop(self, etat: EtatJeu, joueur: Joueur,
                   legales: List[Action]) -> Action:
        force = _force_postflop(joueur.cartes, etat.board)
        face_a_mise = any(a.type == TypeAction.CALL for a in legales)

        if force > 0.70:
            # Très forte main : value-bet / overbet aléatoire
            fraction = self._rng.choice([0.75, 1.00, 1.30])
            action = _construire_raise(legales, etat, joueur, fraction)
            return action or _get_action(legales, TypeAction.CALL, TypeAction.CHECK)

        elif force > 0.50:
            # Main correcte : mise / float en position
            if not face_a_mise:
                fraction = self._rng.choice([0.60, 0.80])
                action = _construire_raise(legales, etat, joueur, fraction)
                if action:
                    return action
            # Face à une mise : call, ou raise 30 % des fois
            if face_a_mise and self._rng.random() < 0.30:
                action = _construire_raise(legales, etat, joueur, 1.0)
                if action:
                    return action
            return _get_action(legales, TypeAction.CALL, TypeAction.CHECK, TypeAction.FOLD)

        elif force > 0.25:
            # Main faible : bluff 30 % du temps, sinon check/call léger
            if not face_a_mise and self._rng.random() < 0.30:
                action = _construire_raise(legales, etat, joueur, 0.80)
                if action:
                    return action
            if face_a_mise:
                a_payer = etat.mise_courante - joueur.mise_tour
                # Float en position pour voler plus tard
                if a_payer <= etat.pot * 0.40 and self._rng.random() < 0.40:
                    return _get_action(legales, TypeAction.CALL)
                return _get_action(legales, TypeAction.FOLD, TypeAction.CHECK)
            return _get_action(legales, TypeAction.CHECK)

        else:
            # Air : bluff pur 25 % du temps, sinon check/fold
            if not face_a_mise and self._rng.random() < 0.25:
                action = _construire_raise(legales, etat, joueur, 1.00)
                if action:
                    return action
            return _get_action(legales, TypeAction.CHECK, TypeAction.FOLD)

    def __repr__(self) -> str:
        return "AgentLAG"


# =============================================================================
# BOT 3 — Régulier de casino
# =============================================================================

class AgentRegulier:
    """
    Simule un régulier de casino — le semi-pro "standard".

    Ce joueur représente la cible réelle d'AXIOM dans 2 mois.
    Il est solide sur ses grosses mains mais fait des erreurs classiques :
    appelle trop souvent avec des mains moyennes, bluff rare et lisible.

    Caractéristiques :
    ─ Préflop : joue uniquement ses bonnes mains (force ≥ 3), parfois les paires
    ─ Postflop : call trop souvent avec top pair faible kicker
    ─ Bluff    : rare, principalement des semi-bluffs sur tirage
    ─ Sizing   : peu varié (bet 75 % pot partout), facilement lisible
    """

    def __init__(self, graine: int = None):
        self._rng = random.Random(graine)

    def choisir_action(self, etat: EtatJeu, joueur: Joueur,
                        legales: List[Action]) -> Action:
        if etat.phase == Phase.PREFLOP:
            return self._preflop(etat, joueur, legales)
        return self._postflop(etat, joueur, legales)

    # ── Préflop ────────────────────────────────────────────────────────────

    def _preflop(self, etat: EtatJeu, joueur: Joueur,
                  legales: List[Action]) -> Action:
        force        = _force_preflop(joueur.cartes)
        face_a_raise = etat.mise_courante > etat.grande_blinde

        if force == 4:
            # Premium (AA/KK/QQ/JJ/TT/AK) : raise gros systématiquement
            action = _construire_raise(legales, etat, joueur, 3.0)
            return action or _get_action(legales, TypeAction.CALL, TypeAction.CHECK)

        elif force == 3:
            # Fort : raise en open, call si relancé (jamais fold une bonne main)
            if not face_a_raise:
                action = _construire_raise(legales, etat, joueur, 2.5)
                if action:
                    return action
            # Appelle trop souvent face à un raise (erreur classique)
            a_payer = etat.mise_courante - joueur.mise_tour
            if a_payer <= joueur.stack * 0.40:
                return _get_action(legales, TypeAction.CALL, TypeAction.CHECK)
            return _get_action(legales, TypeAction.FOLD, TypeAction.CHECK)

        elif force == 2:
            # Jouable : limp ou check, fold face à raise sauf paires
            r1 = TreysCard.get_rank_int(joueur.cartes[0]) if (
                _TREYS_DISPONIBLE and joueur.cartes) else 0
            r2 = TreysCard.get_rank_int(joueur.cartes[1]) if (
                _TREYS_DISPONIBLE and len(joueur.cartes) > 1) else 0
            est_paire = (r1 == r2)
            if face_a_raise and not est_paire:
                return _get_action(legales, TypeAction.FOLD, TypeAction.CHECK)
            return _get_action(legales, TypeAction.CHECK, TypeAction.CALL, TypeAction.FOLD)

        else:
            # Déchets / spéculatif : fold ou check gratuit
            return _get_action(legales, TypeAction.CHECK, TypeAction.FOLD)

    # ── Postflop ───────────────────────────────────────────────────────────

    def _postflop(self, etat: EtatJeu, joueur: Joueur,
                   legales: List[Action]) -> Action:
        force = _force_postflop(joueur.cartes, etat.board)
        face_a_mise = any(a.type == TypeAction.CALL for a in legales)

        if force > 0.72:
            # Très forte main : value-bet 75 % pot, sizing peu varié
            action = _construire_raise(legales, etat, joueur, 0.75)
            return action or _get_action(legales, TypeAction.CALL, TypeAction.CHECK)

        elif force > 0.45:
            # Main moyenne (top pair faible kicker, etc.) :
            # ERREUR CLASSIQUE — appelle trop souvent
            if not face_a_mise:
                # Bet 75 % des fois avec main correcte
                if self._rng.random() < 0.75:
                    action = _construire_raise(legales, etat, joueur, 0.75)
                    if action:
                        return action
            else:
                # Call presque toujours, même si pot odds défavorables
                a_payer = etat.mise_courante - joueur.mise_tour
                if a_payer <= etat.pot * 0.75:   # appelle jusqu'à ¾ pot
                    return _get_action(legales, TypeAction.CALL)
                return _get_action(legales, TypeAction.FOLD)
            return _get_action(legales, TypeAction.CHECK)

        elif force > 0.28:
            # Main faible : semi-bluff 20 % (tirage), sinon check
            if not face_a_mise and self._rng.random() < 0.20:
                action = _construire_raise(legales, etat, joueur, 0.60)
                if action:
                    return action
            if face_a_mise:
                a_payer = etat.mise_courante - joueur.mise_tour
                if a_payer <= etat.pot * 0.25:   # call uniquement si très bon prix
                    return _get_action(legales, TypeAction.CALL)
                return _get_action(legales, TypeAction.FOLD)
            return _get_action(legales, TypeAction.CHECK)

        else:
            # Air : check ou fold — le régulier ne bluff presque jamais
            return _get_action(legales, TypeAction.CHECK, TypeAction.FOLD)

    def __repr__(self) -> str:
        return "AgentRegulier"


# =============================================================================
# MOTEUR DE SELF-PLAY
# =============================================================================

class MoteurSelfPlay:
    """
    Simule des parties complètes de Texas Hold'em entre agents IA.

    Pas d'interface graphique — uniquement la boucle de jeu interne.
    Conçu pour être rapide et produire des statistiques reproductibles.

    Usage typique
    -------------
        moteur = MoteurSelfPlay(agents=[agent_axiom, AgentAleatoire(), AgentCallOnly()])
        resultat = moteur.jouer_parties(nb_parties=100, verbose=False)
        print(resultat.winrate_bb100(0, grande_blinde=20))

    Paramètres de construction
    --------------------------
    agents : list
        Liste de 3 agents (dans l'ordre BTN, SB, BB).
        Chaque agent doit avoir une méthode choisir_action(etat, joueur, legales).
    stacks_depart : int
        Stack de départ pour chaque joueur (défaut : STACK_DEPART de settings.py).
    graine : int
        Graine aléatoire pour la reproductibilité (None = aléatoire).
    blindes_fixes : tuple
        (petite_blinde, grande_blinde) fixes pour toutes les mains.
        Si None, utilise la structure de blindes croissantes de settings.py.
    """

    def __init__(self,
                 agents         : list,
                 stacks_depart  : int             = STACK_DEPART,
                 graine         : Optional[int]   = None,
                 blindes_fixes  : Optional[tuple] = (10, 20)):

        if len(agents) != NB_JOUEURS:
            raise ValueError(f"Il faut exactement {NB_JOUEURS} agents, "
                             f"reçu {len(agents)}.")

        self.agents        = agents
        self.stacks_depart = stacks_depart
        self.blindes_fixes = blindes_fixes

        if graine is not None:
            random.seed(graine)

        # Petite et grande blinde de référence (pour bb/100)
        pb, gb = blindes_fixes if blindes_fixes else (10, 20)
        self.petite_blinde = pb
        self.grande_blinde = gb

    # ==================================================================
    # SIMULATION DE N PARTIES
    # ==================================================================

    def jouer_parties(self,
                       nb_parties  : int  = 100,
                       verbose     : bool = False,
                       verbose_main: bool = False) -> ResultatPartie:
        """
        Joue nb_parties parties complètes et retourne les résultats agrégés.

        nb_parties   : nombre de parties à simuler
        verbose      : afficher un résumé toutes les 10 parties
        verbose_main : afficher le détail de chaque main

        Retourne un ResultatPartie avec les statistiques cumulées.
        """
        debut = time.time()

        gains_total      = [0] * NB_JOUEURS
        nb_mains_total   = 0
        nb_preflop_total = 0
        nb_showdowns     = 0
        historique_all   = []

        for partie_idx in range(nb_parties):
            res = self.jouer_une_partie(verbose=verbose_main)

            for i in range(NB_JOUEURS):
                gains_total[i] += res.gains_nets[i]

            nb_mains_total   += res.nb_mains
            nb_preflop_total += res.nb_mains_preflop
            nb_showdowns     += res.nb_showdowns
            historique_all.extend(res.historique)

            if verbose and (partie_idx + 1) % 10 == 0:
                bb = self.grande_blinde
                print(f"  Partie {partie_idx+1:>4}/{nb_parties} | "
                      f"mains={nb_mains_total:,} | "
                      + " | ".join(
                          f"{self.agents[i]!r}={gains_total[i]:+,}"
                          f" ({ResultatPartie(gains_total, nb_mains_total, 0, 0).winrate_bb100(i, bb):+.1f}bb/100)"
                          for i in range(NB_JOUEURS)
                      ))

        duree = time.time() - debut

        return ResultatPartie(
            gains_nets       = gains_total,
            nb_mains         = nb_mains_total,
            nb_mains_preflop = nb_preflop_total,
            nb_showdowns     = nb_showdowns,
            historique       = historique_all,
            duree_s          = duree,
        )

    # ==================================================================
    # UNE PARTIE COMPLÈTE
    # ==================================================================

    def jouer_une_partie(self, verbose: bool = False) -> ResultatPartie:
        """
        Simule une partie complète jusqu'à ce qu'il reste 1 joueur avec des jetons.

        Pour les simulations (évaluation des winrates), on joue un nombre
        fixe de mains plutôt qu'une partie à élimination, en réinitialisant
        les stacks à chaque main — ce qui est la convention standard
        pour mesurer le bb/100.

        Retourne un ResultatPartie pour cette seule partie.
        """
        gains_nets     = [0] * NB_JOUEURS
        nb_mains       = 0
        nb_preflop     = 0
        nb_showdowns   = 0
        historique     = []

        # Jouer exactement 1 main (le moteur jouer_parties boucle)
        pb = self.petite_blinde
        gb = self.grande_blinde

        # Créer les joueurs
        joueurs = [
            Joueur(f"J{i}", TypeJoueur.AXIOM, self.stacks_depart, i)
            for i in range(NB_JOUEURS)
        ]

        etat = EtatJeu(joueurs, pb, gb)

        # Capturer les stacks AVANT les blindes (pour un gain net correct)
        stacks_avant = [j.stack for j in joueurs]

        etat.nouvelle_main()

        if verbose:
            etat.afficher()

        res_main = self._jouer_une_main(etat, verbose=verbose)

        # Recalculer les gains nets depuis les stacks d'origine (avant blindes)
        stacks_apres = [j.stack for j in etat.joueurs]
        gains_reels  = [stacks_apres[i] - stacks_avant[i] for i in range(NB_JOUEURS)]

        nb_mains += 1
        if res_main.phase_finale == 'PREFLOP':
            nb_preflop += 1
        if res_main.phase_finale == 'SHOWDOWN':
            nb_showdowns += 1

        for i in range(NB_JOUEURS):
            gains_nets[i] += gains_reels[i]

        historique.append(res_main)

        return ResultatPartie(
            gains_nets       = gains_nets,
            nb_mains         = nb_mains,
            nb_mains_preflop = nb_preflop,
            nb_showdowns     = nb_showdowns,
            historique       = historique,
        )

    # ==================================================================
    # UNE MAIN COMPLÈTE
    # ==================================================================

    def _jouer_une_main(self, etat: EtatJeu,
                         verbose: bool = False) -> ResultatMain:
        """
        Joue une main complète de A à Z sur un EtatJeu initialisé.
        Gère le tour de parole pour chaque street.
        Retourne un ResultatMain avec les gains nets de chaque joueur.
        """
        stacks_avant = [j.stack for j in etat.joueurs]
        nb_actions   = 0
        phase_finale = etat.phase.name

        # Jouer chaque street
        for phase in [Phase.PREFLOP, Phase.FLOP, Phase.TURN, Phase.RIVER]:
            if etat.phase == Phase.TERMINEE:
                break
            if len(etat.joueurs_actifs_dans_main()) < 2:
                break

            nb_actions  += self._jouer_tour(etat, verbose)
            phase_finale = etat.phase.name

            if (etat.phase not in (Phase.SHOWDOWN, Phase.TERMINEE)
                    and len(etat.joueurs_actifs_dans_main()) >= 2):
                etat.passer_phase_suivante()

        # Résoudre la main (distribuer le pot)
        gagnant_idx = self._resoudre(etat, verbose)

        if len(etat.joueurs_actifs_dans_main()) <= 1:
            phase_finale = 'PREFLOP' if len(etat.board) == 0 else etat.phase.name
        else:
            phase_finale = 'SHOWDOWN'

        stacks_apres = [j.stack for j in etat.joueurs]
        gains_nets   = [stacks_apres[i] - stacks_avant[i]
                        for i in range(NB_JOUEURS)]

        return ResultatMain(
            numero       = 0,
            gains_nets   = gains_nets,
            pot_final    = etat.pot,
            nb_actions   = nb_actions,
            phase_finale = phase_finale,
            gagnant_idx  = gagnant_idx,
        )

    def _jouer_tour(self, etat: EtatJeu, verbose: bool = False) -> int:
        """
        Joue un tour de parole complet (une street).
        Retourne le nombre d'actions effectuées.

        Condition de fin correcte :
          - Tous les joueurs encore actifs ont agi au moins une fois CE tour
          - ET toutes leurs mises sont égales à mise_courante
        Cela corrige deux bugs de l'ancienne version :
          1. L'option BB préflop (BB devait pouvoir checker/relancer même si
             BTN et SB ont juste callé, rendant toutes les mises égales).
          2. Le post-flop qui ne jouait jamais (mise_courante=0 → condition
             "toutes mises égales" vraie dès le départ → 0 action par street).
        """
        nb_actions = 0
        max_actions = NB_JOUEURS * 6   # sécurité anti-boucle infinie
        ont_agi    = set()             # id() des joueurs ayant agi ce tour

        while nb_actions < max_actions:
            # ── Condition de fin (vérifiée EN PREMIER pour éviter toute boucle) ──
            # Tous les joueurs encore actifs ont agi au moins une fois CE tour
            # ET toutes leurs mises sont égales à mise_courante.
            pouvant_agir = etat.joueurs_pouvant_agir()
            if not pouvant_agir:
                break
            tous_ont_agi = all(id(j) in ont_agi              for j in pouvant_agir)
            mises_egales = all(j.mise_tour == etat.mise_courante for j in pouvant_agir)
            if tous_ont_agi and mises_egales:
                break

            joueur = etat.joueur_actif()
            if not joueur.peut_agir:
                etat.passer_au_suivant()
                continue

            legales = actions_legales(
                joueur,
                mise_a_suivre  = etat.mise_courante,
                pot            = etat.pot,
                mise_min_raise = etat.mise_min_raise,
            )

            # Obtenir l'action via l'agent correspondant
            agent = self.agents[etat.joueurs.index(joueur)]
            action = agent.choisir_action(etat, joueur, legales)

            # Appliquer l'action
            self._appliquer_action(etat, joueur, action)
            etat.enregistrer_action(joueur, action)

            # OFT — notifier tous les agents de l'action jouée
            seat_joueur = etat.joueurs.index(joueur)
            phase_nom   = etat.phase.name.lower()
            for ag in self.agents:
                if hasattr(ag, 'enregistrer_action'):
                    ag.enregistrer_action(
                        seat        = seat_joueur,
                        action      = action.type.name,
                        street      = phase_nom,
                        contexte    = {'est_cbet_opp': False},
                    )

            if verbose:
                print(f"    {joueur.nom} → {action}")

            ont_agi.add(id(joueur))
            etat.passer_au_suivant()
            nb_actions += 1

            if len(etat.joueurs_actifs_dans_main()) <= 1:
                break

        return nb_actions

    def _tour_termine(self, etat: EtatJeu) -> bool:
        """
        Conservé pour compatibilité. Non utilisé par _jouer_tour
        (la logique est désormais inline avec le suivi 'ont_agi').
        """
        pouvant_agir = etat.joueurs_pouvant_agir()
        if not pouvant_agir:
            return True
        return all(j.mise_tour == etat.mise_courante for j in pouvant_agir)

    def _appliquer_action(self, etat: EtatJeu, joueur: Joueur, action: Action):
        """Applique une action sur l'état (repris de engine/game.py)."""
        if action.type == TypeAction.FOLD:
            joueur.statut = StatutJoueur.FOLD

        elif action.type == TypeAction.CHECK:
            pass

        elif action.type == TypeAction.CALL:
            a_payer    = etat.mise_courante - joueur.mise_tour
            mise_reelle = joueur.miser(a_payer)
            etat.pot   += mise_reelle

        elif action.type in (TypeAction.RAISE, TypeAction.ALL_IN):
            a_payer    = action.montant - joueur.mise_tour
            mise_reelle = joueur.miser(a_payer)
            etat.pot   += mise_reelle
            if action.montant > etat.mise_courante:
                increment          = action.montant - etat.mise_courante
                etat.mise_min_raise = max(etat.mise_min_raise, increment)
                etat.mise_courante  = action.montant

    def _resoudre(self, etat: EtatJeu, verbose: bool = False) -> int:
        """
        Distribue le pot aux gagnants.
        Retourne l'index du gagnant (-1 si égalité ou plusieurs gagnants).
        """
        actifs = etat.joueurs_actifs_dans_main()

        if len(actifs) == 1:
            gagnant = actifs[0]
            gagnant.recevoir(etat.pot)
            if verbose:
                print(f"    → {gagnant.nom} remporte {etat.pot} (tous fold)")
            return etat.joueurs.index(gagnant)

        # Showdown
        gagnants       = determiner_gagnants(actifs, etat.board)
        gain_par_gagnant = etat.pot // len(gagnants)
        for g in gagnants:
            g.recevoir(gain_par_gagnant)
            if verbose:
                print(f"    → {g.nom} remporte {gain_par_gagnant}")

        if len(gagnants) == 1:
            return etat.joueurs.index(gagnants[0])
        return -1   # égalité


# =============================================================================
# FONCTION UTILITAIRE PRINCIPALE
# =============================================================================

def simuler(agents         : list,
            nb_mains       : int   = 1000,
            stacks_depart  : int   = STACK_DEPART,
            blindes_fixes  : tuple = (10, 20),
            graine         : int   = None,
            verbose        : bool  = True) -> ResultatPartie:
    """
    Lance une simulation de nb_mains mains et affiche les résultats.

    agents       : liste de 3 agents (chacun avec choisir_action())
    nb_mains     : nombre de mains à simuler
    stacks_depart: stack de départ (réinitialisé à chaque main)
    blindes_fixes: (petite_blinde, grande_blinde)
    graine       : graine aléatoire (None = non reproductible)
    verbose      : afficher le résumé final

    Retourne un ResultatPartie.
    """
    moteur = MoteurSelfPlay(
        agents        = agents,
        stacks_depart = stacks_depart,
        blindes_fixes = blindes_fixes,
        graine        = graine,
    )
    resultat = moteur.jouer_parties(nb_parties=nb_mains, verbose=False)

    if verbose:
        _afficher_resultats(agents, resultat, blindes_fixes[1])

    return resultat


def _afficher_resultats(agents, resultat: ResultatPartie, grande_blinde: int):
    """Affiche un tableau de résultats lisible."""
    print(f"\n{'═'*60}")
    print(f"  AXIOM — Résultats Self-Play")
    print(f"{'═'*60}")
    print(f"  Mains jouées     : {resultat.nb_mains:,}")
    print(f"  Preflop folds    : {resultat.nb_mains_preflop:,} "
          f"({100*resultat.nb_mains_preflop//max(resultat.nb_mains,1)}%)")
    print(f"  Showdowns        : {resultat.nb_showdowns:,} "
          f"({100*resultat.nb_showdowns//max(resultat.nb_mains,1)}%)")
    print(f"  Durée            : {resultat.duree_s:.2f}s "
          f"({resultat.nb_mains/max(resultat.duree_s,0.001):.0f} mains/s)")
    print()
    print(f"  {'Agent':20} | {'Gain net':>10} | {'bb/100':>8}")
    print(f"  {'─'*20}-+-{'─'*10}-+-{'─'*8}")
    for i, agent in enumerate(agents):
        bb100 = resultat.winrate_bb100(i, grande_blinde)
        print(f"  {str(agent):20} | {resultat.gains_nets[i]:>+10,} | {bb100:>+8.1f}")
    print(f"{'═'*60}\n")


# =============================================================================
# TEST RAPIDE
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  AXIOM — Test self_play.py (Phase 10)")
    print("="*60)

    from ai.agent import AgentAXIOM

    # ── Test 1 : AXIOM vs deux agents aléatoires ──────────────────────────
    print("\n  Test 1 : AXIOM vs AgentAleatoire vs AgentCallOnly (100 mains)")
    agent_axiom = AgentAXIOM(mode_deterministe=False)
    agents = [agent_axiom, AgentAleatoire(graine=42), AgentCallOnly()]

    resultat = simuler(
        agents        = agents,
        nb_mains      = 100,
        blindes_fixes = (10, 20),
        graine        = 42,
        verbose       = True,
    )

    assert resultat.nb_mains == 100, f"Attendu 100 mains, obtenu {resultat.nb_mains}"
    somme_gains = sum(resultat.gains_nets)
    assert abs(somme_gains) <= 100 * NB_JOUEURS, \
        f"Les gains ne sont pas nuls : {resultat.gains_nets} (somme={somme_gains})"

    # ── Test 2 : vitesse ───────────────────────────────────────────────────
    print("  Test 2 : vitesse (500 mains, 3 agents aléatoires)")
    agents_rapides = [AgentAleatoire(42), AgentAleatoire(43), AgentAleatoire(44)]
    debut = time.time()
    res2 = simuler(agents_rapides, nb_mains=500, verbose=True)
    duree = time.time() - debut
    print(f"  Vitesse : {500/duree:.0f} mains/s")
    assert res2.nb_mains == 500

    # ── Test 3 : agents de référence ──────────────────────────────────────
    print("  Test 3 : AgentRaiseOnly vs deux AgentCallOnly (50 mains)")
    agents3 = [AgentRaiseOnly(), AgentCallOnly(), AgentCallOnly()]
    res3 = simuler(agents3, nb_mains=50, verbose=True)
    assert res3.nb_mains == 50

    # ── Test 4 : bots semi-pro TAG / LAG / Régulier ────────────────────────
    print("\n  Test 4 : TAG vs LAG vs Régulier (200 mains)")
    agents4 = [AgentTAG(graine=1), AgentLAG(graine=2), AgentRegulier(graine=3)]
    res4 = simuler(agents4, nb_mains=200, blindes_fixes=(10, 20),
                   graine=99, verbose=True)
    assert res4.nb_mains == 200, f"Attendu 200 mains, obtenu {res4.nb_mains}"
    somme4 = sum(res4.gains_nets)
    assert abs(somme4) <= 200 * NB_JOUEURS, \
        f"Conservation des jetons violée : {res4.gains_nets}"
    print("  ✅ AgentTAG, AgentLAG, AgentRegulier fonctionnent correctement")

    # ── Test 5 : AXIOM vs TAG vs LAG ──────────────────────────────────────
    print("\n  Test 5 : AXIOM vs TAG vs Régulier (100 mains)")
    agents5 = [agent_axiom, AgentTAG(graine=10), AgentRegulier(graine=20)]
    res5 = simuler(agents5, nb_mains=100, blindes_fixes=(10, 20),
                   graine=77, verbose=True)
    assert res5.nb_mains == 100

    print("  ✅ Tous les tests self_play.py sont passés !")
    print("="*60 + "\n")
