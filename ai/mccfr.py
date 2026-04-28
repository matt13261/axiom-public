# =============================================================================
# AXIOM — ai/mccfr.py
# Algorithme CFR (Counterfactual Regret Minimization)
#
# Phase 3a : Vanilla CFR sur Kuhn Poker (validation de l'algorithme).
#            Le moteur Kuhn Poker est intégré directement dans ce fichier.
# Phase 3b : External Sampling MCCFR sur Texas Hold'em (blueprint stratégie).
#
# ─────────────────────────────────────────────────────────────────────────────
# PRINCIPE DE CFR
# ─────────────────────────────────────────────────────────────────────────────
# CFR est l'algorithme qui a permis de résoudre le poker (Libratus, Pluribus).
#
# Intuition : imaginer qu'un joueur rejoue la même main des milliers de fois.
# À chaque fois, il note "si j'avais joué autrement, j'aurais gagné X de plus".
# Ce manque à gagner s'appelle le REGRET. CFR accumule ces regrets et ajuste
# la stratégie pour minimiser les regrets futurs.
#
# Regret Matching :
#   → Si regret(action A) > 0 : jouer A plus souvent
#   → Si regret(action A) ≤ 0 : ne jamais jouer A
#   → Prob(A) = regret+(A) / somme des regrets positifs
#   → Si tous les regrets ≤ 0 : stratégie uniforme (on ne sait pas encore)
#
# Convergence :
#   Après T itérations, la STRATÉGIE MOYENNE (non la courante) converge vers
#   l'Équilibre de Nash. L'exploitabilité décroît en O(1/√T).
#
# ─────────────────────────────────────────────────────────────────────────────
# VANILLA CFR (Phase 3a — Kuhn Poker)
# ─────────────────────────────────────────────────────────────────────────────
# Explore TOUT l'arbre de jeu à chaque itération.
# Pas d'échantillonnage → convergence déterministe et vérifiable.
# Idéal pour valider l'implémentation sur un jeu dont la solution est connue.
#
# ─────────────────────────────────────────────────────────────────────────────
# EXTERNAL SAMPLING MCCFR (Phase 3b — Texas Hold'em)
# ─────────────────────────────────────────────────────────────────────────────
# Pour Hold'em, l'arbre est gigantesque → on ne peut pas tout explorer.
# External Sampling : on échantillonne les nœuds adverses et les cartes (chance),
# mais on explore TOUTES les actions du joueur traversant.
# → Beaucoup plus rapide, convergence stochastique mais garantie.
# → C'est la variante utilisée dans Pluribus.
#
# Référence : "An Introduction to Counterfactual Regret Minimization"
#             Todd W. Neller & Marc Lanctot, AAAI 2013
# =============================================================================


# =============================================================================
# PARTIE 1 — MOTEUR KUHN POKER (intégré ici pour la validation de CFR)
#
# Kuhn Poker est le jeu le plus simple avec bluff et information imparfaite.
# Il permet de valider CFR car sa solution analytique est connue.
#
# RÈGLES :
#   • 2 joueurs, 3 cartes : J (Jack=0), Q (Queen=1), K (King=2)
#   • Chaque joueur paye 1 jeton d'ante → pot de départ = 2
#   • Joueur 0 agit EN PREMIER
#   • Deux actions : p = passer (check/fold) | b = miser (bet/call 1 jeton)
#
# SÉQUENCES TERMINALES : pp | bp | bb | pbp | pbb
#
# SOLUTION ANALYTIQUE (Nash) :
#   Valeur du jeu pour J0 = -1/18 ≈ -0.05556
#   J0 : Q ne mise jamais | K mise 3× plus souvent que J
#   J1 : K mise toujours après check | J fold toujours face à une mise
#        J bluff ≈ 1/3 après check (contrainte d'indifférence de J0 avec Q)
# =============================================================================

# Constantes Kuhn Poker (préfixées _KP_ pour éviter les collisions de noms)
_KP_CARTES      = [0, 1, 2]
_KP_NOM_CARTE   = {0: 'J', 1: 'Q', 2: 'K'}
_KP_ACTIONS     = ['p', 'b']          # p = passer | b = miser
_KP_NB_ACTIONS  = 2
_KP_VALEUR_NASH = -1.0 / 18.0         # ≈ -0.055556


def _kp_est_terminal(historique: str) -> bool:
    """Retourne True si la séquence est un nœud terminal du Kuhn Poker."""
    return historique in ('pp', 'bp', 'bb', 'pbp', 'pbb')


def _kp_joueur_actif(historique: str) -> int:
    """Joueur 0 aux positions paires, joueur 1 aux positions impaires."""
    return len(historique) % 2


def _kp_gain_terminal(carte_j0: int, carte_j1: int, historique: str) -> float:
    """
    Gain NET pour le joueur 0 dans un nœud terminal.
    Positif = J0 gagne | Négatif = J0 perd.
    """
    j0_gagne = (carte_j0 > carte_j1)
    if historique == 'pp':
        return  1.0 if j0_gagne else -1.0
    elif historique == 'bp':
        return  1.0                             # J1 fold → J0 gagne l'ante
    elif historique == 'pbp':
        return -1.0                             # J0 fold → J1 gagne l'ante
    elif historique == 'bb':
        return  2.0 if j0_gagne else -2.0
    elif historique == 'pbb':
        return  2.0 if j0_gagne else -2.0
    raise ValueError(f"Séquence Kuhn Poker non reconnue : '{historique}'")


def _kp_cle_infoset(carte: int, historique: str) -> str:
    """
    Clé d'information set pour Kuhn Poker.
    Ex : (2, "p") → "Kp"  |  (0, "") → "J"
    """
    return _KP_NOM_CARTE[carte] + historique


def _kp_tous_les_deals() -> list:
    """Retourne les 6 deals possibles (c_j0, c_j1) sans remise."""
    return [(c0, c1) for c0 in _KP_CARTES for c1 in _KP_CARTES if c0 != c1]


# =============================================================================
# PARTIE 2 — NŒUD CFR (commun à Kuhn Poker et Hold'em)
# =============================================================================

class NoeudCFR:
    """
    Représente un nœud de l'arbre CFR = un information set.

    Stocke pour chaque action :
    ─────────────────────────────────────────────────────────────────
    regrets_cumules[a] :
        Somme cumulée des regrets contrefactuels pour l'action a.
        Regret positif = on aurait dû jouer cette action plus souvent.

    strategie_somme[a] :
        Somme cumulée des stratégies pondérées par la probabilité de reach.
        La stratégie MOYENNE = strategie_somme normalisée → converge vers Nash.
    ─────────────────────────────────────────────────────────────────
    """

    def __init__(self, nb_actions: int = _KP_NB_ACTIONS):
        self.nb_actions      = nb_actions
        self.regrets_cumules = [0.0] * nb_actions
        self.strategie_somme = [0.0] * nb_actions
        self.nb_visites      = 0

    def strategie_courante(self, proba_reach: float) -> list:
        """
        Calcule la stratégie COURANTE par Regret Matching et accumule
        la somme pour le calcul de la stratégie moyenne finale.

        Regret Matching :
            prob(a) = max(regret(a), 0) / somme_regrets_positifs
            Si tous regrets ≤ 0 → stratégie uniforme (exploration)

        proba_reach : probabilité que ce joueur atteigne ce nœud.
        Retourne    : liste de probabilités (somme = 1.0).
        """
        regrets_pos = [max(r, 0.0) for r in self.regrets_cumules]
        somme_pos   = sum(regrets_pos)

        if somme_pos > 0:
            strategie = [r / somme_pos for r in regrets_pos]
        else:
            strategie = [1.0 / self.nb_actions] * self.nb_actions

        for i in range(self.nb_actions):
            self.strategie_somme[i] += proba_reach * strategie[i]

        self.nb_visites += 1
        return strategie

    def strategie_moyenne(self) -> list:
        """
        Stratégie MOYENNE cumulée — la stratégie finale utilisée en production.
        Converge vers l'Équilibre de Nash après suffisamment d'itérations.
        """
        somme = sum(self.strategie_somme)
        if somme > 0:
            return [s / somme for s in self.strategie_somme]
        return [1.0 / self.nb_actions] * self.nb_actions

    def __repr__(self):
        strat = self.strategie_moyenne()
        return (f"NoeudCFR(strat={[round(s, 3) for s in strat]} | "
                f"regrets={[round(r, 3) for r in self.regrets_cumules]} | "
                f"visites={self.nb_visites})")


# =============================================================================
# PARTIE 3 — VANILLA CFR POUR KUHN POKER
# =============================================================================

class CFRKUHN:
    """
    Vanilla CFR pour le Kuhn Poker à 2 joueurs.

    Vanilla CFR = exploration exhaustive de TOUT l'arbre à chaque itération.
    Convergence déterministe et vérifiable contre la solution analytique connue.

    Usage :
        cfr = CFRKUHN()
        cfr.entrainer(nb_iterations=10_000, verbose=True)
        cfr.afficher_strategie()
    """

    def __init__(self):
        self.noeuds: dict = {}   # clé_infoset (str) → NoeudCFR

    def _obtenir_noeud(self, cle: str) -> NoeudCFR:
        """Récupère ou crée un nœud CFR pour la clé d'infoset donnée."""
        if cle not in self.noeuds:
            self.noeuds[cle] = NoeudCFR(_KP_NB_ACTIONS)
        return self.noeuds[cle]

    # ------------------------------------------------------------------
    # TRAVERSÉE CFR RÉCURSIVE
    # ------------------------------------------------------------------

    def cfr(self, carte_j0: int, carte_j1: int, historique: str,
            proba_j0: float, proba_j1: float) -> float:
        """
        Traversée récursive Vanilla CFR.

        À chaque appel, explore TOUTES les branches et calcule les regrets
        contrefactuels (ce qu'on aurait gagné avec une autre action).

        Paramètres :
            carte_j0, carte_j1 : cartes distribuées
            historique         : actions jouées jusqu'ici (ex: "pb")
            proba_j0           : probabilité de reach du joueur 0
                                 (produit des probas de ses actions passées)
            proba_j1           : probabilité de reach du joueur 1

        Retourne : valeur espérée du nœud pour le JOUEUR 0.
        """
        # ── Nœud terminal ────────────────────────────────────────────
        if _kp_est_terminal(historique):
            return _kp_gain_terminal(carte_j0, carte_j1, historique)

        # ── Joueur actif et information set ──────────────────────────
        joueur = _kp_joueur_actif(historique)
        carte  = carte_j0 if joueur == 0 else carte_j1
        cle    = _kp_cle_infoset(carte, historique)
        noeud  = self._obtenir_noeud(cle)

        # ── Stratégie courante par regret matching ────────────────────
        proba_reach_joueur = proba_j0 if joueur == 0 else proba_j1
        strategie = noeud.strategie_courante(proba_reach_joueur)

        # ── Exploration exhaustive de toutes les actions ──────────────
        valeurs_actions = [0.0] * _KP_NB_ACTIONS
        for i, action in enumerate(_KP_ACTIONS):
            h_suivant = historique + action
            if joueur == 0:
                valeurs_actions[i] = self.cfr(
                    carte_j0, carte_j1, h_suivant,
                    proba_j0 * strategie[i],   # reach J0 mis à jour
                    proba_j1)
            else:
                valeurs_actions[i] = self.cfr(
                    carte_j0, carte_j1, h_suivant,
                    proba_j0,
                    proba_j1 * strategie[i])   # reach J1 mis à jour

        # ── Valeur du nœud (espérance selon stratégie courante) ───────
        valeur_noeud = sum(strategie[i] * valeurs_actions[i]
                           for i in range(_KP_NB_ACTIONS))

        # ── Mise à jour des regrets ───────────────────────────────────
        # regret(a) += proba_adversaire × (val(a) - val_nœud)
        # proba_adversaire = reach de TOUS sauf le joueur actif
        proba_adversaire = proba_j1 if joueur == 0 else proba_j0
        for i in range(_KP_NB_ACTIONS):
            if joueur == 0:
                regret_i = proba_adversaire * (valeurs_actions[i] - valeur_noeud)
            else:
                # J1 : inverser le signe (valeurs exprimées du point de vue J0)
                regret_i = proba_adversaire * (-(valeurs_actions[i] - valeur_noeud))
            # CFR+ : plancher à 0 (convergence ~2× plus rapide)
            noeud.regrets_cumules[i] = max(0.0, noeud.regrets_cumules[i] + regret_i)

        return valeur_noeud

    # ------------------------------------------------------------------
    # ENTRAÎNEMENT
    # ------------------------------------------------------------------

    def entrainer(self, nb_iterations: int, verbose: bool = False) -> float:
        """
        Lance nb_iterations de Vanilla CFR sur tous les deals possibles.

        À chaque itération, traverse l'arbre complet pour les 6 deals.
        Retourne la valeur moyenne du jeu pour J0 (cible : -1/18 ≈ -0.05556).
        """
        valeur_totale = 0.0
        deals         = _kp_tous_les_deals()
        nb_deals      = len(deals)   # = 6

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Entraînement CFR — Kuhn Poker | {nb_iterations} itérations")
            print(f"  Valeur Nash analytique : {_KP_VALEUR_NASH:.6f}")
            print(f"{'─'*60}")

        for iteration in range(1, nb_iterations + 1):
            for carte_j0, carte_j1 in deals:
                valeur_totale += self.cfr(
                    carte_j0, carte_j1, '',
                    proba_j0=1.0, proba_j1=1.0)

            if verbose:
                pas = max(1, nb_iterations // 10)
                if iteration % pas == 0:
                    valeur_moy = valeur_totale / (iteration * nb_deals)
                    ecart      = abs(valeur_moy - _KP_VALEUR_NASH)
                    print(f"  It. {iteration:6d}/{nb_iterations} | "
                          f"Valeur = {valeur_moy:+.6f} | "
                          f"Écart Nash = {ecart:.6f}")

        valeur_finale = valeur_totale / (nb_iterations * nb_deals)

        if verbose:
            print(f"{'─'*60}")
            print(f"  ✅ Entraînement terminé")
            print(f"  Valeur finale   : {valeur_finale:+.6f}")
            print(f"  Valeur Nash     : {_KP_VALEUR_NASH:+.6f}")
            print(f"  Écart           : {abs(valeur_finale - _KP_VALEUR_NASH):.6f}")
            print(f"  Nœuds créés     : {len(self.noeuds)}")
            print(f"{'─'*60}\n")

        return valeur_finale

    # ------------------------------------------------------------------
    # AFFICHAGE DE LA STRATÉGIE
    # ------------------------------------------------------------------

    def afficher_strategie(self):
        """Affiche la stratégie moyenne de chaque infoset avec interprétation Nash."""
        print(f"\n{'='*65}")
        print(f"  STRATÉGIE FINALE — Kuhn Poker (Équilibre de Nash)")
        print(f"{'='*65}")
        print(f"  {'Infoset':8} | {'Passer':6} | {'Miser':6} | Interprétation")
        print(f"  {'─'*8}-+-{'─'*6}-+-{'─'*6}-+-{'─'*30}")

        interpretations = {
            'J'   : "J0 : bluff rare (≈α, idéalement 0-33%)",
            'Q'   : "J0 : TOUJOURS passer ← Nash pur",
            'K'   : "J0 : value bet fréquent (≈3α)",
            'Jp'  : "J1 : bluff ≈ 1/3 (indifférence Q de J0)",
            'Qp'  : "J1 : passe (main médiane)",
            'Kp'  : "J1 : TOUJOURS miser (value bet)",
            'Jb'  : "J1 : TOUJOURS fold face à une mise",
            'Qb'  : "J1 : call 33% du temps",
            'Kb'  : "J1 : TOUJOURS call",
            'Jpb' : "J0 : TOUJOURS fold (check-raise avec J)",
            'Qpb' : "J0 : call ≈ 1/3 (indifférence J1 avec J)",
            'Kpb' : "J0 : TOUJOURS call",
        }

        ordre = ['J', 'Q', 'K', 'Jp', 'Qp', 'Kp', 'Jb', 'Qb', 'Kb', 'Jpb', 'Qpb', 'Kpb']
        for cle in ordre:
            if cle in self.noeuds:
                strat  = self.noeuds[cle].strategie_moyenne()
                interp = interpretations.get(cle, '')
                print(f"  {cle:8} | {strat[0]:.4f} | {strat[1]:.4f} | {interp}")
        print(f"{'='*65}\n")

    # ------------------------------------------------------------------
    # ÉVALUATION ET MESURE DE CONVERGENCE
    # ------------------------------------------------------------------

    def valeur_du_jeu(self) -> float:
        """Valeur espérée du jeu pour J0 avec la stratégie MOYENNE. Cible : -1/18."""
        total = 0.0
        deals = _kp_tous_les_deals()
        for c0, c1 in deals:
            total += self._evaluer_noeud(c0, c1, '')
        return total / len(deals)

    def _evaluer_noeud(self, c0: int, c1: int, historique: str) -> float:
        """Évaluation récursive avec la stratégie MOYENNE (pas la courante)."""
        if _kp_est_terminal(historique):
            return _kp_gain_terminal(c0, c1, historique)
        joueur = _kp_joueur_actif(historique)
        carte  = c0 if joueur == 0 else c1
        cle    = _kp_cle_infoset(carte, historique)
        strat  = (self.noeuds[cle].strategie_moyenne()
                  if cle in self.noeuds else [0.5, 0.5])
        return sum(strat[i] * self._evaluer_noeud(c0, c1, historique + a)
                   for i, a in enumerate(_KP_ACTIONS))

    def exploitabilite(self) -> float:
        """
        NashConv = gain_BR_J0 + gain_BR_J1.
        Mesure à quel point la stratégie est exploitable.
        Tend vers 0 quand la stratégie converge vers l'Équilibre de Nash.
        """
        return self._meilleure_reponse(0) + self._meilleure_reponse(1)

    def _meilleure_reponse(self, joueur_br: int) -> float:
        """Best response de joueur_br contre la stratégie MOYENNE de l'adversaire."""
        total = 0.0
        for c_br in _KP_CARTES:
            total += self._rcfv_br(c_br, '', joueur_br)
        return total / 6.0   # P(c_br)=1/3, 2 valeurs c_adv → normaliser par 6

    def _reach_adversaire(self, c_adv: int, historique: str, joueur_adv: int) -> float:
        """
        Probabilité de reach de joueur_adv avec la carte c_adv
        jusqu'à l'historique donné (selon sa stratégie MOYENNE).
        """
        prob = 1.0
        for i in range(len(historique)):
            if i % 2 == joueur_adv:
                cle   = _kp_cle_infoset(c_adv, historique[:i])
                idx   = _KP_ACTIONS.index(historique[i])
                strat = (self.noeuds[cle].strategie_moyenne()
                         if cle in self.noeuds else [0.5, 0.5])
                prob *= strat[idx]
        return prob

    def _rcfv_br(self, c_br: int, historique: str, joueur_br: int) -> float:
        """
        Reach-Weighted Counterfactual Value pour la best response.

        Le joueur_br choisit la meilleure action GLOBALEMENT pour son infoset
        (sans voir la carte adverse) — contrainte d'information imparfaite du poker.
        """
        joueur_adv = 1 - joueur_br
        if _kp_est_terminal(historique):
            total = 0.0
            for c_adv in _KP_CARTES:
                if c_adv == c_br:
                    continue
                reach = self._reach_adversaire(c_adv, historique, joueur_adv)
                gain  = (_kp_gain_terminal(c_br, c_adv, historique)
                         if joueur_br == 0
                         else -_kp_gain_terminal(c_adv, c_br, historique))
                total += reach * gain
            return total
        if _kp_joueur_actif(historique) == joueur_br:
            # Meilleure action pour le joueur_br (sans voir c_adv)
            return max(self._rcfv_br(c_br, historique + a, joueur_br)
                       for a in _KP_ACTIONS)
        # Nœud adversaire : sommer sur ses actions (reach déjà intégré aux terminaux)
        return sum(self._rcfv_br(c_br, historique + a, joueur_br)
                   for a in _KP_ACTIONS)

    def reinitialiser(self):
        """Remet tous les nœuds à zéro pour recommencer l'entraînement."""
        self.noeuds.clear()

    def __repr__(self):
        return f"CFRKUHN(nœuds={len(self.noeuds)})"
# =============================================================================
# PARTIE 4 — EXTERNAL SAMPLING MCCFR TEXAS HOLD'EM (Phase 3b)
#
# Implémentation de l'External Sampling MCCFR adapté au Texas Hold'em
# No Limit 3 joueurs. Construit la stratégie blueprint d'AXIOM.
#
# ─────────────────────────────────────────────────────────────────────────────
# EXTERNAL SAMPLING MCCFR — PRINCIPE
# ─────────────────────────────────────────────────────────────────────────────
# À chaque itération, on échantillonne un deal aléatoire (cartes + board).
# Pour le joueur traversant i : on explore TOUTES ses actions (comme Vanilla).
# Pour les adversaires j ≠ i  : on échantillonne UNE seule action.
# 3 traversées par itération (une par joueur traversant → couverture complète).
#
# Avantages vs Vanilla CFR :
#   → L'espace est trop grand pour tout explorer (10^18 états en Hold'em)
#   → ES-MCCFR converge stochastiquement mais garantit l'équilibre de Nash
#   → Variante utilisée dans Pluribus (IA qui a battu les pros en 2019)
#
# État de jeu léger :
#   On n'utilise PAS EtatJeu ici pour des raisons de performance.
#   Un dict Python est copié en O(1) et évite la surcharge des classes.
#   Les buckets (coûteux à calculer) sont pré-calculés une fois par deal
#   et partagés (read-only) entre toutes les copies d'état de la même traversée.
#
# Format des clés d'infoset :
#   Compatible avec abstraction/info_set.py :
#   "PHASE|pos=X|bucket=Y|pot=Z|stacks=(a,b,c)|hist=ABC"
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS SUPPLÉMENTAIRES POUR LA PARTIE 4
# (placés ici car la Partie 4 est ajoutée à la suite d'un fichier existant)
# ─────────────────────────────────────────────────────────────────────────────
import random as _rand_h
from treys import Deck as _TreysDeckH, Evaluator as _EvalH
from config.settings import TAILLES_MISE_PREFLOP as _TAILLES_PREFLOP_H, TAILLES_MISE_POSTFLOP as _TAILLES_POSTFLOP_H, ALL_IN as _ALL_IN_H
from abstraction.card_abstraction import AbstractionCartes as _AbsCartes, AbstractionCartesV2 as _AbsCartesV2
from abstraction.action_abstraction import AbstractionAction as _AbsAction
from abstraction.info_set import (
    _normaliser, PALIERS_POT, PALIERS_STACK, PALIERS_STACK_SPIN_RUSH,
    _discretiser_raise_frac, _format_hist_avec_cap,
)
from engine.actions import Action as _ActionH, TypeAction as _TypeActionH


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES PARTIE 4
# ─────────────────────────────────────────────────────────────────────────────

# Point 7 — Élagage epsilon (Pluribus)
# Dans _PROBA_ELAGAGE = 95% des traversées adverses, les actions dont la
# probabilité < _EPSILON_ELAGAGE sont ignorées pour accélérer l'entraînement.
# Les 5% restants explorent toutes les actions normalement (non-biaisé).
_EPSILON_ELAGAGE = 0.05   # seuil de probabilité minimale pour ne pas élaguer
_PROBA_ELAGAGE   = 0.95   # proportion des traversées avec élagage actif

# Statuts d'un joueur dans l'état léger (int pour performance)
_H_ACTIF  = 0   # encore dans la main, peut agir
_H_FOLD   = 1   # s'est couché cette main
_H_ALLIN  = 2   # a misé tout son stack, ne peut plus agir

# Phases de jeu
_H_PREFLOP = 0
_H_FLOP    = 1
_H_TURN    = 2
_H_RIVER   = 3
_H_NOM_PHASE = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']

# Ordre d'action
# Preflop  : BTN(0) agit en premier, puis SB(1), puis BB(2) avec l'option
# Postflop : SB(1) agit en premier, puis BB(2), puis BTN(0)
_ORDRE_PREFLOP  = [0, 1, 2]
_ORDRE_POSTFLOP = [1, 2, 0]

# Évaluateur treys — singleton instancié une seule fois pour toute la session
_eval_holdem = _EvalH()


# =============================================================================
# BIAIS DE CONTINUATION (Pluribus — Brown & Sandholm 2019)
# =============================================================================
#
# Idée : lors d'une traversée CFR, on ajoute un bonus α·pot à l'utilité des
# actions d'une CATÉGORIE choisie (FOLD, CALL, ou RAISE). Ce bonus "nudge"
# la convergence CFR vers une stratégie qui joue plus souvent les actions
# de cette catégorie, sans briser la cohérence interne de l'équilibre :
# les actions de la même catégorie conservent leurs regrets RELATIFS.
#
# En entraînant 4 blueprints en parallèle — baseline, fold, call, raise —
# on obtient 4 "continuations" mixables en temps réel : un adversaire qui
# apprend à exploiter l'une d'elles s'expose aux trois autres (non-
# exploitabilité accrue, cf. Pluribus vs humains pros, 2019).
#
# ⚠️ IMPORTANT : le biais est LOCAL au regret update. La valeur retournée
# vers l'appelant est l'utilité NON biaisée — sinon le biais se propagerait
# dans l'arbre et corromprait l'équilibre aux nœuds ancestraux.
#
# Valeur recommandée (Pluribus paper) : alpha = 0.05
# =============================================================================

class BiaisContinuation:
    """
    Biais de continuation pour produire une variante de blueprint.

    Attributs :
      categorie : 'fold' | 'call' | 'raise' | None   (None → baseline, pas de biais)
      alpha     : magnitude du bonus (typique 0.05 ; 0 désactive même si catégorie ≠ None)

    Usage :
        biais = BiaisContinuation('raise', alpha=0.05)
        mccfr = MCCFRHoldEm(biais=biais)
        mccfr.entrainer(...)
    """

    CATEGORIES_VALIDES = (None, 'fold', 'call', 'raise')

    def __init__(self, categorie: str = None, alpha: float = 0.05):
        assert categorie in self.CATEGORIES_VALIDES, \
            f"Catégorie inconnue : {categorie} (attendu : {self.CATEGORIES_VALIDES})"
        self.categorie = categorie
        self.alpha     = float(alpha) if categorie else 0.0

    @property
    def actif(self) -> bool:
        """True si le biais s'applique (catégorie définie et alpha > 0)."""
        return self.categorie is not None and self.alpha > 0.0

    def bonus(self, action, pot: float) -> float:
        """Retourne α·pot si l'action correspond à la catégorie, 0 sinon."""
        if not self.actif:
            return 0.0
        t = action.type
        if self.categorie == 'fold'  and t == _TypeActionH.FOLD:
            return self.alpha * pot
        if self.categorie == 'call'  and t in (_TypeActionH.CHECK, _TypeActionH.CALL):
            return self.alpha * pot
        if self.categorie == 'raise' and t in (_TypeActionH.RAISE, _TypeActionH.ALL_IN):
            return self.alpha * pot
        return 0.0

    def __repr__(self):
        if not self.actif:
            return "BiaisContinuation(baseline)"
        return f"BiaisContinuation(categorie={self.categorie!r}, alpha={self.alpha})"


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class MCCFRHoldEm:
    """
    External Sampling MCCFR pour Texas Hold'em No Limit 3 joueurs.

    Construit la stratégie blueprint d'AXIOM par auto-jeu :
    les 3 joueurs jouent contre eux-mêmes en accumulant des regrets,
    jusqu'à converger vers un Équilibre de Nash approximatif.

    Usage typique :
        mccfr = MCCFRHoldEm()
        mccfr.entrainer(nb_iterations=100_000, verbose=True,
                        save_every=10_000,
                        chemin='data/strategy/blueprint_v1.pkl')
        mccfr.afficher_stats()

    Reprendre un entraînement interrompu :
        from ai.strategy import charger_blueprint
        mccfr.noeuds = charger_blueprint('data/strategy/blueprint_v1.pkl')
        mccfr.entrainer(nb_iterations=100_000, ...)
    """

    def __init__(self, biais: 'BiaisContinuation' = None):
        """
        biais : BiaisContinuation optionnel (Pluribus k=4). None = blueprint
                baseline (pas de biais). Voir BiaisContinuation pour les
                4 variantes : fold / call / raise.
        """
        self.noeuds: dict    = {}    # cle_infoset (str) → NoeudCFR
        self.iterations: int = 0     # nombre d'itérations complètes effectuées

        # Point 1 — Linear CFR : numéro global d'itération, mis à jour dans
        # entrainer() avant chaque appel à _es_mccfr(). Utilisé pour pondérer
        # strategie_somme par t (les itérations récentes pèsent davantage).
        self._iteration_courante: int = 0

        # Point 2 — Continuation Strategies (Pluribus) : biais optionnel
        # appliqué aux regrets pendant la traversée CFR. Baseline si None.
        self._biais: BiaisContinuation = biais if biais is not None else BiaisContinuation()

        # Instances d'abstraction partagées (créées une seule fois)
        # nb_simulations=100 : compromis vitesse/précision pour l'entraînement
        self._abs_cartes  = _AbsCartesV2()
        self._abs_actions = _AbsAction()

    # ------------------------------------------------------------------
    # ENTRAÎNEMENT PRINCIPAL
    # ------------------------------------------------------------------

    def entrainer(self, nb_iterations: int, stacks: int = 1500,
                  pb: int = 10, gb: int = 20, verbose: bool = False,
                  save_every: int = 0, chemin: str = None) -> None:
        """
        Lance nb_iterations d'External Sampling MCCFR.

        Chaque itération comprend :
          1. Un deal aléatoire (cartes + board complet précalculé)
          2. 3 traversées de l'arbre (une par joueur traversant)

        La stratégie MOYENNE (strategie_somme / nb_visites) converge
        vers l'Équilibre de Nash après suffisamment d'itérations.

        nb_iterations : nombre d'itérations à effectuer
        stacks        : stack de départ de chaque joueur (ex: 1500)
        pb, gb        : petite et grande blinde (ex: 10, 20)
        verbose       : affiche la progression toutes les 10%
        save_every    : sauvegarde auto toutes les N itérations (0 = désactivé)
        chemin        : chemin du fichier .pkl (requis si save_every > 0)
        """
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  AXIOM — MCCFR Hold'em | {nb_iterations:,} itérations")
            print(f"  Stacks={stacks} | Blindes {pb}/{gb}")
            print(f"  Infosets existants : {len(self.noeuds):,}")
            print(f"{'─'*60}")

        for it in range(1, nb_iterations + 1):
            # Point 1 — Linear CFR : numéro global de l'itération en cours
            # (self.iterations compte les itérations déjà faites avant ce run)
            self._iteration_courante = self.iterations + it

            # Un deal aléatoire par itération (cartes + board)
            etat_base = self._dealer_aleatoire(stacks, pb, gb)

            # 3 traversées : une par joueur traversant
            for joueur_traversant in range(3):
                etat = self._copier_etat(etat_base)
                self._es_mccfr(etat, joueur_traversant)

            self.iterations += 1

            # Progression
            if verbose:
                pas = max(1, nb_iterations // 10)
                if it % pas == 0:
                    print(f"  It. {it:8,}/{nb_iterations:,} | "
                          f"Infosets : {len(self.noeuds):,}")

            # Sauvegarde automatique
            if save_every > 0 and chemin and it % save_every == 0:
                from ai.strategy import sauvegarder_blueprint
                sauvegarder_blueprint(self.noeuds, chemin)
                if verbose:
                    print(f"  💾 Sauvegarde → {chemin}")

        if verbose:
            print(f"{'─'*60}")
            print(f"  ✅ Entraînement terminé | {len(self.noeuds):,} infosets créés")
            print(f"{'─'*60}\n")

    # ------------------------------------------------------------------
    # TRAVERSÉE RÉCURSIVE ES-MCCFR
    # ------------------------------------------------------------------

    def _es_mccfr(self, etat: dict, joueur_traversant: int) -> float:
        """
        Traversée récursive External Sampling MCCFR.

        Retourne l'utilité (gain net en jetons) pour le joueur_traversant.

        Trois types de nœuds gérés :
          1. Terminal (≤1 joueur actif)         → _gain_fold()
          2. Fin de street (file vide)           → _passer_street() ou showdown
          3. Nœud de décision                   → explorer (traversant) ou
                                                   échantillonner (adversaire)
        """
        # ── Compter les joueurs encore dans la main ──────────────────────
        actifs_allin = [i for i in range(3)
                        if etat['statuts'][i] in (_H_ACTIF, _H_ALLIN)]
        actifs_pouvant_agir = [i for i in actifs_allin
                               if etat['statuts'][i] == _H_ACTIF]

        # ── Nœud terminal : un seul joueur non-fold ─────────────────────
        if len(actifs_allin) <= 1:
            return self._gain_fold(etat, joueur_traversant)

        # ── File vide : fin du tour de parole ────────────────────────────
        if not etat['joueurs_en_attente']:
            if etat['phase'] == _H_RIVER:
                return self._gain_showdown(etat, joueur_traversant)

            # Tous en all-in → distribuer les cartes restantes et showdown
            if not actifs_pouvant_agir:
                etat = self._copier_etat(etat)
                while etat['phase'] < _H_RIVER:
                    self._passer_street(etat)
                return self._gain_showdown(etat, joueur_traversant)

            # Street suivante normale
            etat = self._copier_etat(etat)
            self._passer_street(etat)
            return self._es_mccfr(etat, joueur_traversant)

        # ── Joueur courant dans la file ──────────────────────────────────
        joueur_idx = etat['joueurs_en_attente'][0]

        # Passer les joueurs fold / all-in présents dans la file
        if etat['statuts'][joueur_idx] != _H_ACTIF:
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

        # ── Joueur traversant : explorer TOUTES les actions ──────────────
        if joueur_idx == joueur_traversant:
            # Point 1 — Linear CFR : pondération par le numéro d'itération t.
            # strategie_courante(t) met à jour strategie_somme += t × strategie,
            # donnant plus de poids aux stratégies récentes (meilleure convergence).
            strat_full = noeud.strategie_courante(float(self._iteration_courante))
            # Tronquer à len(actions) et renormaliser (stacks variables)
            strategie = strat_full[:len(actions)]
            somme_s   = sum(strategie)
            if somme_s > 0:
                strategie = [s / somme_s for s in strategie]
            else:
                strategie = [1.0 / len(actions)] * len(actions)

            # Explorer chaque action possible
            valeurs = []
            for action in actions:
                etat_copie = self._copier_etat(etat)
                # Retirer le joueur courant de la file
                etat_copie['joueurs_en_attente'] = list(
                    etat['joueurs_en_attente'][1:])
                self._appliquer_action(etat_copie, joueur_idx, action)
                v = self._es_mccfr(etat_copie, joueur_traversant)
                valeurs.append(v)

            # Point 2 — Continuation Strategies : biais LOCAL sur les regrets.
            # On ajoute α·pot à l'utilité des actions de la catégorie favorisée
            # (fold / call / raise), sans propager le biais vers l'appelant.
            if self._biais.actif:
                pot_biais = float(etat['pot'])
                valeurs_regret = [valeurs[i] + self._biais.bonus(actions[i], pot_biais)
                                   for i in range(len(actions))]
                valeur_noeud_regret = sum(strategie[i] * valeurs_regret[i]
                                           for i in range(len(actions)))
                for i in range(len(actions)):
                    noeud.regrets_cumules[i] = max(
                        0.0,
                        noeud.regrets_cumules[i]
                        + valeurs_regret[i] - valeur_noeud_regret
                    )
                # Valeur retournée : NON biaisée (empêche la propagation du
                # biais dans l'arbre et la corruption des regrets ancestraux).
                return sum(strategie[i] * valeurs[i] for i in range(len(actions)))

            # Baseline (pas de biais) : comportement CFR+ standard.
            valeur_noeud = sum(strategie[i] * valeurs[i]
                               for i in range(len(actions)))

            # CFR+ : plancher à 0 (convergence ~2× plus rapide)
            for i in range(len(actions)):
                noeud.regrets_cumules[i] = max(
                    0.0, noeud.regrets_cumules[i] + valeurs[i] - valeur_noeud
                )

            return valeur_noeud

        # ── Adversaire : échantillonner UNE action ───────────────────────
        else:
            strat_full = self._regret_matching(noeud)
            # Tronquer à len(actions) et renormaliser (stacks variables)
            strategie = strat_full[:len(actions)]
            somme_s   = sum(strategie)
            if somme_s > 0:
                strategie = [s / somme_s for s in strategie]
            else:
                strategie = [1.0 / len(actions)] * len(actions)

            # Point 1 — Linear CFR : pondérer la stratégie somme par t.
            # Les itérations récentes contribuent proportionnellement davantage.
            t = float(self._iteration_courante)
            for i in range(len(actions)):
                noeud.strategie_somme[i] += t * strategie[i]
            noeud.nb_visites += 1

            # Point 7 — Élagage epsilon : dans _PROBA_ELAGAGE% des traversées,
            # ignorer les actions avec probabilité < _EPSILON_ELAGAGE pour
            # accélérer l'entraînement sans biais notable sur la stratégie finale.
            if _rand_h.random() < _PROBA_ELAGAGE:
                actions_retenues  = [a for a, p in zip(actions, strategie)
                                     if p >= _EPSILON_ELAGAGE]
                strategie_retenue = [p for p in strategie
                                     if p >= _EPSILON_ELAGAGE]
                if actions_retenues:
                    somme_r = sum(strategie_retenue)
                    strategie_retenue = [p / somme_r for p in strategie_retenue]
                    idx_action = self._echantillonner(strategie_retenue)
                    action = actions_retenues[idx_action]
                else:
                    # Aucune action ne passe le seuil → échantillonnage normal
                    idx_action = self._echantillonner(strategie)
                    action = actions[idx_action]
            else:
                # Exploration complète (5% des traversées)
                idx_action = self._echantillonner(strategie)
                action = actions[idx_action]

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
        Crée un état de départ avec un deal aléatoire.

        Distribue 2 cartes à chaque joueur + 5 cartes pour le board complet.
        Poste les blindes automatiquement (SB=1 → pb, BB=2 → gb).
        Précalcule les buckets pour toutes les phases × tous les joueurs.

        Positions de table :
          BTN=0 (dealer button), SB=1, BB=2
        """
        # Mélange et distribution
        deck = list(_TreysDeckH().cards)
        _rand_h.shuffle(deck)

        cartes = [deck[0:2], deck[2:4], deck[4:6]]   # [BTN, SB, BB]
        board_complet = deck[6:11]                    # 5 cartes (précalculées)

        # Stacks et blindes
        stacks_liste  = [stacks, stacks, stacks]
        contributions = [0, 0, 0]
        mises_tour    = [0, 0, 0]

        # SB (joueur 1) poste la petite blinde
        mise_sb = min(pb, stacks_liste[1])
        stacks_liste[1]  -= mise_sb
        contributions[1] += mise_sb
        mises_tour[1]     = mise_sb

        # BB (joueur 2) poste la grande blinde
        mise_bb = min(gb, stacks_liste[2])
        stacks_liste[2]  -= mise_bb
        contributions[2] += mise_bb
        mises_tour[2]     = mise_bb

        # Statuts (ALL_IN si stack = 0 après blindes)
        statuts = [
            _H_ALLIN if stacks_liste[i] == 0 else _H_ACTIF
            for i in range(3)
        ]

        # Précalcul des buckets ET équités (coûteux → fait UNE SEULE FOIS par deal,
        # puis partagé en read-only entre toutes les copies d'état)
        buckets, equites = self._precomputer_buckets_et_equites(cartes, board_complet)

        return {
            'cartes'             : cartes,
            'board_complet'      : board_complet,   # partagé, read-only
            'board_visible'      : [],
            'stacks'             : stacks_liste,
            'contributions'      : contributions,
            'mises_tour'         : mises_tour,
            'mise_courante'      : gb,
            'pot'                : mise_sb + mise_bb,
            'statuts'            : statuts,
            'phase'              : _H_PREFLOP,
            'joueurs_en_attente' : list(_ORDRE_PREFLOP),  # [0, 1, 2]
            'hist_phases'        : ['', '', '', ''],
            'grande_blinde'      : gb,
            'buckets'            : buckets,         # partagé, read-only
            'equites'            : equites,         # partagé, read-only
            # raise_fracs[0] initialisé avec la fraction BB/pot pour que
            # encoder_infoset voie correctement le blind à suivre dès la
            # première décision du BTN (sinon le réseau croit qu'il n'y a
            # pas de mise alors que mise_courante = gb > 0).
            'raise_fracs'        : [gb / max(mise_sb + mise_bb, 1),
                                    0.0, 0.0, 0.0],
        }

    def _precomputer_buckets_et_equites(self, cartes: list,
                                        board_complet: list) -> tuple:
        """
        Précalcule buckets ET équités brutes pour chaque joueur × chaque phase.

        Retourne (buckets, equites) où :
          buckets[joueur][phase] → int (0 à NB_BUCKETS-1)
          equites[joueur][phase] → float (0.0 à 1.0)

        Le calcul Monte Carlo est fait UNE SEULE FOIS par deal via
        bucket_et_equite() (évite la double simulation MC).
        """
        boards_par_phase = [
            [],                   # PREFLOP
            board_complet[:3],    # FLOP
            board_complet[:4],    # TURN
            board_complet[:5],    # RIVER
        ]
        buckets = []
        equites = []
        for j in range(3):
            bkts_j = []
            eqs_j  = []
            for board in boards_par_phase:
                b, e = self._abs_cartes.bucket_et_equite(cartes[j], board)
                bkts_j.append(b)
                eqs_j.append(e)
            buckets.append(bkts_j)
            equites.append(eqs_j)
        return buckets, equites

    # ------------------------------------------------------------------
    # CLÉ D'INFOSET
    # ------------------------------------------------------------------

    def _cle_infoset(self, etat: dict, joueur_idx: int) -> str:
        """
        Construit la clé d'infoset pour un joueur dans l'état courant.

        Format compatible avec abstraction/info_set.py :
          "PREFLOP|pos=0|bucket=7|pot=2|stacks=(100,100,95)|hist=xcr"

        Toutes les valeurs continues sont normalisées en multiples de la GB
        et arrondies à des paliers discrets pour réduire l'espace des infosets.
        """
        phase = etat['phase']
        gb    = max(etat['grande_blinde'], 1)

        # Pot et stacks normalisés par la grande blinde
        # P7 : stacks bucketisés Spin & Rush (7 niveaux), hist abstrait S/M/L + cap 6
        pot_norm   = _normaliser(etat['pot'] / gb, PALIERS_POT)
        stacks_str = ','.join(
            str(_normaliser(etat['stacks'][i] / gb, PALIERS_STACK_SPIN_RUSH))
            for i in range(3)
        )

        raise_bucket = _discretiser_raise_frac(
            etat['mise_courante'] / max(etat['pot'], 1)
        )

        return (
            f"{_H_NOM_PHASE[phase]}"
            f"|pos={joueur_idx}"
            f"|bucket={etat['buckets'][joueur_idx][phase]}"
            f"|pot={pot_norm}"
            f"|stacks=({stacks_str})"
            f"|hist={_format_hist_avec_cap(etat['hist_phases'][phase])}"
            f"|raise={raise_bucket}"
        )

    # ------------------------------------------------------------------
    # ACTIONS ABSTRAITES
    # ------------------------------------------------------------------

    def _actions_abstraites(self, etat: dict, joueur_idx: int) -> list:
        """
        Retourne les actions légales abstraites pour un joueur dans l'état courant.

        Actions possibles :
          FOLD   : si une mise est à suivre (a_payer > 0)
          CHECK  : si rien à suivre (a_payer == 0)
          CALL   : si mise à suivre et stack suffisant
          RAISE  : aux fractions du pot définies dans TAILLES_MISE
          ALL_IN : toujours si stack > 0

        Les montants de raise sont dédoublonnés.
        """
        stack         = etat['stacks'][joueur_idx]
        mise_tour     = etat['mises_tour'][joueur_idx]
        mise_courante = etat['mise_courante']
        pot           = etat['pot']
        gb            = etat['grande_blinde']
        a_payer       = mise_courante - mise_tour

        actions = []

        if a_payer > 0:
            actions.append(_ActionH(_TypeActionH.FOLD))

        if a_payer == 0:
            actions.append(_ActionH(_TypeActionH.CHECK))

        if 0 < a_payer < stack:
            actions.append(_ActionH(_TypeActionH.CALL, montant=mise_courante))

        tailles = _TAILLES_PREFLOP_H if etat['phase'] == _H_PREFLOP else _TAILLES_POSTFLOP_H
        for fraction in tailles:
            montant_raise = mise_courante + int(pot * fraction)
            montant_raise = max(montant_raise, mise_courante + gb)
            if montant_raise < mise_tour + stack:
                actions.append(_ActionH(_TypeActionH.RAISE,
                                        montant=montant_raise))

        if _ALL_IN_H and stack > 0:
            montant_allin = mise_tour + stack
            if not any(a.type == _TypeActionH.RAISE and a.montant == montant_allin
                       for a in actions):
                actions.append(_ActionH(_TypeActionH.ALL_IN,
                                        montant=montant_allin))

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
                          action: _ActionH) -> None:
        """
        Applique une action sur l'état (en place — l'état a déjà été copié).

        Pour RAISE et ALL_IN : si le raise dépasse la mise courante,
        reconstruit la file d'attente (tous les autres actifs doivent répondre).
        """
        phase     = etat['phase']
        stack     = etat['stacks'][joueur_idx]
        mise_tour = etat['mises_tour'][joueur_idx]

        if action.type == _TypeActionH.FOLD:
            etat['statuts'][joueur_idx] = _H_FOLD
            etat['hist_phases'][phase] += 'f'

        elif action.type == _TypeActionH.CHECK:
            etat['hist_phases'][phase] += 'x'

        elif action.type == _TypeActionH.CALL:
            a_payer      = etat['mise_courante'] - mise_tour
            a_payer_reel = min(a_payer, stack)
            etat['stacks'][joueur_idx]        -= a_payer_reel
            etat['mises_tour'][joueur_idx]    += a_payer_reel
            etat['contributions'][joueur_idx] += a_payer_reel
            etat['pot']                       += a_payer_reel
            if etat['stacks'][joueur_idx] == 0:
                etat['statuts'][joueur_idx] = _H_ALLIN
            etat['hist_phases'][phase] += 'c'

        elif action.type in (_TypeActionH.RAISE, _TypeActionH.ALL_IN):
            # Point 3 : calculer le bucket de sizing AVANT de modifier le pot
            # frac = montant_total_raise / pot_courant (avant la mise)
            frac_raise   = action.montant / max(etat['pot'], 1)
            raise_bucket = _discretiser_raise_frac(frac_raise)

            a_payer      = action.montant - mise_tour
            a_payer_reel = min(a_payer, stack)
            etat['stacks'][joueur_idx]        -= a_payer_reel
            etat['mises_tour'][joueur_idx]    += a_payer_reel
            etat['contributions'][joueur_idx] += a_payer_reel
            etat['pot']                       += a_payer_reel

            nouvelle_mise = etat['mises_tour'][joueur_idx]
            if nouvelle_mise > etat['mise_courante']:
                etat['mise_courante'] = nouvelle_mise
                # Mettre à jour raise_fracs pour la phase courante
                etat['raise_fracs'][phase] = etat['mise_courante'] / max(etat['pot'], 1)
                # Réinsérer les adversaires actifs dans la file
                self._reinserer_apres_raise(etat, joueur_idx)

            if etat['stacks'][joueur_idx] == 0:
                etat['statuts'][joueur_idx] = _H_ALLIN

            # 'r{bucket}' pour un raise (2 chars), 'a' pour all-in (1 char)
            if action.type == _TypeActionH.RAISE:
                code = f'r{raise_bucket}'
            else:
                code = 'a'
            etat['hist_phases'][phase] += code

    def _reinserer_apres_raise(self, etat: dict, raiser_idx: int) -> None:
        """
        Après un raise, tous les joueurs ACTIF (sauf le raiser) doivent
        répondre à la nouvelle mise. On reconstruit la file d'attente
        dans l'ordre circulaire, en partant du joueur après le raiser.

        Exemple postflop, ordre [1,2,0], raise par joueur 2 :
          ordre depuis après raiser_2 = [0, 1]
          → joueurs 0 et 1 doivent répondre (si actifs)
        """
        ordre = (_ORDRE_PREFLOP if etat['phase'] == _H_PREFLOP
                 else _ORDRE_POSTFLOP)

        pos = ordre.index(raiser_idx) if raiser_idx in ordre else 0
        # Ordre circulaire à partir du joueur APRÈS le raiser
        depuis_apres = ordre[pos + 1:] + ordre[:pos + 1]

        etat['joueurs_en_attente'] = [
            j for j in depuis_apres
            if j != raiser_idx and etat['statuts'][j] == _H_ACTIF
        ]

    # ------------------------------------------------------------------
    # TRANSITION DE STREET
    # ------------------------------------------------------------------

    def _passer_street(self, etat: dict) -> None:
        """
        Passe à la street suivante (modification en place).

        - Remet les mises du tour à 0
        - Avance la phase (PREFLOP→FLOP→TURN→RIVER)
        - Met à jour le board visible
        - Reconstruit la file d'attente en ordre postflop (SB→BB→BTN)
        """
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

        # File postflop : SB(1) → BB(2) → BTN(0), joueurs ACTIF seulement
        etat['joueurs_en_attente'] = [
            j for j in _ORDRE_POSTFLOP
            if etat['statuts'][j] == _H_ACTIF
        ]

    # ------------------------------------------------------------------
    # GAINS AUX NŒUDS TERMINAUX
    # ------------------------------------------------------------------

    def _gain_fold(self, etat: dict, joueur_traversant: int) -> float:
        """
        Gain net pour le joueur_traversant quand la main se termine par fold
        (un seul joueur encore actif/allin, il remporte le pot entier).

        Gain = pot_reçu - contribution_versée
        """
        actifs = [i for i in range(3)
                  if etat['statuts'][i] in (_H_ACTIF, _H_ALLIN)]
        if not actifs:
            return -etat['contributions'][joueur_traversant]

        gagnant = actifs[0]
        if joueur_traversant == gagnant:
            return etat['pot'] - etat['contributions'][joueur_traversant]
        else:
            return -etat['contributions'][joueur_traversant]

    def _gain_showdown(self, etat: dict, joueur_traversant: int) -> float:
        """
        Gain net pour le joueur_traversant au showdown.

        Évalue toutes les mains actives avec treys (score le plus bas = gagnant).
        En cas d'égalité, le pot est partagé équitablement.

        Gain = (pot / nb_gagnants) - contribution  si gagnant
        Gain = - contribution                       si perdant
        """
        board = (etat['board_visible']
                 if len(etat['board_visible']) == 5
                 else etat['board_complet'])

        actifs = [i for i in range(3)
                  if etat['statuts'][i] in (_H_ACTIF, _H_ALLIN)]

        if len(actifs) == 0:
            return 0.0
        if len(actifs) == 1:
            return self._gain_fold(etat, joueur_traversant)

        # Évaluer chaque main (score bas = main forte dans treys)
        meilleur_score = None
        gagnants       = []
        for i in actifs:
            try:
                score = _eval_holdem.evaluate(board, etat['cartes'][i])
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

    def _obtenir_noeud(self, cle: str, nb_actions: int) -> 'NoeudCFR':
        """Récupère ou crée un NoeudCFR. Étend le nœud si besoin (cas rare).

        Quand nb_actions augmente (stack plus profond → raises supplémentaires
        disponibles), on étend les tableaux ET on remet strategie_somme à zéro.
        Les regrets accumulés sont conservés (signal utile), mais la stratégie
        MOYENNE était calculée sur un espace d'actions incomplet — la garder
        biaise la strategie_moyenne() vers les anciennes actions au détriment
        des nouvelles dont la somme repart de 0.
        """
        if cle not in self.noeuds:
            self.noeuds[cle] = NoeudCFR(nb_actions)
        noeud = self.noeuds[cle]
        if noeud.nb_actions < nb_actions:
            diff = nb_actions - noeud.nb_actions
            noeud.regrets_cumules.extend([0.0] * diff)
            noeud.nb_actions = nb_actions
            # Réinitialiser toute la strategie_somme : elle était calculée
            # sur l'ancien espace (actions manquantes → biais irréparable).
            noeud.strategie_somme = [0.0] * nb_actions
        return noeud

    def _copier_etat(self, etat: dict) -> dict:
        """
        Copie légère de l'état pour la traversée de l'arbre de jeu.

        Listes mutables : copiées (shallow copy) — chaque branche est indépendante.
        'board_complet' et 'buckets' : partagés en lecture seule → pas de recopie.
        """
        return {
            'cartes'             : etat['cartes'],          # read-only
            'board_complet'      : etat['board_complet'],   # read-only
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
            'buckets'            : etat['buckets'],         # read-only
            'equites'            : etat['equites'],         # read-only
            'raise_fracs'        : list(etat['raise_fracs']),
        }

    def _regret_matching(self, noeud: 'NoeudCFR') -> list:
        """
        Calcule la stratégie par regret matching SANS effets de bord.
        (Ne modifie pas strategie_somme ni nb_visites.)
        Utilisé pour les nœuds adversaires.
        """
        regrets_pos = [max(r, 0.0) for r in noeud.regrets_cumules]
        somme_pos   = sum(regrets_pos)
        if somme_pos > 0:
            return [r / somme_pos for r in regrets_pos]
        return [1.0 / noeud.nb_actions] * noeud.nb_actions

    def _echantillonner(self, strategie: list) -> int:
        """Roulette wheel sampling — retourne un index selon la distribution."""
        r = _rand_h.random()
        cumul = 0.0
        for i, p in enumerate(strategie):
            cumul += p
            if r <= cumul:
                return i
        return len(strategie) - 1   # sécurité

    # ------------------------------------------------------------------
    # STATISTIQUES ET AFFICHAGE
    # ------------------------------------------------------------------

    def afficher_stats(self) -> None:
        """Affiche un résumé du blueprint : répartition par phase, visites moyennes."""
        if not self.noeuds:
            print("  Aucun infoset — lancez d'abord entrainer().")
            return

        comptage        = {nom: 0 for nom in _H_NOM_PHASE}
        visites_totales = 0

        for cle, noeud in self.noeuds.items():
            for nom in _H_NOM_PHASE:
                if cle.startswith(nom + '|'):
                    comptage[nom] += 1
                    break
            visites_totales += noeud.nb_visites

        print(f"\n{'─'*50}")
        print(f"  AXIOM Blueprint | {self.iterations:,} itérations")
        print(f"  Infosets total  : {len(self.noeuds):,}")
        for nom in _H_NOM_PHASE:
            print(f"    {nom:8} : {comptage[nom]:,}")
        moy = visites_totales / len(self.noeuds) if self.noeuds else 0
        print(f"  Visites moy/infoset : {moy:.1f}")
        print(f"{'─'*50}\n")

    def reinitialiser(self) -> None:
        """Remet tous les nœuds et le compteur d'itérations à zéro."""
        self.noeuds.clear()
        self.iterations = 0

    def __repr__(self) -> str:
        return (f"MCCFRHoldEm(iterations={self.iterations:,}, "
                f"infosets={len(self.noeuds):,})")