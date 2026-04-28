# =============================================================================
# AXIOM — training/evaluator.py
# Évaluation des performances d'AXIOM (Phase 5 + mise à jour Phase 10).
#
# ─────────────────────────────────────────────────────────────────────────────
# NOUVEAUTÉ Phase 10 — Statistiques d'utilisation des sources
# ─────────────────────────────────────────────────────────────────────────────
#
# Pour chaque scénario et chaque position (BTN/SB/BB), l'évaluateur collecte
# les statistiques de décision d'AXIOM :
#   - % blueprint 3J utilisé
#   - % Deep CFR utilisé
#   - % heuristique utilisé
#
# Cela permet d'identifier précisément où le blueprint manque de couverture
# et où le Deep CFR prend le relai — information clé pour orienter l'entraînement.
# =============================================================================

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from training.self_play import (
    MoteurSelfPlay, simuler,
    AgentAleatoire, AgentCallOnly, AgentRaiseOnly,
    AgentTAG, AgentLAG, AgentRegulier,
    ResultatPartie,
)
from config.settings import STACK_DEPART, NB_JOUEURS


# =============================================================================
# STRUCTURE DE STATISTIQUES DE SOURCE
# =============================================================================

@dataclass
class StatsSource:
    """Statistiques d'utilisation des sources pour une position donnée."""
    blueprint_hu   : float = 0.0
    blueprint_3j   : float = 0.0
    deep_cfr       : float = 0.0
    heuristique    : float = 0.0
    total_decisions: int   = 0


# =============================================================================
# STRUCTURE DE RÉSULTATS
# =============================================================================

@dataclass
class RapportEvaluation:
    """
    Rapport complet d'évaluation d'un agent AXIOM.
    Phase 10 : winrates + stats d'utilisation des sources par scénario/position.
    """
    # ── Baselines ──────────────────────────────────────────────────────────
    winrate_vs_aleatoire  : float = 0.0
    winrate_vs_call       : float = 0.0
    winrate_vs_raise      : float = 0.0

    # ── Semi-pros ─────────────────────────────────────────────────────────
    winrate_vs_tag        : float = 0.0
    winrate_vs_lag        : float = 0.0
    winrate_vs_regulier   : float = 0.0

    # ── Agrégats ──────────────────────────────────────────────────────────
    winrate_moyen         : float = 0.0
    winrate_moyen_semipro : float = 0.0

    # ── Sources ───────────────────────────────────────────────────────────
    sources_agent         : str  = "inconnu"
    # {label_scenario: {0: StatsSource(BTN), 1: StatsSource(SB), 2: StatsSource(BB), 'total': StatsSource}}
    stats_sources         : Dict = field(default_factory=dict)

    # ── Divers ────────────────────────────────────────────────────────────
    exploitabilite_approx : float = float('nan')
    nb_mains_par_test     : int   = 0
    duree_totale_s        : float = 0.0
    timestamp             : str   = ""

    def afficher(self) -> None:
        print(f"\n{'═'*62}")
        print(f"  AXIOM — Rapport d'évaluation Phase 10")
        print(f"{'═'*62}")
        print(f"  Agent            : {self.sources_agent}")
        print(f"  Mains/test       : {self.nb_mains_par_test:,}")
        print(f"  Durée totale     : {self.duree_totale_s:.1f}s")
        print()
        print(f"  {'Métrique':32} | {'Valeur':>10}")
        print(f"  {'─'*32}-+-{'─'*10}")
        print(f"  {'[BASELINE] vs Aléatoire':32} | {self.winrate_vs_aleatoire:>+10.1f} bb/100")
        print(f"  {'[BASELINE] vs Call-Only':32} | {self.winrate_vs_call:>+10.1f} bb/100")
        print(f"  {'[BASELINE] vs Raise-Only':32} | {self.winrate_vs_raise:>+10.1f} bb/100")
        print(f"  {'─'*32}-+-{'─'*10}")
        print(f"  {'[SEMI-PRO] vs TAG':32} | {self.winrate_vs_tag:>+10.1f} bb/100")
        print(f"  {'[SEMI-PRO] vs LAG':32} | {self.winrate_vs_lag:>+10.1f} bb/100")
        print(f"  {'[SEMI-PRO] vs Régulier':32} | {self.winrate_vs_regulier:>+10.1f} bb/100")
        print(f"  {'─'*32}-+-{'─'*10}")
        print(f"  {'Winrate moyen (6 scénarios)':32} | {self.winrate_moyen:>+10.1f} bb/100")
        print(f"  {'Winrate moyen SEMI-PRO':32} | {self.winrate_moyen_semipro:>+10.1f} bb/100")
        if not np.isnan(self.exploitabilite_approx):
            print(f"  {'Exploitabilité approx.':32} | {self.exploitabilite_approx:>10.4f}")
        print(f"{'═'*62}\n")
        self._afficher_verdict()
        if self.stats_sources:
            self.afficher_stats_sources()

    def _afficher_verdict(self) -> None:
        wr = self.winrate_moyen_semipro
        print(f"  ─── Verdict semi-pro : {wr:>+.1f} bb/100")
        if wr > 20:
            print("  🟢 AXIOM domine les semi-pros — prêt pour la confrontation réelle")
        elif wr > 0:
            print("  🟡 AXIOM est légèrement positif — des ajustements peuvent aider")
        else:
            print("  🔴 AXIOM est en difficulté face aux semi-pros — entraînement recommandé")
        print()

    def afficher_stats_sources(self) -> None:
        """
        Tableau d'utilisation des sources par scénario adversaire × position.
        Identifie précisément où le blueprint est insuffisant (Deep CFR élevé).
        """
        print(f"\n{'═'*80}")
        print(f"  AXIOM — Utilisation des sources de stratégie")
        print(f"  par scénario adversaire × position (BTN / SB / BB)")
        print(f"{'═'*80}")
        print(f"  {'Scénario':20} │ {'Pos':5} │ "
              f"{'BP-HU':>7}  {'BP-3J':>7}  {'DeepCFR':>8}  {'Heurist':>8}  "
              f"{'Décisions':>10}  {'Note':>4}")
        print(f"  {'─'*20}─┼─{'─'*5}─┼─"
              f"{'─'*7}──{'─'*7}──{'─'*8}──{'─'*8}──{'─'*10}──{'─'*6}")

        noms_pos = {0: 'BTN', 1: 'SB', 2: 'BB', 'total': 'TOTAL'}

        for scenario, stats_par_pos in self.stats_sources.items():
            premier = True
            for pos_key in [0, 1, 2, 'total']:
                if pos_key not in stats_par_pos:
                    continue
                s = stats_par_pos[pos_key]
                label = scenario if premier else ""
                premier = False

                # Indicateur visuel selon couverture blueprint
                if pos_key == 'total':
                    note = "─────"
                elif s.deep_cfr >= 60:
                    note = "⚠️ CFR"
                elif s.deep_cfr >= 35:
                    note = "🟡 mix"
                else:
                    note = "✅ BP "

                print(f"  {label:20} │ {noms_pos[pos_key]:5} │ "
                      f"{s.blueprint_hu:>6.1f}%  "
                      f"{s.blueprint_3j:>6.1f}%  "
                      f"{s.deep_cfr:>7.1f}%  "
                      f"{s.heuristique:>7.1f}%  "
                      f"{s.total_decisions:>10,}  "
                      f"{note}")

            print(f"  {'─'*20}─┼─{'─'*5}─┼─"
                  f"{'─'*7}──{'─'*7}──{'─'*8}──{'─'*8}──{'─'*10}──{'─'*6}")

        print()
        print("  BP-HU = Blueprint Heads-Up  │  BP-3J = Blueprint 3 joueurs")
        print("  ⚠️ CFR  = Deep CFR > 60% (le blueprint ne couvre pas assez cette situation)")
        print("  🟡 mix  = Deep CFR 35-60% (couverture partielle)")
        print("  ✅ BP   = Deep CFR < 35% (blueprint bien couvert)")
        print(f"{'═'*80}\n")

    def score_global(self) -> float:
        score = (
            self.winrate_vs_aleatoire
            + self.winrate_vs_call
            + self.winrate_vs_raise
            + self.winrate_vs_tag      * 2.0
            + self.winrate_vs_lag      * 2.0
            + self.winrate_vs_regulier * 2.0
        ) / 9.0
        if not np.isnan(self.exploitabilite_approx):
            score -= self.exploitabilite_approx * 10.0
        return score


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class Evaluateur:
    """
    Évalue les performances d'un agent AXIOM.
    Phase 10 : 6 scénarios, rotation des 3 positions, stats sources.
    """

    def __init__(self,
                 nb_mains          : int   = 500,
                 blindes           : tuple = (10, 20),
                 graine            : Optional[int] = 42,
                 verbose           : bool  = False,
                 inclure_baselines : bool  = True):

        self.nb_mains          = nb_mains
        self.blindes           = blindes
        self.graine            = graine
        self.verbose           = verbose
        self.inclure_baselines = inclure_baselines

    def evaluer(self, agent) -> RapportEvaluation:
        """
        Lance l'évaluation complète et retourne un RapportEvaluation.
        Ne l'affiche PAS — c'est l'appelant qui appelle rapport.afficher().
        """
        debut = time.time()
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.verbose:
            print(f"\n  🔬 Début évaluation ({self.nb_mains} mains/scénario)...")

        stats_sources_global = {}

        # ── Baselines ──────────────────────────────────────────────────────
        if self.inclure_baselines:
            wr_aleatoire, ss = self._winrate_scenario(
                agent,
                [AgentAleatoire(self.graine),
                 AgentAleatoire(self.graine + 1 if self.graine else None)],
                label="[BASE] vs Aléatoire",
            )
            stats_sources_global["vs Aléatoire"] = ss

            wr_call, ss = self._winrate_scenario(
                agent, [AgentCallOnly(), AgentCallOnly()],
                label="[BASE] vs Call-Only",
            )
            stats_sources_global["vs Call-Only"] = ss

            wr_raise, ss = self._winrate_scenario(
                agent, [AgentRaiseOnly(), AgentRaiseOnly()],
                label="[BASE] vs Raise-Only",
            )
            stats_sources_global["vs Raise-Only"] = ss
        else:
            wr_aleatoire = float('nan')
            wr_call      = float('nan')
            wr_raise     = float('nan')

        # ── Semi-pros ─────────────────────────────────────────────────────
        g = self.graine

        wr_tag, ss = self._winrate_scenario(
            agent,
            [AgentTAG(g), AgentTAG(g + 10 if g else None)],
            label="[SEMI-PRO] vs TAG",
        )
        stats_sources_global["vs TAG"] = ss

        wr_lag, ss = self._winrate_scenario(
            agent,
            [AgentLAG(g), AgentLAG(g + 20 if g else None)],
            label="[SEMI-PRO] vs LAG",
        )
        stats_sources_global["vs LAG"] = ss

        wr_regulier, ss = self._winrate_scenario(
            agent,
            [AgentRegulier(g), AgentRegulier(g + 30 if g else None)],
            label="[SEMI-PRO] vs Régulier",
        )
        stats_sources_global["vs Régulier"] = ss

        # ── Agrégats ──────────────────────────────────────────────────────
        valeurs_valides = [v for v in [wr_aleatoire, wr_call, wr_raise,
                                        wr_tag, wr_lag, wr_regulier]
                           if not (isinstance(v, float) and np.isnan(v))]
        winrate_moyen         = float(np.mean(valeurs_valides)) if valeurs_valides else 0.0
        winrate_moyen_semipro = (wr_tag + wr_lag + wr_regulier) / 3.0
        duree_totale          = time.time() - debut

        # ── Sources ───────────────────────────────────────────────────────
        sources = []
        if hasattr(agent, '_blueprint_hu') and agent._blueprint_hu is not None:
            sources.append(f"blueprint_hu({len(agent._blueprint_hu):,})")
        if hasattr(agent, '_blueprint') and agent._blueprint is not None:
            sources.append(f"blueprint({len(agent._blueprint):,})")
        if hasattr(agent, '_reseaux_strategie') and agent._reseaux_strategie is not None:
            sources.append("deep_cfr")
        if not sources:
            sources.append("heuristique")

        return RapportEvaluation(
            winrate_vs_aleatoire  = wr_aleatoire,
            winrate_vs_call       = wr_call,
            winrate_vs_raise      = wr_raise,
            winrate_vs_tag        = wr_tag,
            winrate_vs_lag        = wr_lag,
            winrate_vs_regulier   = wr_regulier,
            winrate_moyen         = winrate_moyen,
            winrate_moyen_semipro = winrate_moyen_semipro,
            sources_agent         = ", ".join(sources),
            stats_sources         = stats_sources_global,
            exploitabilite_approx = float('nan'),
            nb_mains_par_test     = self.nb_mains,
            duree_totale_s        = duree_totale,
            timestamp             = timestamp,
        )

    def evaluer_avec_exploitabilite(self, agent,
                                     nb_mains_exploit: int = 200) -> RapportEvaluation:
        rapport = self.evaluer(agent)
        wr_raise_adv, _ = self._winrate_scenario(
            AgentRaiseOnly(), [agent, agent],
            label="best_response",
            agent_est_pos0=False,
            nb_mains_override=nb_mains_exploit,
        )
        rapport.exploitabilite_approx = max(0.0, wr_raise_adv)
        return rapport

    # ==================================================================
    # WINRATE PAR SCÉNARIO avec stats sources
    # ==================================================================

    def _winrate_scenario(self,
                           agent,
                           adversaires      : list,
                           label            : str   = "",
                           agent_est_pos0   : bool  = True,
                           nb_mains_override: int   = None):
        """
        Mesure le winrate sur 3 positions et collecte les stats d'utilisation
        des sources (blueprint HU, blueprint 3J, Deep CFR, heuristique).

        Retourne : (winrate_moyen: float, stats_par_pos: dict)
        """
        nb = nb_mains_override or self.nb_mains
        gb = self.blindes[1]

        if self.verbose:
            print(f"    {label:26} : {nb} mains...", end=' ', flush=True)

        # ── Cas spécial best_response (pas de rotation) ────────────────────
        if not agent_est_pos0:
            agents_liste = adversaires + [agent]
            resultat = simuler(
                agents        = agents_liste,
                nb_mains      = nb,
                stacks_depart = STACK_DEPART,
                blindes_fixes = self.blindes,
                graine        = self.graine,
                verbose       = False,
            )
            wr = resultat.winrate_bb100(2, gb)
            if self.verbose:
                print(f"{wr:>+.1f} bb/100")
            return wr, {}

        # ── Rotation sur les 3 positions ───────────────────────────────────
        nb_par_pos = max(nb // 3, 1)
        winrates   = []
        stats_pos  = {}

        graines_pos = [
            self.graine,
            self.graine + 1000 if self.graine else None,
            self.graine + 2000 if self.graine else None,
        ]

        for pos_axiom in range(3):
            # Réinitialiser les stats de l'agent avant cette position
            if hasattr(agent, 'reinitialiser_stats'):
                agent.reinitialiser_stats()

            adv = list(adversaires)
            if pos_axiom == 0:
                agents_liste = [agent, adv[0], adv[1]]
            elif pos_axiom == 1:
                agents_liste = [adv[0], agent, adv[1]]
            else:
                agents_liste = [adv[0], adv[1], agent]

            res = simuler(
                agents        = agents_liste,
                nb_mains      = nb_par_pos,
                stacks_depart = STACK_DEPART,
                blindes_fixes = self.blindes,
                graine        = graines_pos[pos_axiom],
                verbose       = False,
            )
            winrates.append(res.winrate_bb100(pos_axiom, gb))

            # Collecter les stats de source pour cette position
            if hasattr(agent, 'stats'):
                s     = agent.stats
                total = max(s.get('total', 0), 1)
                stats_pos[pos_axiom] = StatsSource(
                    blueprint_hu    = 100.0 * s.get('blueprint_hu', 0) / total,
                    blueprint_3j    = 100.0 * s.get('blueprint', 0)    / total,
                    deep_cfr        = 100.0 * s.get('deep_cfr', 0)     / total,
                    heuristique     = 100.0 * s.get('heurist', 0)      / total,
                    total_decisions = s.get('total', 0),
                )

        wr = sum(winrates) / 3.0

        # ── Total pondéré sur les 3 positions ─────────────────────────────
        if stats_pos:
            tot = sum(s.total_decisions for s in stats_pos.values())
            if tot > 0:
                def _moyenne(champ):
                    return sum(getattr(s, champ) * s.total_decisions
                               for s in stats_pos.values()) / tot

                stats_pos['total'] = StatsSource(
                    blueprint_hu    = _moyenne('blueprint_hu'),
                    blueprint_3j    = _moyenne('blueprint_3j'),
                    deep_cfr        = _moyenne('deep_cfr'),
                    heuristique     = _moyenne('heuristique'),
                    total_decisions = tot,
                )

        if self.verbose:
            print(f"{wr:>+.1f} bb/100  "
                  f"(BTN={winrates[0]:+.1f} SB={winrates[1]:+.1f} BB={winrates[2]:+.1f})")

        # Réinitialiser les stats après le scénario complet
        if hasattr(agent, 'reinitialiser_stats'):
            agent.reinitialiser_stats()

        return wr, stats_pos

    # ==================================================================
    # COMPARAISON DE PLUSIEURS AGENTS
    # ==================================================================

    def comparer(self, agents_dict: dict) -> dict:
        rapports = {}
        for nom, agent in agents_dict.items():
            if self.verbose:
                print(f"\n  Agent : {nom}")
            rapport = self.evaluer(agent)
            rapports[nom] = rapport

        rapports_tries = dict(
            sorted(rapports.items(),
                   key=lambda x: x[1].winrate_moyen_semipro,
                   reverse=True)
        )
        if self.verbose:
            self._afficher_classement(rapports_tries)
        return rapports_tries

    def _afficher_classement(self, rapports: dict) -> None:
        print(f"\n{'═'*70}")
        print(f"  AXIOM — Classement des agents (Phase 10)")
        print(f"{'═'*70}")
        print(f"  {'Rang':>4} | {'Agent':20} | {'WR semi-pro':>12} | {'WR moyen':>10} | {'vs TAG':>8}")
        print(f"  {'─'*4}-+-{'─'*20}-+-{'─'*12}-+-{'─'*10}-+-{'─'*8}")
        for rang, (nom, rapport) in enumerate(rapports.items(), 1):
            print(f"  {rang:>4} | {nom:20} | "
                  f"{rapport.winrate_moyen_semipro:>+12.1f} | "
                  f"{rapport.winrate_moyen:>+10.1f} | "
                  f"{rapport.winrate_vs_tag:>+8.1f}")
        print(f"{'═'*70}\n")


# =============================================================================
# FONCTIONS UTILITAIRES RAPIDES
# =============================================================================

def evaluer_agent(agent,
                  nb_mains          : int   = 500,
                  blindes           : tuple = (10, 20),
                  graine            : int   = 42,
                  verbose           : bool  = True,
                  inclure_baselines : bool  = True) -> RapportEvaluation:
    evaluateur = Evaluateur(
        nb_mains          = nb_mains,
        blindes           = blindes,
        graine            = graine,
        verbose           = verbose,
        inclure_baselines = inclure_baselines,
    )
    rapport = evaluateur.evaluer(agent)
    if verbose:
        rapport.afficher()
    return rapport


def benchmark_rapide(agent, nb_mains: int = 200) -> float:
    evaluateur = Evaluateur(
        nb_mains          = nb_mains,
        graine            = 42,
        verbose           = False,
        inclure_baselines = False,
    )
    rapport = evaluateur.evaluer(agent)
    return rapport.winrate_moyen_semipro


# =============================================================================
# TEST RAPIDE
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*62)
    print("  AXIOM — Test evaluator.py (Phase 10 — stats sources)")
    print("="*62)

    from ai.agent import AgentAXIOM
    from training.self_play import AgentTAG, AgentLAG, AgentRegulier

    print("\n  Test 1 : évaluation complète (200 mains/scénario)")
    agent = AgentAXIOM(mode_deterministe=False)
    evaluateur = Evaluateur(nb_mains=200, graine=42, verbose=True)
    rapport = evaluateur.evaluer(agent)
    rapport.afficher()

    assert isinstance(rapport.winrate_vs_tag,        float)
    assert isinstance(rapport.winrate_moyen_semipro, float)
    assert len(rapport.stats_sources) > 0, "Stats sources manquantes"
    print("  ✅ Stats sources collectées")

    print("\n  Test 2 : évaluation semi-pro seulement (100 mains)")
    evaluateur2 = Evaluateur(nb_mains=100, graine=42, verbose=True,
                             inclure_baselines=False)
    rapport2 = evaluateur2.evaluer(agent)
    rapport2.afficher()
    assert np.isnan(rapport2.winrate_vs_aleatoire)
    print("  ✅ Mode sans baselines OK")

    print("\n  Test 3 : benchmark_rapide (100 mains)")
    wr = benchmark_rapide(AgentAXIOM(), nb_mains=100)
    print(f"  Winrate semi-pro moyen : {wr:+.1f} bb/100")
    assert isinstance(wr, float)
    print("  ✅ benchmark_rapide OK")

    print("\n  ✅ Tous les tests evaluator.py sont passés !")
    print("="*62 + "\n")
