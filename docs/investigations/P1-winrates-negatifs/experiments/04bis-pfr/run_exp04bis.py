"""
Exp 04 bis — Evaluation PFR fix (hyper_agressif)
=================================================
3 runs x 6 bots x 1000 mains = 18 000 mains totales.
Compare vs Exp 04 ET vs baseline H3-final.

Usage :
    python docs/.../04bis-pfr/run_exp04bis.py              # run complet
    python docs/.../04bis-pfr/run_exp04bis.py --sanity     # mini run 50 mains vs raise-only
    python docs/.../04bis-pfr/run_exp04bis.py --seed 42    # un seul seed
"""
import argparse
import json
import os
import sys
import statistics
import time
from datetime import datetime

import numpy as np

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')
)
sys.path.insert(0, ROOT)

from ai.agent import AgentAXIOM, creer_agent
from ai.opponent_tracker import OpponentTracker
from ai.exploit_mixer import ExploitMixer
from training.self_play import (
    simuler,
    AgentAleatoire, AgentCallOnly, AgentRaiseOnly,
    AgentTAG, AgentLAG, AgentRegulier,
)
from config.settings import STACK_DEPART, EXPLOIT_LOG_ENABLED, EXPLOIT_LOG_PATH

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(ROOT, EXPLOIT_LOG_PATH)

SEEDS    = [42, 123, 2026]
NB_MAINS = 1000
BLINDES  = (10, 20)
GB       = BLINDES[1]

# Baseline H3-final
BASELINE = {
    'aleatoire': [-77.6, -59.2, -69.9],
    'call':      [-71.5, -10.4, -18.4],
    'raise':     [-86.7, -73.4, -95.1],
    'tag':       [+21.1, +25.6, +14.3],
    'lag':       [ -3.6,  -1.7,  +4.2],
    'regulier':  [ +6.6,  +8.1,  -1.0],
}

# Exp 04 resultats (pour comparaison 3 colonnes)
EXP04 = {
    'aleatoire': [-75.1, -69.2, -68.5],
    'call':      [ -2.8, +43.4,  +2.7],
    'raise':     [-76.4, -83.8, -71.7],
    'tag':       [ +6.0, +10.6,  +4.5],
    'lag':       [ -4.6, -11.6,  -4.7],
    'regulier':  [+12.9,  +6.8, +20.0],
}

BOTS = ['aleatoire', 'call', 'raise', 'tag', 'lag', 'regulier']

# Criteres Exp 04 bis
# raise-only : objectif +30 vs Exp 04 (-77.3 + 30 = -47.3)
# call-only  : pas de regression > 1.5*sigma_exp04 (~25 => seuil -23.4)
CRITERES = {
    'raise':     {'seuil': -47.3,  'desc': '>= -47.3 (+30 vs Exp04 -77.3)'},
    'call':      {'seuil': -23.4,  'desc': '>= -23.4 (pas regression call-only)'},
    'tag':       {'seuil':   0.0,  'desc': '> 0 (critique)'},
    'lag':       {'seuil': -15.0,  'desc': '>= -15.0'},
    'regulier':  {'seuil':   0.0,  'desc': '> 0 (critique)'},
    'aleatoire': {'seuil': -75.9,  'desc': '>= -75.9 (pas regression > 5 vs Exp04)'},
}

# =============================================================================
# EXPLOIT LOGGING
# =============================================================================

_log_entries  = []
_current_bot  = 'inconnu'
_current_seed = 0
_orig_ajuster = (ExploitMixer.ajuster.__func__
                 if hasattr(ExploitMixer.ajuster, '__func__')
                 else ExploitMixer.ajuster)


def _ajuster_avec_log(self, distribution_blueprint, seat_index, game_type):
    c      = self._tracker.confiance(seat_index)
    profil = self._detecter_profil(seat_index) if c > 0.0 else 'neutre'
    result = _orig_ajuster(self, distribution_blueprint, seat_index, game_type)
    _log_entries.append({
        'bot':          _current_bot,
        'seed':         _current_seed,
        'seat':         seat_index,
        'profil':       profil,
        'confiance':    round(float(c), 3),
        'game_type':    game_type,
        'vpip':         round(float(self._tracker.vpip(seat_index)), 3),
        'pfr':          round(float(self._tracker.pfr(seat_index)), 3),
        'fold_to_cbet': round(float(self._tracker.fold_to_cbet(seat_index)), 3),
        'mains_obs':    self._tracker.mains_observees(seat_index),
        'modified':     not np.allclose(result, distribution_blueprint, atol=1e-6),
    })
    return result


def _activer_logging():
    if EXPLOIT_LOG_ENABLED:
        ExploitMixer.ajuster = _ajuster_avec_log


def _sauver_log():
    if not EXPLOIT_LOG_ENABLED or not _log_entries:
        return
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        for entry in _log_entries:
            f.write(json.dumps(entry) + '\n')
    print(f"  Log exploit : {len(_log_entries):,} entrees -> {LOG_PATH}")


# =============================================================================
# UTILITAIRES
# =============================================================================

def _reset_tracker(agent):
    agent.tracker = OpponentTracker()
    agent.mixer   = ExploitMixer(agent.tracker)


def _faire_bots(nom, seed):
    s2 = seed + 1 if seed else None
    return {
        'aleatoire': (AgentAleatoire(seed), AgentAleatoire(s2)),
        'call':      (AgentCallOnly(),       AgentCallOnly()),
        'raise':     (AgentRaiseOnly(),      AgentRaiseOnly()),
        'tag':       (AgentTAG(seed),        AgentTAG(seed + 10 if seed else None)),
        'lag':       (AgentLAG(seed),        AgentLAG(seed + 20 if seed else None)),
        'regulier':  (AgentRegulier(seed),   AgentRegulier(seed + 30 if seed else None)),
    }[nom]


def evaluer_bot(agent, bot_nom, seed, nb_mains=NB_MAINS, verbose=False):
    global _current_bot, _current_seed
    _current_bot  = bot_nom
    _current_seed = seed

    _reset_tracker(agent)
    adv1, adv2  = _faire_bots(bot_nom, seed)
    nb_par_pos  = nb_mains // 3
    winrates    = []
    graines_pos = [seed, seed + 1000 if seed else None, seed + 2000 if seed else None]

    for pos_axiom in range(3):
        agents_liste = (
            [agent, adv1, adv2] if pos_axiom == 0 else
            [adv1, agent, adv2] if pos_axiom == 1 else
            [adv1, adv2, agent]
        )
        res = simuler(
            agents        = agents_liste,
            nb_mains      = nb_par_pos,
            stacks_depart = STACK_DEPART,
            blindes_fixes = BLINDES,
            graine        = graines_pos[pos_axiom],
            verbose       = False,
        )
        winrates.append(res.winrate_bb100(pos_axiom, GB))

    wr = sum(winrates) / 3.0
    if verbose:
        print(f"    vs {bot_nom:10s} (seed={seed}): {wr:+.1f} bb/100  "
              f"[BTN={winrates[0]:+.1f} SB={winrates[1]:+.1f} BB={winrates[2]:+.1f}]")
    return wr


# =============================================================================
# SANITY CHECK (50 mains vs raise-only, seed=42)
# =============================================================================

def run_sanity_check():
    print("\n" + "="*62)
    print("  SANITY CHECK — 50 mains vs raise-only (seed=42)")
    print("="*62)

    agent = creer_agent(verbose=True)
    _log_entries.clear()

    wr = evaluer_bot(agent, 'raise', seed=42, nb_mains=50, verbose=True)
    print(f"\n  Winrate : {wr:+.1f} bb/100")

    ok = True

    if EXPLOIT_LOG_ENABLED and _log_entries:
        from collections import Counter
        dist_profils = Counter(e['profil'] for e in _log_entries)
        print(f"  Distribution profils : {dict(dist_profils)}")

        n_hyper = dist_profils.get('hyper_agressif', 0)
        n_call  = dist_profils.get('calling_station', 0)

        if n_hyper > 0:
            print(f"  [OK] 'hyper_agressif' detecte ({n_hyper} fois)")
        else:
            print(f"  [FAIL] 'hyper_agressif' jamais detecte — bug PFR ?")
            ok = False

        if n_call > 0:
            print(f"  [WARN] 'calling_station' encore detecte ({n_call} fois) — "
                  f"normal en debut de run si pfr momentanement bas")

        conf_max = max(e['confiance'] for e in _log_entries)
        pfr_max  = max(e['pfr']       for e in _log_entries)
        print(f"  Confiance max : {conf_max:.3f}")
        print(f"  PFR max       : {pfr_max:.3f}")
        if pfr_max >= 0.5:
            print(f"  [OK] PFR eleve confirme (>= 0.5)")
        else:
            print(f"  [WARN] PFR max = {pfr_max:.3f} — tracker peut etre sous-alimente")
    else:
        print("  [INFO] Logging desactive ou aucune entree")

    print(f"\n  Statut : {'OK — pret pour eval complete' if ok else 'STOP — debug requis'}")
    print("="*62 + "\n")
    return ok


# =============================================================================
# EVALUATION COMPLETE
# =============================================================================

def run_exp04bis(seeds=None, nb_mains=NB_MAINS):
    seeds        = seeds or SEEDS
    resultats    = {bot: [] for bot in BOTS}
    timing_debut = time.time()

    print("\n" + "="*62)
    print(f"  Exp 04 bis — PFR fix (hyper_agressif)")
    print(f"  {len(seeds)} seeds x 6 bots x {nb_mains} mains = "
          f"{len(seeds)*6*nb_mains:,} mains totales")
    print(f"  EXPLOIT_LOG_ENABLED = {EXPLOIT_LOG_ENABLED}")
    print("="*62)

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        agent = creer_agent(verbose=False)

        resultats_seed = {}
        for bot in BOTS:
            t0 = time.time()
            wr = evaluer_bot(agent, bot, seed, nb_mains=nb_mains, verbose=False)
            dt = time.time() - t0
            resultats_seed[bot] = wr
            resultats[bot].append(wr)
            print(f"  vs {bot:10s}: {wr:+.1f} bb/100  ({dt:.0f}s)")

        out_path = os.path.join(OUT_DIR, f'results_seed_{seed}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                'seed':      seed,
                'nb_mains':  nb_mains,
                'timestamp': datetime.now().isoformat(),
                'resultats': resultats_seed,
            }, f, indent=2)
        print(f"  Sauvegarde : {out_path}")

    _sauver_log()
    _afficher_rapport(resultats, nb_mains, seeds)

    duree = time.time() - timing_debut
    print(f"\n  Duree totale : {duree/60:.1f} min")
    return resultats


# =============================================================================
# RAPPORT TRIPLE COMPARAISON
# =============================================================================

def _afficher_rapport(resultats, nb_mains, seeds):
    print("\n" + "="*110)
    print("  RAPPORT FINAL — Exp 04 bis PFR fix")
    print("="*110)
    print(f"\n  {'Bot':12} | {'Moy 04bis':>9} | {'sigma':>5} | "
          f"{'Moy 04':>7} | {'D vs 04':>7} | "
          f"{'Moy base':>8} | {'D vs base':>9} | Statut")
    print("  " + "-"*102)

    verdicts = {}
    for bot in BOTS:
        vals_bis  = resultats[bot]
        moy_bis   = statistics.mean(vals_bis)
        std_bis   = statistics.stdev(vals_bis) if len(vals_bis) > 1 else 0.0
        moy_04    = statistics.mean(EXP04[bot])
        moy_base  = statistics.mean(BASELINE[bot])
        d04       = moy_bis - moy_04
        dbase     = moy_bis - moy_base

        seuil = CRITERES[bot]['seuil']
        ok    = moy_bis > seuil if bot in ('tag', 'regulier') else moy_bis >= seuil
        verdicts[bot] = ok
        statut = "OK" if ok else ("REVERT!" if bot in ('tag', 'regulier') else "ECHEC")

        print(f"  {bot:12} | {moy_bis:+9.1f} | {std_bis:5.1f} | "
              f"{moy_04:+7.1f} | {d04:+7.1f} | "
              f"{moy_base:+8.1f} | {dbase:+9.1f} | {statut}")

    print("  " + "-"*102)

    n_ok   = sum(1 for v in verdicts.values() if v)
    revert = not verdicts.get('tag', True) or not verdicts.get('regulier', True)

    print(f"\n  Criteres reussis  : {n_ok}/6")
    print(f"  Revert necessaire : {'OUI' if revert else 'NON'}")

    print(f"\n  {'─'*60}")
    if revert:
        print("  VERDICT : REVERT — regression TAG ou Regulier")
    elif n_ok >= 5:
        print("  VERDICT : MERGE RECOMMANDE — >= 5/6 criteres reussis")
    elif n_ok >= 3:
        print("  VERDICT : MERGE CONDITIONNEL — ameliorations partielles")
    else:
        print("  VERDICT : ANALYSE REQUISE")
    print(f"  {'─'*60}")

    if EXPLOIT_LOG_ENABLED and _log_entries:
        _analyser_log_exploit()


def _analyser_log_exploit():
    from collections import Counter, defaultdict

    print("\n" + "="*80)
    print("  ANALYSE LOG EXPLOIT")
    print("="*80)

    total       = len(_log_entries)
    non_neutres = [e for e in _log_entries if e['profil'] != 'neutre']
    actives     = [e for e in non_neutres if e['confiance'] >= 0.3]

    print(f"\n  Appels totaux    : {total:,}")
    print(f"  Profil != neutre : {len(non_neutres):,} ({100*len(non_neutres)/max(total,1):.1f}%)")
    print(f"  Taux activation  : {len(actives):,} ({100*len(actives)/max(total,1):.1f}%)")
    print(f"  Decisions modif. : {sum(1 for e in _log_entries if e['modified']):,}")

    print(f"\n  Distribution profils :")
    for profil, cnt in Counter(e['profil'] for e in _log_entries).most_common():
        print(f"    {profil:20s}: {cnt:6,} ({100*cnt/total:.1f}%)")

    print(f"\n  Par bot (conf >= 0.3) :")
    par_bot = defaultdict(list)
    for e in _log_entries:
        par_bot[e['bot']].append(e)

    attendu = {
        'aleatoire': ['calling_station', 'hyper_agressif', 'neutre'],
        'call':      ['calling_station'],
        'raise':     ['hyper_agressif'],
        'tag':       ['neutre'],
        'lag':       ['neutre'],
        'regulier':  ['neutre'],
    }

    coherence_ok = True
    for bot in BOTS:
        entries = par_bot.get(bot, [])
        if not entries:
            print(f"    {bot:12s}: aucune entree")
            continue
        actifs     = [e for e in entries if e['confiance'] >= 0.3 and e['profil'] != 'neutre']
        profil_dom = (Counter(e['profil'] for e in actifs).most_common(1)[0][0]
                      if actifs else 'neutre')
        conf_moy   = statistics.mean(e['confiance'] for e in entries)
        vpip_moy   = statistics.mean(e['vpip'] for e in entries)
        pfr_moy    = statistics.mean(e['pfr']  for e in entries)
        pct_hyper  = (100 * sum(1 for e in actifs if e['profil'] == 'hyper_agressif')
                      / max(len(actifs), 1))

        coh = "OK" if profil_dom in attendu.get(bot, ['neutre']) else "INATTENDU"
        if coh != "OK":
            coherence_ok = False

        print(f"    {bot:12s}: profil={profil_dom:20s} conf={conf_moy:.3f} "
              f"vpip={vpip_moy:.3f} pfr={pfr_moy:.3f} "
              f"hyper%={pct_hyper:.0f}%  [{coh}]")

    print(f"\n  Coherence : {'OK' if coherence_ok else 'INCOHERENCES DETECTEES'}")
    print("="*80 + "\n")


# =============================================================================
# POINT D ENTREE
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exp 04 bis PFR evaluation')
    parser.add_argument('--sanity', action='store_true',
                        help='Mini run 50 mains vs raise-only (sanity check)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Lancer un seul seed (ex: --seed 42)')
    parser.add_argument('--mains', type=int, default=NB_MAINS,
                        help=f'Mains par run (defaut: {NB_MAINS})')
    args = parser.parse_args()

    _activer_logging()

    if args.sanity:
        ok = run_sanity_check()
        sys.exit(0 if ok else 1)
    else:
        seeds = [args.seed] if args.seed else SEEDS
        run_exp04bis(seeds=seeds, nb_mains=args.mains)
