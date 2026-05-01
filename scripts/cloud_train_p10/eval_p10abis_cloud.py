#!/usr/bin/env python3
"""
P10.A.bis cloud - variance OFT (tracker frais par baseline).

36 runs : 6 baselines x 3 seeds x 2 conditions, 1500 mains/run.
Agent recree pour chaque (oft, seed, baseline).
Contrainte memoire : RSS cap 7500 MB / worker.
"""
import gc, json, math, os, statistics, sys, time
from multiprocessing import Pool, set_start_method
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

NB_MAINS  = 1500
SEEDS     = [42, 1337, 2024]
BASELINES = ['aleatoire', 'call', 'raise', 'tag', 'lag', 'regulier']
BLUEPRINT = 'data/strategy/blueprint_v1.pkl'
N_WORKERS = int(os.environ.get('P10_WORKERS', '8'))
RSS_CAP_MB = 7500
OUTPUT_LOG = Path('p10abis_runs.jsonl')
OUTPUT_MD  = Path('docs/investigations/P10/audit_oft.md')


def _check_rss():
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        if rss_mb > RSS_CAP_MB:
            print(f"[RSS-OVERFLOW] worker pid={os.getpid()} RSS={rss_mb:.0f}MB > {RSS_CAP_MB}MB", flush=True)
            sys.exit(1)
        return rss_mb
    except ImportError:
        return -1.0


def _make_baseline(nom, seed):
    from training.self_play import (
        AgentAleatoire, AgentCallOnly, AgentRaiseOnly,
        AgentTAG, AgentLAG, AgentRegulier,
    )
    g = seed
    d = seed + 100 if seed else None
    if nom == 'aleatoire': return [AgentAleatoire(g),  AgentAleatoire(d)]
    if nom == 'call':      return [AgentCallOnly(),    AgentCallOnly()]
    if nom == 'raise':     return [AgentRaiseOnly(),   AgentRaiseOnly()]
    if nom == 'tag':       return [AgentTAG(g),        AgentTAG((g + 10) if g else None)]
    if nom == 'lag':       return [AgentLAG(g),        AgentLAG((g + 20) if g else None)]
    if nom == 'regulier':  return [AgentRegulier(g),   AgentRegulier((g + 30) if g else None)]
    raise ValueError(f"Baseline inconnue : {nom}")


def _run_one(args):
    oft, seed, baseline = args
    t0 = time.time()
    sys.path.insert(0, str(ROOT))
    from ai.agent import creer_agent
    from training.evaluator import Evaluateur

    agent = creer_agent(chemin_blueprint=BLUEPRINT, verbose=False, enable_oft=oft)
    evaluator = Evaluateur(nb_mains=NB_MAINS, blindes=(10, 20), graine=seed,
                            verbose=False, inclure_baselines=False)
    adversaires = _make_baseline(baseline, seed)
    wr, _ = evaluator._winrate_scenario(agent, adversaires, label=f"{baseline}/seed{seed}/oft{oft}")

    rss_mb = _check_rss()
    duree = time.time() - t0
    result = {'oft': oft, 'seed': seed, 'baseline': baseline,
              'wr_bb100': wr, 'duree_s': duree, 'rss_mb': rss_mb, 'pid': os.getpid()}
    print(json.dumps(result), flush=True)
    del agent, evaluator, adversaires
    gc.collect()
    return result


def main():
    print(f"=== P10.A.bis cloud - variance OFT (tracker frais) ===", flush=True)
    print(f"Start    : {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"NB_MAINS : {NB_MAINS} | Seeds : {SEEDS} | Workers : {N_WORKERS}", flush=True)
    jobs = [(oft, s, b) for oft in (True, False) for s in SEEDS for b in BASELINES]
    print(f"Total    : {len(jobs)} runs\n", flush=True)
    t_global = time.time()
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    results = []
    with Pool(processes=N_WORKERS) as pool:
        for r in pool.imap_unordered(_run_one, jobs):
            results.append(r)
            with open(OUTPUT_LOG, 'a') as f:
                f.write(json.dumps(r) + '\n')
    duree_total = time.time() - t_global
    pic_rss = max((r['rss_mb'] for r in results if r['rss_mb'] > 0), default=-1)
    print(f"\n=== Tous runs termines en {duree_total/60:.1f} min === Pic RSS {pic_rss:.0f} MB", flush=True)

    labels = {'aleatoire':'Aleatoire','call':'Call-Only','raise':'Raise-Only','tag':'TAG','lag':'LAG','regulier':'Regulier'}
    synth = {}
    for b in BASELINES:
        on  = sorted([r['wr_bb100'] for r in results if r['baseline']==b and r['oft'] is True])
        off = sorted([r['wr_bb100'] for r in results if r['baseline']==b and r['oft'] is False])
        m_on  = statistics.mean(on)  if on  else 0.0
        s_on  = statistics.stdev(on) if len(on)  > 1 else 0.0
        m_off = statistics.mean(off) if off else 0.0
        s_off = statistics.stdev(off) if len(off) > 1 else 0.0
        delta = m_on - m_off
        sigma_delta = math.sqrt(s_on**2 + s_off**2)
        ic95 = 1.96 * sigma_delta / math.sqrt(len(SEEDS))
        synth[b] = {'on':on,'off':off,'m_on':m_on,'s_on':s_on,'m_off':m_off,'s_off':s_off,
                    'delta':delta,'ic95_half':ic95,
                    'verdict': 'NUISIBLE' if delta+ic95<0 else ('UTILE' if delta-ic95>0 else 'BRUIT')}

    print("\n=== Synthese par baseline ===", flush=True)
    print(f"{'Baseline':<11} | {'mu_ON':>7} | {'s_ON':>6} | {'mu_OFF':>7} | {'s_OFF':>6} | {'Delta':>7} | {'IC95':>7} | Verdict", flush=True)
    for b in BASELINES:
        s = synth[b]
        print(f"{labels[b]:<11} | {s['m_on']:>+7.2f} | {s['s_on']:>6.2f} | {s['m_off']:>+7.2f} | {s['s_off']:>6.2f} | {s['delta']:>+7.2f} | +-{s['ic95_half']:>5.2f} | {s['verdict']}", flush=True)

    md = ["\n---\n\n",
          "## P10.A.bis cloud - variance OFT (tracker frais)\n\n",
          f"**Date** : {time.strftime('%Y-%m-%d')}\n\n",
          "**Protocole corrige** vs P10.A : agent FRAIS recree pour chaque (oft, seed, baseline) -> tracker OFT vide en debut de chaque baseline. L'audit cross-baseline a confirme que P10.A polluait le tracker.\n\n",
          f"**Parametres** : {NB_MAINS} mains x {len(SEEDS)} seeds x {len(BASELINES)} baselines x 2 conditions = {len(jobs)} runs\n\n",
          f"**Cloud** : VM axiom-training-24 (n2-standard-32), {N_WORKERS} workers paralleles\n\n",
          f"**Duree** : {duree_total/60:.1f} min\n\n",
          f"**Pic RSS observe** : {pic_rss:.0f} MB / worker (cap {RSS_CAP_MB} MB)\n\n",
          "### Tableau synthese moyenne, sigma, IC 95%\n\n",
          "| Baseline | mu OFT_ON | sigma ON | mu OFT_OFF | sigma OFF | Delta (ON-OFF) | IC 95% | Verdict |\n",
          "|---|---:|---:|---:|---:|---:|---:|---|\n"]
    for b in BASELINES:
        s = synth[b]
        md.append(f"| {labels[b]} | {s['m_on']:+.2f} | {s['s_on']:.2f} | {s['m_off']:+.2f} | {s['s_off']:.2f} | **{s['delta']:+.2f}** | +-{s['ic95_half']:.2f} | {s['verdict']} |\n")
    md.append("\n### Donnees brutes (36 runs)\n\n```jsonl\n")
    for r in sorted(results, key=lambda x: (x['oft'], x['seed'], x['baseline'])):
        md.append(json.dumps(r) + "\n")
    md.append("```\n")

    if OUTPUT_MD.exists():
        with open(OUTPUT_MD, 'a', encoding='utf-8') as f:
            f.writelines(md)
    else:
        OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
            f.writelines(md)
    print(f"\nRapport: {OUTPUT_MD}\nLog   : {OUTPUT_LOG}", flush=True)


if __name__ == '__main__':
    main()
