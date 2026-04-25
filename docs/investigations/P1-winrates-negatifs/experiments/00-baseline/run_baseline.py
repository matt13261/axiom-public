"""
Baseline P1 — 3 runs indépendants (seeds=[42, 123, 2026]), 500 mains × 6 bots.
Produit baseline.json et result.md.
Usage : python docs/investigations/P1-winrates-negatifs/experiments/00-baseline/run_baseline.py
"""
import json
import os
import sys

# Ajouter la racine du projet au path (5 niveaux au-dessus de ce fichier)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
sys.path.insert(0, ROOT)

from ai.agent import creer_agent
from training.evaluator import Evaluateur

SEEDS = [42, 123, 2026]
NB_MAINS = 500
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BOTS = [
    ('aleatoire', 'winrate_vs_aleatoire'),
    ('call',      'winrate_vs_call'),
    ('raise',     'winrate_vs_raise'),
    ('tag',       'winrate_vs_tag'),
    ('lag',       'winrate_vs_lag'),
    ('regulier',  'winrate_vs_regulier'),
]


def run_baseline():
    agent = creer_agent()
    resultats = {bot: [] for bot, _ in BOTS}

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        evaluateur = Evaluateur(nb_mains=NB_MAINS, graine=seed, verbose=False)
        rapport = evaluateur.evaluer(agent)
        for bot, attr in BOTS:
            wr = getattr(rapport, attr)
            resultats[bot].append(wr)
            print(f"  vs {bot:10s}: {wr:+.1f} bb/100")

    # Sauver baseline.json
    json_path = os.path.join(OUT_DIR, 'baseline.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'seeds': SEEDS, 'nb_mains': NB_MAINS, 'resultats': resultats}, f, indent=2)
    print(f"\nSauvé : {json_path}")

    # Générer result.md
    import statistics
    lines = [
        "# Baseline P1 — Résultats",
        "",
        f"**Seeds :** {SEEDS}  ",
        f"**Mains par run :** {NB_MAINS}  ",
        "",
        "| Adversaire | Run 1 | Run 2 | Run 3 | Moyenne | Écart-type |",
        "|------------|------:|------:|------:|--------:|-----------:|",
    ]
    for bot, _ in BOTS:
        vals = resultats[bot]
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        row  = f"| {bot:10s} | {vals[0]:+6.1f} | {vals[1]:+6.1f} | {vals[2]:+6.1f} | {mean:+7.1f} | {std:9.1f} |"
        lines.append(row)

    lines += [
        "",
        "## Critère kill-switch",
        "",
        "Après Exp 01 + Exp 02, si winrate vs random < (baseline_mean + 5 bb/100) → passer directement à Exp 04.",
        "",
        f"**Baseline vs random (moyenne) : {statistics.mean(resultats['aleatoire']):+.1f} bb/100**",
        f"**Seuil kill-switch : {statistics.mean(resultats['aleatoire']) + 5:+.1f} bb/100**",
    ]

    md_path = os.path.join(OUT_DIR, 'result.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Sauvé : {md_path}")

    return resultats


if __name__ == '__main__':
    run_baseline()
