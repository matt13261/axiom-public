"""
Exp 01-H3-final : Re-mesure baseline post-revert (seed=42, 100 sims).
Verifie que les winrates sont equivalents a la baseline initiale (+-3 bb/100).

Usage : python docs/investigations/.../experiments/01-H3-final/run_exp01_final.py
"""
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
sys.path.insert(0, ROOT)

SEEDS    = [42, 123, 2026]
NB_MAINS = 500
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

BOTS = [
    ('aleatoire', 'winrate_vs_aleatoire'),
    ('call',      'winrate_vs_call'),
    ('raise',     'winrate_vs_raise'),
    ('tag',       'winrate_vs_tag'),
    ('lag',       'winrate_vs_lag'),
    ('regulier',  'winrate_vs_regulier'),
]

BASELINE = {
    'aleatoire': -68.4, 'call': 1.6, 'raise': -78.4,
    'tag': 13.5, 'lag': -5.8, 'regulier': 17.4,
}
SEUIL = 3.0


def run():
    from ai.agent import creer_agent
    from training.evaluator import Evaluateur

    agent    = creer_agent()
    resultats = {bot: [] for bot, _ in BOTS}

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        rapport = Evaluateur(nb_mains=NB_MAINS, graine=seed, verbose=False).evaluer(agent)
        for bot, attr in BOTS:
            wr = getattr(rapport, attr)
            resultats[bot].append(wr)
            print(f"  vs {bot:10s}: {wr:+.1f} bb/100")

    _ecrire_result_md(resultats)
    return resultats


def _ecrire_result_md(resultats):
    alerte = False
    lines = [
        "# Exp 01-H3-final — Post-revert baseline (seed=42, 100 sims)",
        "",
        f"**Seeds :** {SEEDS}  ",
        f"**Seuil degradation acceptable :** +/-{SEUIL} bb/100  ",
        "",
        "| Adversaire | Run 1 | Run 2 | Run 3 | Moyenne | Baseline | Delta | Statut |",
        "|------------|------:|------:|------:|--------:|---------:|------:|:-------|",
    ]
    for bot, _ in BOTS:
        vals  = resultats[bot]
        mean  = statistics.mean(vals)
        base  = BASELINE[bot]
        delta = mean - base
        ok    = abs(delta) <= SEUIL
        if not ok:
            alerte = True
        statut = "OK" if ok else f"ALERTE (|{delta:+.1f}|>{SEUIL})"
        lines.append(
            f"| {bot:10s}"
            f" | {vals[0]:+6.1f} | {vals[1]:+6.1f} | {vals[2]:+6.1f}"
            f" | {mean:+7.1f} | {base:+8.1f} | {delta:+5.1f} | {statut} |"
        )

    verdict = "BASELINE CONFIRMEE" if not alerte else "ALERTE — reverter aussi la seed"
    lines += ["", f"**Verdict : {verdict}**"]

    with open(os.path.join(OUT_DIR, 'result.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nVerdict : {verdict}")


if __name__ == '__main__':
    run()
