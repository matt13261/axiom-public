# P1 — Winrates négatifs vs bots triviaux : Contexte

## Mesures reproductibles (2 runs indépendants, 500 mains chacun)

| Adversaire       | Run 1 (bb/100) | Run 2 (bb/100) | Statut   |
|------------------|---------------|----------------|----------|
| AgentAleatoire   | -71.9         | -73.2          | NEGATIF  |
| AgentCallOnly    | -31.1         | -73.6          | NEGATIF  |
| AgentRaiseOnly   | -73.3         | -82.0          | NEGATIF  |
| AgentTAG         | +17.9         | +12.5          | POSITIF  |
| AgentLAG         | -14.6         | -15.9          | NEGATIF  |
| AgentRegulier    | +17.2         | +3.3           | POSITIF  |

Log eval Deep CFR iter 109 (2026-04-24 13:18 et 13:24) — même pattern :
- vs 20M MCCFR blueprint : -68.6 / -69.4 vs random
- vs 10K iterations : -89.3 / -85.0 vs raise-only

## Architecture au moment du diagnostic

- **MCCFR** : External Sampling, Linear MCCFR, 20M itérations
  - Blueprint : 2,584,465 infosets, ~373 MB pkl
  - Hit rate blueprint : 63%
  - Fallback heuristique : 37%
- **Deep CFR** : 109/500 itérations (en cours)
  - 3 réseaux par joueur (regret, stratégie, valeur)
  - 281K paramètres par réseau regret/stratégie
  - Buffers : ~570 MB (regrets + stratégies)
- **Abstraction** : 8 buckets postflop, 7 actions abstraites
  - Bucketing : Monte Carlo 100 simulations, seed NON FIXÉE (bruité)
- **Stacks** : 25 BB (court, proche du push/fold)
- **Heuristique fallback BTN** : CHECK=0.60, CALL=0.60 (passive)
- **Module d'exploitation** : ABSENT (MCCFR cherche Nash pur)

## Observations clés

1. AXIOM bat les adversaires **exploitables mais intelligents** (TAG +13-18 bb/100,
   Régulier +3-17 bb/100) → la stratégie Nash fonctionne contre les joueurs
   qui ont des fuites standard.

2. AXIOM perd massivement contre les adversaires **absurdes** (random, call-only,
   raise-only) → comportement GTO pur face à adversaires non-exploitables.

3. **Libratus/Pluribus** résolvent ce problème via un module d'exploitation
   en temps réel (self-improver / safe subgame solving adaptatif).

4. Le 37% de fallback heuristique est une source probable de pertes importantes :
   la heuristique BTN (CHECK=0.60) est trop passive face aux raises répétés.

5. Le bucketing bruité (100 MC sans seed fixe) peut provoquer des incohérences
   lookup blueprint → action prise ≠ action optimale pour la main réelle.

## Fichiers pertinents

- `ai/agent.py` — logique de décision, fallback heuristique
- `ai/mccfr.py` — algorithme MCCFR
- `abstraction/card_abstraction.py` — bucketing 8 buckets, 100 MC sims
- `abstraction/info_set.py` — construction des clés d'infoset
- `training/evaluator.py` — calcul des winrates (vérifié correct)
- `config/settings.py` — hyperparamètres (TAILLES_MISE, STACK_DEPART=500)
- `data/logs/training_log.csv` — historique des évaluations
