# P8.diag.critique — V1 sans Deep CFR vs P7 sans Deep CFR

**Date** : 2026-04-30
**Objectif** : trancher entre 4 hypothèses sur l'origine du déficit P7 vs V1
en ajoutant la quatrième cellule manquante au tableau diagnostique.
**Protocole** : éval V1 (1 seed=42 × 6 baselines × 10K mains) avec patch
`agent._reseaux_strategie = None` et `agent._reseaux_valeur = None`, sous
branche `p6-pre-bascule` (commit `2331436`).

**STOP avant interprétation** comme demandé. Tableau livré, interprétation
à faire ensemble.

---

## 1. Tableau comparatif 4 cellules

Métrique : winrate **bb/100** (positif = AXIOM gagne, négatif = AXIOM perd).
Toutes les évals sont sur seed 42 × 10 000 mains × 6 baselines.

| Baseline    | V1 + DCFR | **V1 sans DCFR** | P7 + DCFR | **P7 sans DCFR** |
|---          |---:       |---:              |---:       |---:              |
| Aléatoire   | -74.82    | **-90.06**       | -48.11    | **-117.93**      |
| Call-Only   | +17.77    | **+2.54**        | -14.55    | **-10.85**       |
| Raise-Only  | -83.85    | **-578.44**      | -51.62    | **-696.70**      |
| TAG         | +15.28    | **+1.63**        | -8.97     | **-34.14**       |
| LAG         | -4.29     | **-35.56**       | -17.87    | **-102.44**      |
| Régulier    | +10.61    | **-28.47**       | -8.45     | **-72.97**       |
| **Moyenne** | **-19.88** | **-121.39**     | **-24.93** | **-172.51**     |

### Δ (sans DCFR − avec DCFR), par bloc

| Baseline    | Δ V1 (sans − avec)  | Δ P7 (sans − avec)  |
|---          |---:                  |---:                  |
| Aléatoire   | -15.24               | -69.82               |
| Call-Only   | -15.23               | +3.70                |
| Raise-Only  | **-494.59**          | **-645.08**          |
| TAG         | -13.65               | -25.17               |
| LAG         | -31.27               | -84.57               |
| Régulier    | -39.08               | -64.52               |
| **Moy**     | **-101.51**          | **-147.58**          |

---

## 2. Sources données

| Cellule | Fichier |
|---|---|
| V1 + DCFR (seed 42) | `data/strategy/p8_eval_v1_results.json` (clé `42`) |
| V1 sans DCFR (seed 42) | `data/strategy/p8_eval_v1_no_dcfr_results.json` |
| P7 + DCFR (seed 42) | `data/strategy/p8_eval_p7_results.json` (clé `42`) |
| P7 sans DCFR (seed 42) | `data/strategy/p8_eval_p7_no_dcfr_results.json` |

Branche d'éval :
- V1 cellules : `git checkout p6-pre-bascule` (commit `2331436`, V1 buckets actifs)
- P7 cellules : `git checkout main` (HEAD `967f0e1`, V2 + P7 actifs)

Patch appliqué (sans commit) dans les cellules "sans DCFR" :
```python
agent._reseaux_strategie = None
agent._reseaux_valeur    = None
```

---

## 3. Quelques observations factuelles (pas d'interprétation)

1. **V1 sans DCFR n'est PAS catastrophique sur tous les bots.** Reste positif
   contre Call-Only (+2.54) et TAG (+1.63). Les autres dégradent sensiblement.
2. **V1 sans DCFR vs Raise-Only = -578.44** : effondrement même en V1, similaire
   à P7 sans DCFR (-696.70). Le pattern raise-only mécanique est mortel sans
   DCFR dans les deux régimes.
3. **L'écart V1 ↔ V1-sans-DCFR (Δ moy -101.51) est plus petit que P7 ↔ P7-sans-DCFR
   (Δ moy -147.58)** — DCFR aide plus en P7 qu'en V1 (en valeur absolue moyenne).
4. **Tous les bots structurés (Call, TAG, LAG, Régulier) en V1+DCFR sont positifs ou
   ~0** ; en V1-sans-DCFR les positifs deviennent ~0 ou négatifs marginaux ; en
   P7+DCFR ils sont tous négatifs ; en P7-sans-DCFR ils sont catastrophiques.
5. **Le patch agit bien** : log montre `Après patch : reseaux_strategie=absent`
   dans les deux runs. Pas de crash.
6. **Durée V1-sans-DCFR (317s)** beaucoup plus rapide que V1-avec-DCFR
   (~900s) — pas de forward réseau à chaque décision.

---

## 4. État technique post-diag

| Item | État |
|---|---|
| Branche actuelle | `main` |
| Patch reverté | ✓ (n'a jamais été commit, juste en mémoire d'un process Python) |
| Blueprints intacts | ✓ md5 V1 et P7-4M identiques au backup `/tmp/axiom_blueprints_safe/` |
| Stash restored | ✓ |
| Résultats sauvegardés | `data/strategy/p8_eval_v1_no_dcfr_results.json` |

---

## 5. STOP

Tableau 4 cellules livré, données factuelles ci-dessus. **Pas
d'interprétation finale dans ce rapport** — à faire ensemble.

Les 3 cas que tu as définis (A, B, C) correspondent à des distributions
différentes des deltas observés. À toi pour décider lequel s'applique le
plus, et ce qu'on en déduit pour la suite.
