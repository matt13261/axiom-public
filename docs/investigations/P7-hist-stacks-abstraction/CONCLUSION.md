# P7 — Conclusion et archive

**Statut** : **ARCHIVÉ** — reverté vers pre-p7 le 2026-04-30
**Tag de référence** : `p7-complete-archived` (commit `c371e21`)
**Tag de retour** : `pre-p7` (commit `edefa7f`)

---

## Le finding majeur (le vrai gain de ce sprint)

**`creer_agent()` n'activait PAS les composants Pluribus essentiels** — les
continuation strategies (k=4) et les solveurs real-time (FLOP / TURN-RIVER)
existaient en code mais n'étaient appelés que dans `screen/jouer.py` (mode
prod). Toutes les évaluations baseline depuis P1 (avril 2026) ont mesuré un
**agent mutilé** au lieu de la "vraie" stratégie AXIOM.

L'audit architectural (rapport `architecture_audit.md`) a révélé ce trou
TA1 et le fix `feat(agent): activate continuation strategies + real-time
solver in creer_agent` (commit `de0c573`) le corrige. **Ce fix est préservé
post-revert** car indépendant de l'abstraction P7.

L'effet est massif :
- V1 mode mutilé : -20.32 bb/100 moyen
- V1 mode complet : **+1.22 bb/100 moyen** (positif sur 4/6 baselines)
- Δ moyen : **+21.54 bb/100** uniquement par activation Pluribus complet

C'est le seul "vrai" résultat post-V1 qui montre que l'agent peut être
positif contre les baselines. Tous les chiffres antérieurs (P1, P8) étaient
sous-estimés.

---

## P7 — Objectifs initiaux

P7 visait à résoudre l'explosion combinatoire des clés d'infoset détectée
en P6 :
- segment `hist` : 19 492 valeurs uniques en 2K iter MCCFR
- segment `stacks` : 1 416 valeurs uniques

Voir `spec.md` (commit `0cdf400`) et `p6-abstraction/spec.md` pour le
diagnostic.

---

## Ce qui a été fait

### Migration code (P7.1 → P7.4)
1. Spec P7 : abstraction sizing 3 buckets S/M/L (Variante B "Spin & Rush
   short-stack"), cap=4 actions/street, paliers stack 7 niveaux
   (`PALIERS_STACK_SPIN_RUSH = [0, 5, 8, 13, 19, 27, 41]`)
2. 15 tests RED atomiques (12 RED + 3 GUARDRAIL)
3. Implémentation helpers + migration `info_set.py`, `mccfr.py`, `train_hu.py`,
   `solver/depth_limited.py`, `solver/subgame_solver.py`
4. 207 tests GREEN, format clé 7 segments préservé, DIM_INPUT=52 préservé

### Mesures cardinalité (P7.5)
- hist : **23 146 → 417** (÷55× réduction)
- stacks : **1 391 → 229** (÷6× réduction)
- 5 paliers de saturation power-law (5K, 10K, 25K, 50K, 100K iter)
  → α = 0.667 → extrapolation 5M iter ratio ~1.3, blueprint ~1 GB

### Cloud (P7.6 + P7.7)
- Pilot 500K cloud (1.80€) : ratio 4.53 vs prédit 6.0 ✓
- Pilot 5M cloud (interrompu réseau VM à 4M iter, 24€) : 2.99M infosets,
  blueprint 487 MB

### Évaluation (P8 + P9)
- P8 (agent mutilé) : V1 +20.32 vs P7 -24.56 — P7 perd globalement
- P9 (agent Pluribus complet, post fix TA1) :
  V1 +1.22 vs P7 -19.94 — P7 perd toujours globalement

### Diagnostics (P8.diag, P8.diag.critique)
- Confound "Deep CFR pollué" RÉFUTÉ (désactiver DCFR aggrave dans les 2 régimes)
- Architecture audit révélant TA1

---

## Pourquoi le revert

### Critère décision validé

L'utilisateur a fixé : "Si l'écart V1 vs P7 sur les baselines structurées
(TAG, LAG, Régulier) est > 15 bb/100 : signal clair, on décide
directement."

| Baseline structurée | Δ V1 − P7 (Pluribus complet) |
|---|---:|
| TAG | **+25.75** ✓ |
| LAG | **+19.77** ✓ |
| Régulier | **+20.27** ✓ |

**Les 3 critères sont franchis.** P7 dégrade significativement les
performances vs V1 contre les bots structurés.

### Asymétrie continuations

P7 a été testé sans continuations k=4 (les variantes V1 sont incompatibles,
aucune variante P7 entraînée). Cela explique partiellement l'écart sur
Call-Only (+84.36 bb/100) mais pas l'écart structurel sur TAG/LAG/Régulier.

Re-train continuations P7 cloud (~50€) aurait pu combler une partie, mais
le risque de ne pas combler l'écart structurel est élevé. Décision : ne pas
investir.

### Coût opportunité

Plutôt que continuer P7 (re-train continuations, variante C 4-sizings, etc.),
mieux vaut consolider V1+TA1 et investiguer pourquoi les bots LAG sont
proches de 0 (LAG = +7.20 bb/100 en V1 complet, marginal).

---

## Ce qui est préservé après revert

| Catégorie | Préservé ? |
|---|---|
| Code applicatif (engine, abstraction, ai, solver, training) | revert pre-p7 |
| TA1 fix (creer_agent active continuations + solveurs) | **✓ cherry-pick** |
| Test `test_agent_pluribus_complet.py` | **✓** |
| Rapports `docs/investigations/P7-*` (10 fichiers) | **✓ tous archivés** |
| Blueprints `data/strategy/blueprint_p7_*.pkl` (3 fichiers, 996 MB) | **✓ archivés** |
| Tags git `pre-p7`, `pre-p7-cloud-5m`, `p7-complete-archived` | **✓** |

**Pour reprendre P7 plus tard** :
```bash
git checkout p7-complete-archived
# Tout l'état P7 est restauré : code, helpers, tests RED, etc.
```

---

## Leçons apprises

1. **Toujours auditer l'architecture agent avant d'évaluer.** Un agent
   mutilé donne des chiffres trompeurs. Le rapport `architecture_audit.md`
   sera la référence pour les évals futures.

2. **Les "continuations Pluribus k=4" sont essentielles.** Elles apportent
   ~+22 bb/100 par activation, soit potentiellement plus que toutes les
   autres améliorations algorithmiques.

3. **Les solveurs real-time (depth-limited / subgame) sont coûteux** mais
   nécessaires en mode complet. Budget par défaut 0.3s/décision = ~5h pour
   30K mains/blueprint.

4. **Une réduction de cardinalité massive (÷55× hist) ne garantit PAS une
   meilleure performance.** P7 réduit la cardinalité mais perd contre les
   bots structurés. La granularité a un coût stratégique (sur-agrégation).

5. **Power-law saturation valide pour MCCFR** sur 3 ordres de grandeur
   (5K → 4M iter). Utile pour planifier les futures runs cloud.

6. **L'incident réseau VM GCP** (P7.7 batch 9/10 perdu) n'a pas affecté les
   conclusions : 4M iter offraient ~99% du blueprint final attendu à 5M.

---

## Pointeurs

- Spec : [`spec.md`](spec.md)
- Saturation : [`results.md`](results.md)
- Cloud pilot 500K : [`cloud_pilot_500k.md`](cloud_pilot_500k.md)
- Cloud 4M : [`cloud_5m.md`](cloud_5m.md)
- Eval P8 (agent mutilé) : [`p8_evaluation.md`](p8_evaluation.md)
- Diag DCFR off : [`p8_diag_no_deepcfr.md`](p8_diag_no_deepcfr.md)
- Diag critique 4 cellules : [`p8_diag_critique.md`](p8_diag_critique.md)
- **Architecture audit (le finding majeur)** : [`architecture_audit.md`](architecture_audit.md)
- Eval P9 Pluribus complet : [`p9_evaluation_pluribus.md`](p9_evaluation_pluribus.md)
- Bypasses TDD Guard : [`tdd_guard_bypasses.md`](tdd_guard_bypasses.md)
- **Revert final** : [`final_revert.md`](final_revert.md)
