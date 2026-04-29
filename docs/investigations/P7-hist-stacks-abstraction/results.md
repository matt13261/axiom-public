# P7.5 — Mesure saturation MCCFR (10K → 100K iter local)

**Date** : 2026-04-28 / 2026-04-29
**Contexte** : valider en local que l'abstraction P7 (cap=4 hist, stacks 7
niveaux Spin & Rush) atteint le ratio infosets/iter cible (< 60) en régime
asymptotique avant tout cloud train. Mesure menée seed=42 fixe sur 5 paliers
cumulatifs (mémoire seule, blueprint non sauvegardé).

---

## 1. Tableau complet des paliers

| N | infosets | ratio | hist | stacks | elapsed | Δratio (vs N préc.) |
|---|---|---|---|---|---|---|
| 5 000   | 647 197   | 129.44 | 417 | 219 | 9 min  | — |
| 10 000  | 889 705   | 88.97  | 417 | 223 | 20 min | ÷1.45 |
| 25 000  | 1 230 260 | 49.21  | 417 | 225 | 56 min | ÷1.81 |
| 50 000  | 1 493 373 | 29.87  | 417 | 228 | 2 h 07 | ÷1.65 |
| **100 000** | **1 755 100** | **17.55** | **417** | **229** | **4 h 54** | **÷1.70** |

Vitesse moyenne : 5.0–8.9 it/s (décroît avec la taille du dict).
Hist & stacks plafonnent dès 5K iter (saturation des dimensions abstraites).

---

## 2. Régression power-law

Modèle ajusté sur les 5 paliers :

```
ratio(N) = 129.44 × (5000 / N)^0.667
```

| N | ratio mesuré | ratio prédit | écart |
|---|---|---|---|
| 5K   | 129.44 | 129.44 | 0%   |
| 10K  | 88.97  | 81.51  | -8.4% |
| 25K  | 49.21  | 44.09  | -10.4% |
| 50K  | 29.87  | 27.77  | -7.0% |
| 100K | 17.55  | 17.49  | -0.3% |

α = **0.667** (cohérent avec l'asymptotique standard MCCFR : la croissance des
infosets visités est sous-linéaire en N à mesure que l'espace explore sature).

R² visuel élevé sur 2 ordres de grandeur (5K → 100K). Pente stable.

---

## 3. Extrapolations cloud

Application directe de la formule à des N supérieurs :

| N | ratio prédit | infosets totaux estimés | Taille blueprint (~160 B/infoset) |
|---|---|---|---|
| 500 000   | ~6.0  | 3.0 M  | ~480 MB |
| 1 000 000 | ~3.8  | 3.8 M  | ~610 MB |
| **5 000 000** | **~1.3** | **~6.5 M** | **~1.0 GB** |
| 20 000 000 | ~0.5 | ~10 M  | ~1.6 GB |

**Implication** : un pilot cloud 5M iter produirait un blueprint ≈ 1 GB,
parfaitement gérable en RAM sur la VM `axiom-training-24` (n2-standard-32,
128 GB) avec la stratégie batch+fusion P6.G.fix.

---

## 4. Distribution longueur hist (invariante 10K → 100K)

| Longueur (actions) | Hists distinctes | % | @ 10K | @ 100K |
|---|---|---|---|---|
| 0 | 1   | 0.2%  | 1   | 1   |
| 1 | 7   | 1.7%  | 7   | 7   |
| 2 | 29  | 7.0%  | 29  | 29  |
| 3 | 89  | 21.3% | 89  | 89  |
| **4** | **291** | **69.8%** | **291** | **291** |

**Total : 417 hists** (identique aux 2 mesures). Distribution **strictement
invariante** entre 10K et 100K iter → cap=4 atteint son régime stable dès 10K
itérations. Aucune nouvelle variante de hist n'apparaît au-delà — tout
l'espace possible des hist abstraits est exploré dès 10K.

---

## 5. Évolution top 10 hists (aplatissement avec N)

### @ 10K

| # | hist | % du volume |
|---|---|---|
| 1 | rMrLrLrL | 2.26% |
| 2 | rLrLrLrL | 2.20% |
| 3 | rMrMrLrL | 2.14% |
| 4 | rMrM     | 1.81% |
| 5 | rM       | 1.67% |
| 6 | rMrMrMrL | 1.61% |
| 7 | xrMrM    | 1.51% |
| 8 | xrM      | 1.49% |
| 9 | rSrM     | 1.47% |
| 10 | rLrMrLrL | 1.42% |

Top 10 cumulé ≈ **17.6%**.

### @ 100K

| # | hist | % du volume |
|---|---|---|
| 1 | rMrLrLrL | 1.88% |
| 2 | rLrLrLrL | 1.81% |
| 3 | rMrMrLrL | 1.70% |
| 4 | rLrMrLrL | 1.51% |
| 5 | rLrLrMrL | 1.43% |
| 6 | rMrMrMrL | 1.40% |
| 7 | rSrLrLrL | 1.34% |
| 8 | rSrMrLrL | 1.26% |
| 9 | rLrMrMrL | 1.20% |
| 10 | rMrM    | 1.19% |

Top 10 cumulé ≈ **14.7%**.

**Observation** : à 100K, les hists courts (`rM`, `xrM`, `rSrM`) descendent
hors du top 10 au profit des séquences 4 actions complètes. La distribution
**s'aplatit** (top 1 passe de 2.26% à 1.88%) — comportement attendu d'un MCCFR
mature qui visite plus uniformément l'espace des hists complexes.

Aucune nouvelle hist n'apparaît, mais leur poids relatif évolue.

---

## 6. Critères P7.5 — ré-évaluation

| Critère | Cible | À 10K | À 50K | À 100K | Statut |
|---|---|---|---|---|---|
| Ratio infosets/iter | < 60 | 89.05 ❌ | 29.87 ✅ | **17.55 ✅** | **OK en régime stable (≥ 50K)** |
| Cardinalité hist | < 800 | 417 ✅ | 417 ✅ | 417 ✅ | OK |
| Cardinalité stacks | < 300 | 222 ✅ | 228 ✅ | 229 ✅ | OK |
| Kuhn convergence | < 0.005 | 0.002 ✅ | (inchangé) | (inchangé) | OK |

**Verdict** : à 10K iter le critère #1 était KO (89), mais la mesure à 50K
(29.87) puis 100K (17.55) confirme que **le ratio plonge sous le seuil 60 dès
50K iter**. Les critères P7.5 sont **tous validés en régime stable**.

---

## 7. Limites et risques résiduels

**Méthodologie** :
- Extrapolation power-law sur **5 points** couvrant **2 ordres de grandeur**
  (5K → 100K). Bonne base mais pas de garantie au-delà.
- L'extrapolation à 5M iter (50× au-delà du dernier point mesuré) repose sur
  l'hypothèse que la pente α=0.667 reste constante. **Risque** : la pente
  pourrait s'incliner différemment au-delà de 100K (effet de bord rare,
  changement de régime).
- Aucune mesure intermédiaire 200K-500K — coût local prohibitif (estimé
  10-15h supplémentaires sans garantie informationnelle proportionnelle).

**Risques techniques** :
- Le ralentissement linéaire it/s (8.9 → 5.0 entre 5K et 100K) suggère que
  la performance dégrade avec la taille du dict. À 5M iter, le throughput
  pourrait descendre à ~1-2 it/s, allongeant fortement le temps de cloud
  training. Compensable par multiprocessing P6.G.fix (32 cores VM).
- Cap=4 a été choisi empiriquement (mesure post-implémentation 417 hists).
  Si le re-train cloud produit un blueprint qui sous-performe contre certaines
  baselines (cf. risque flop multi-way 3-bet documenté dans spec §11), il
  faudra revoir le cap ou réintroduire le flag agresseur préflop (D5).

**Mitigation** :
- Le **pilot cloud 500K iter** (P7.6, ~3-4€) servira à **valider la
  projection sur données réelles**. À ce point, ratio attendu ~6.0,
  ~3M infosets, blueprint ~480 MB. Coût modeste, signal fort.
- Tag `pre-p7` (commit `edefa7f`) permet revert immédiat si problème
  structurel découvert post-cloud.

---

## 8. Décision

**P7 est validé pour aller en cloud.**

Justifications convergentes :
1. Ratio infosets/iter en régime stable < 60 dès 50K iter (mesure directe).
2. Extrapolation power-law cohérente sur 5 points → 5M iter ≈ 1 GB blueprint.
3. Cardinalité hist + stacks plafonnée (417 / 229) — aucune explosion résiduelle.
4. Convergence Kuhn préservée (canary algo).
5. Tag de revert en place + spec documentée pour itération si nécessaire.

**Prochaine étape : P7.6 — pilot cloud 500K iter** (validation projection,
~3-4€ budget, ETA ~2-3h cloud) avant tout pilot 5M.

STOP avant cloud — attente GO explicite pour P7.6.
