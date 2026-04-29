# P7.6 — Pilot cloud 500K iter (validation projection power-law)

**Date** : 2026-04-29
**VM** : `axiom-training-24` (n2-standard-32, europe-west4-a, 32 vCPU, 128 GB RAM)
**Run** : 500K iter MCCFR avec stratégie multiprocessing P6.G.fix
**Config** : 5 batches × 100K iter, 20 workers parallèles × 5K iter chacun
**seed** : 42 (cohérence avec mesures locales P7.5)
**Objectif** : valider la projection power-law (α=0.667) de P7.5 sur données réelles
avant toute décision sur le pilot 5M.

---

## 1. Résultats — projection vs réalité

| Métrique | Prédit (power-law P7.5) | Mesuré cloud | Verdict |
|---|---|---|---|
| Infosets totaux | ~3.0 M | **2 263 447** | ✅ mieux que prédit (-25%) |
| Ratio infosets/iter | ~6.0 | **4.53** | ✅ ÷1.32 mieux |
| Cardinalité hist | 417 (plafond P7.5) | **417** | ✅ identique (plafond confirmé) |
| Cardinalité stacks | 229 (plafond P7.5) | **229** | ✅ identique |
| Blueprint size | ~480 MB | **366 MB** | ✅ -24% |

**Le plafond hist=417 / stacks=229 mesuré localement est identique au cloud.**
Confirme que ces dimensions sont structurellement bornées par l'abstraction P7
(ni le multiprocessing ni l'échelle ne changent leur cardinalité).

**Le ratio est 25% meilleur que prédit** — explication la plus probable :
la fusion multi-worker dédupplique certains infosets co-visités par plusieurs
workers en parallèle (chaque worker visite des deals différents mais partage
des infosets clés communs). Ce gain n'est pas capturable en mono-process local.

---

## 2. Performance run

| Métrique | Valeur |
|---|---|
| Durée totale | **49.6 min** (0.83h) |
| Throughput moyen | **167 it/s** (cumul) |
| Vitesse par batch | ~9.93 min/batch |
| Workers | 20 parallèles |
| Coût VM (1.55 €/h × 0.83h) | **~1.30 €** |
| Coût SCP + setup overhead | ~0.50 € |
| **Coût total estimé** | **~1.80 €** |

Throughput **33× supérieur** au local (5 it/s mono-process Windows).
Multiprocessing P6.G.fix scale comme attendu sur 32-core Linux.

---

## 3. Réajustement projection power-law

### Calcul α réel sur 6 points (5 locaux + 1 cloud)

| N | ratio mesuré | source |
|---|---|---|
| 5 000   | 129.44 | local |
| 10 000  | 88.97  | local |
| 25 000  | 49.21  | local |
| 50 000  | 29.87  | local |
| 100 000 | 17.55  | local |
| **500 000** | **4.53** | **cloud** |

Régression power-law sur 6 points : la pente α augmente légèrement à mesure
que N croît :
- α(5K→100K) = 0.667 (P7.5)
- α(100K→500K) = log(17.55/4.53) / log(500K/100K) = log(3.874)/log(5) = 0.842

**La saturation s'accélère** au-delà de 100K — saturation plus aggressive
qu'extrapolée. Bonne nouvelle pour le pilot 5M.

### Réajustement extrapolations 5M

Avec la nouvelle pente locale α=0.842 sur le segment ≥ 100K :

```
ratio(5M) ≈ ratio(500K) × (500K/5M)^0.842
         = 4.53 × (0.1)^0.842
         = 4.53 × 0.144
         = 0.65
```

| N | ratio (P7.5 α=0.667) | ratio (P7.6 α=0.842 sur ≥100K) | Préférence |
|---|---|---|---|
| 1 M  | 3.8 | 2.5 | ratio P7.6 plus serré |
| **5 M** | **1.3** | **0.65** | ratio P7.6 ≈ ÷2 |
| 20 M | 0.5 | 0.21 | — |

**Estimation 5M cloud** :
- Infosets ≈ 5M × 0.65 = **~3.25 M** (vs estimation P7.5 ~6.5 M)
- Blueprint ≈ **~520 MB** (vs P7.5 ~1 GB)
- Durée ≈ 5M / (167 it/s) = ~8.3 h (single-batch) ou ~6h en multi-batch optimisé
- Coût ≈ 6h × 1.55€/h = **~10 €** (dans le budget initial 37€)

---

## 4. Validation algorithme

| Check | Résultat |
|---|---|
| Centroïdes V2 chargés | ✅ V2 OK |
| Helpers P7 actifs | ✅ `_format_hist_avec_cap` cap=4, `PALIERS_STACK_SPIN_RUSH` |
| Format clé 7 segments | ✅ (vérifié post-pickle) |
| 5 checkpoints batch | ✅ tous présents (`blueprint_p7_pilot_500k_cloud_batch_{1..5}.pkl`) |
| Pas d'erreur worker | ✅ aucun ERREURS dans le log |
| Blueprint final loadable | ✅ `pickle.load()` OK, dict normal |

---

## 5. Mesure pré-cloud vs post-cloud (même blueprint)

Le blueprint `data/strategy/blueprint_p7_pilot_500k_cloud.pkl` (366 MB local)
contient **2 263 447 infosets** identiques à la mesure VM. Aucune perte au transfert.

---

## 6. Décision

**Pilot 500K validé.** Tous les critères P7.6 OK :
- Ratio mesuré (4.53) **inférieur** au prédit (6.0) → saturation plus rapide ✓
- Cardinalité hist (417) et stacks (229) **identiques** local vs cloud ✓
- Throughput cloud (167 it/s) **conforme** aux attentes multiproc ✓
- Coût (1.80€) **largement sous** le budget 3-5€ ✓

**Recalibrage attentes pilot 5M** :
- ETA ~6-8h cloud
- Coût attendu ~10€ (vs 37€ budgété — économie 27€)
- Blueprint final ~520 MB (vs 1 GB anticipé)

**Risque résiduel** : la pente α=0.842 mesurée sur le segment 100K→500K
pourrait s'incliner à nouveau différemment au-delà de 500K. Le pilot 5M
confirmera ou infirmera. Aucun signal négatif observé.

**Prochaine étape** : P7.7 — pilot 5M cloud (~10€, ETA 6-8h).

**STOP** avant 5M — attente GO explicite après lecture de ce rapport.
