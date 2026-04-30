# P8 — Évaluation blueprint P7 (4M) vs blueprint V1 — chiffres bruts

**Date** : 2026-04-30
**Mains/seed/baseline** : 10 000 × 6 baselines × 3 seeds (42, 123, 2026) = 180 000 mains/blueprint
**Format** : 3-max NLHE, blindes 10/20, OFT activé
**Branches utilisées** :
- V1 : `git checkout p6-pre-bascule` (commit `2331436`)
- P7 : `git checkout main` (HEAD `75de903`)

**STOP avant interprétation finale** — chiffres bruts, on interprète ensemble.

---

## ⚠️ LIMITES DE LA COMPARAISON (à lire AVANT les chiffres)

**La comparaison V1 vs P7 N'EST PAS ÉQUITABLE.** Le delta mesuré inclut des
facteurs qu'on ne peut pas isoler sans re-train contrôlé. Listées ici en tête
pour transparence méthodologique :

### Confound #1 — Réseaux Deep CFR pollués pour P7 (le plus grave)

Les deux évals chargent les **mêmes** réseaux `data/models/strategy_net_j*.pt`
et `data/models/valeur_net_j*.pt`. Ces réseaux ont été entraînés à l'époque
**V1** (raw `r1..r4`, V1 buckets, ancien `PALIERS_STACK`).

- **V1 eval** : code utilise V1 abstraction → inputs réseau **cohérents** avec
  format de training → outputs réseau utilisables.
- **P7 eval** : code utilise V2 abstraction + sizing S/M/L + paliers Spin & Rush
  → inputs réseau dans un format **différent** de celui sur lequel les réseaux
  ont été entraînés → outputs réseau **probablement biaisés / garbage**.

L'agent `AgentAXIOM` blend blueprint + Deep CFR + heuristique. Si Deep CFR
sort des prédictions corrompues, l'agent prend de mauvaises décisions, **même
si le blueprint P7 est meilleur que V1**. Cette pollution explique
**probablement une part majeure** de la sous-performance P7 observée.

**Pour isoler P7 proprement, il faudrait re-train les réseaux Deep CFR sur
l'abstraction P7.** Hors scope P8.

### Confound #2 — Conditions d'entraînement différentes

| | V1 | P7 |
|---|---|---|
| Date training | 21 avril | 30 avril |
| Iterations | inconnu (probablement multi-millions) | 4M (8 batches × 500K) |
| Multiprocessing | inconnu | 20 workers cloud n2-standard-32 |
| OFT actif training | inconnu | non |
| Variantes Continuation Strategies (Pluribus) | oui (`blueprint_v1_call/fold/raise.pkl`) — pas utilisées en P8 | non |

Le delta inclut les effets de ces différences en plus du seul effet
abstraction P7.

### Confound #3 — Taille blueprint

- V1 : 2 584 465 infosets
- P7 : 2 988 394 infosets (+15.6 %)

P7 a plus d'infosets distincts. Pas équivalent de "même taille = même qualité"
(plus d'infosets = couverture potentiellement plus fine, mais aussi plus de
nœuds peu visités donc moins bien convergés).

### Confound #4 — Format clé incompatible cross-régime

V1 = (V1 buckets, ancien PALIERS_STACK, hist raw) — testé sous code V1.
P7 = (V2 buckets, PALIERS_STACK_SPIN_RUSH, hist abstrait S/M/L cap=4) — testé sous code P7.
On compare **deux paires (blueprint, code)**, pas deux blueprints isolés.

---

## 1. Tableau comparatif moyennes (3 seeds × 10K mains × 6 bots)

Métrique : winrate **bb/100** (positif = AXIOM gagne, négatif = AXIOM perd).

| Baseline | V1 moy | V1 std | P7 moy | P7 std | **Δ (P7 − V1)** | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Aléatoire   | -73.34 | 1.23 | -48.13 | 1.13 | **+25.21** | P7 mieux |
| Call-Only   | +15.24 | 6.18 | -14.64 | 3.71 | **-29.88** | V1 mieux |
| Raise-Only  | -82.56 | 1.51 | -52.38 | 0.62 | **+30.18** | P7 mieux |
| TAG         | +12.71 | 2.31 | -6.79  | 1.55 | **-19.50** | V1 mieux |
| LAG         | -3.50  | 0.81 | -16.60 | 1.04 | **-13.10** | V1 mieux |
| Régulier    | +9.54  | 2.62 | -8.84  | 1.06 | **-18.38** | V1 mieux |
| **Moyenne** | **-20.32** | — | **-24.56** | — | **-4.24** | légèrement mieux V1 (en moyenne) |

### Distribution Δ par baseline

- Δ favorable P7 : **2 baselines** (Aléatoire, Raise-Only) — bots mécaniques
- Δ favorable V1 : **4 baselines** (Call-Only, TAG, LAG, Régulier) — bots structurés/GTO-like
- **P7 ne dégrade qu'une seule baseline en absolu vers le négatif global** : V1 a 3 baselines positives (Call, TAG, Régulier), P7 en a 0.

### Variance inter-seed

Très faible des deux côtés (stddev 0.6-6.2 bb/100 sur 10K mains/seed). Stats
suffisamment robustes pour conclure sur les signes de delta.

---

## 2. Tableau brut V1 (3 seeds)

| Seed | Aléatoire | Call-Only | Raise-Only | TAG | LAG | Régulier | Durée |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42   | -74.82 | +17.77 | -83.85 | +15.28 | -4.29 | +10.61 | 906s |
| 123  | -71.80 | +6.55  | -80.37 | +9.68  | -3.90 | +11.96 | 836s |
| 2026 | -73.39 | +21.40 | -83.47 | +13.16 | -2.31 | +6.06  | 946s |
| **Moy** | **-73.34** | **+15.24** | **-82.56** | **+12.71** | **-3.50** | **+9.54** | — |

Source : `data/strategy/p8_eval_v1_results.json`

---

## 3. Tableau brut P7 (3 seeds)

| Seed | Aléatoire | Call-Only | Raise-Only | TAG | LAG | Régulier | Durée |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42   | -48.11 | -14.55 | -51.62 | -8.97 | -17.87 | -8.45  | 617s |
| 123  | -49.53 | -19.18 | -53.13 | -5.75 | -16.58 | -10.24 | 598s |
| 2026 | -46.76 | -10.18 | -52.39 | -5.65 | -15.35 | -7.84  | 601s |
| **Moy** | **-48.13** | **-14.64** | **-52.38** | **-6.79** | **-16.60** | **-8.84** | — |

Source : `data/strategy/p8_eval_p7_results.json`

---

## 4. Observations honnêtes (sans interprétation)

1. **P7 améliore les bots mécaniques** (Aléatoire +25, Raise-Only +30). Ces bots
   ne calibrent pas leur stratégie et l'EV exploitable est élevée — le gain
   peut venir d'une meilleure couverture postflop V2 ou simplement d'un
   ajustement OFT plus efficace dans le nouveau format.

2. **P7 dégrade les bots structurés** (Call-Only -30, TAG -20, Régulier -18,
   LAG -13). Tous les bots qui adoptent une stratégie cohérente font perdre
   plus à P7. Hypothèse : le blueprint P7 + réseau Deep CFR pollué (Confound #1)
   produit des décisions stratégiquement faibles contre des adversaires qui
   capitalisent sur les leaks.

3. **Toutes les baselines sont négatives en P7**. L'agent perd contre TOUS les
   adversaires. C'est anormal — V1 avait 3 baselines positives.

4. **P7 plus rapide** (~600s vs ~900s par seed) malgré dict plus gros : possible
   que les hist abstraits réduisent les hash collisions ou que cap=4 limite la
   complexité de lookup.

5. **Variance inter-seed très basse** des deux côtés → résultats reproductibles,
   pas du bruit.

---

## 5. État technique post-P8

| Item | État |
|---|---|
| Branche actuelle | `main` |
| Tests `pytest tests/ -q` | 207 passed ✓ |
| Blueprints intacts (md5) | ✓ V1, P7-4M, P7-pilot-500K identiques au backup |
| Backup `/tmp/axiom_blueprints_safe/` | ✓ conservé |
| Résultats sauvegardés | `data/strategy/p8_eval_v1_results.json`, `p8_eval_p7_results.json` |

---

## 6. Décision STOP avant interprétation

Comme demandé : pas d'interprétation finale ici. Les chiffres bruts +
les confounds sont posés. À discuter ensemble :

- Quelle part du delta négatif vient de la **pollution réseau Deep CFR** vs
  de la qualité intrinsèque du blueprint P7 ?
- Faut-il **re-train les réseaux Deep CFR sous abstraction P7** pour une
  comparaison équitable (P9.x) ?
- Ou bien **re-train un blueprint pré-P7 avec les conditions cloud
  multiprocessing actuelles** pour neutraliser confound #2 ?
- Le pattern "P7 bat les mécaniques, V1 bat les structurés" est-il un
  signal sur la qualité de la nouvelle abstraction, ou un artefact des
  confounds ?

À toi.
