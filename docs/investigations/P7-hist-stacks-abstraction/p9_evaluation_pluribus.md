# P9 — Évaluation Pluribus complet : V1 vs P7

**Date** : 2026-04-30
**Objectif** : mesurer V1 et P7 dans des conditions Pluribus complètes
(continuations + solveurs FLOP/SG actifs) après le fix TA1, pour décider
revert pre-p7 ou continuer P7.
**Mode** : étape 3a rapide, **1 seed (42)**, 5K mains/baseline, 6 baselines.

**STOP avant interprétation finale** comme demandé.

---

## ⚠️ ASYMÉTRIE STRUCTURELLE (à lire d'abord)

Le test P9 compare deux configurations qui ne sont **pas équivalentes** :

| Config | Blueprint | Continuations k=4 | Solveurs FLOP/SG | DCFR | OFT |
|---|---|---|---|---|---|
| V1 complet | `blueprint_v1.pkl` (2.58M infosets) | **4 variantes** (baseline + fold + call + raise) | ✓ | ✓ | ✓ |
| P7 complet | `blueprint_p7_4m_cloud.pkl` (2.99M) | **1 variante** (baseline seul, option 2b) | ✓ | ✓ | ✓ |

**Pourquoi cette asymétrie ?** Les variantes biaisées V1 (`blueprint_v1_call/fold/raise.pkl`)
existent depuis le 20-21 avril, entraînées dans le régime V1. Aucun équivalent
P7 n'a été produit. Selon spec P7 décision D5, le re-train continuations P7
était reporté. Le delta P9 inclut donc :

1. **Effet abstraction P7** (ce qu'on cherche à mesurer)
2. **Effet absence continuations P7** (artefact technique)

On ne peut pas isoler les deux sans re-train cloud P7 continuations (~50€).

Les 6 baselines, OFT, DCFR, solveurs sont identiques. **Seules les
continuations diffèrent en faveur de V1.**

### Note méthodologique seed unique

P9 utilise 1 seed (42) × 5K mains/baseline = 30K mains/blueprint.
Les rapports P8 (V1 et P7 mutilés) utilisent 3 seeds × 10K mains.

Variance inter-seed observée en P8 sur les bots structurés : faible
(~2-6 bb/100 stddev). Donc 1 seed × 5K reste interprétable pour les
deltas > 15 bb/100. Pour les deltas < 15 bb/100, on devrait relancer
seeds 123 et 2026 (étape 3b).

---

## 1. Tableau 4 colonnes (V1/P7 × mutilé/complet)

### Sources données

| Cellule | Fichier | Protocole |
|---|---|---|
| V1 mutilé | `data/strategy/p8_eval_v1_results.json` | **3 seeds × 10K**, baseline+DCFR+OFT |
| V1 complet | `data/strategy/p9_eval_v1_pluribus_results.json` | **1 seed (42) × 5K**, +continuations k=4 +solveurs |
| P7 mutilé | `data/strategy/p8_eval_p7_results.json` | 3 seeds × 10K, baseline+DCFR+OFT |
| P7 complet | `data/strategy/p9_eval_p7_pluribus_results.json` | 1 seed (42) × 5K, +solveurs (1 continuation seule) |

### Tableau

Métrique : winrate **bb/100**.

| Baseline | V1 mutilé | **V1 complet** | P7 mutilé | **P7 complet** |
|---|---:|---:|---:|---:|
| Aléatoire   | -73.34 | **-38.11** | -48.13 | **-34.48** |
| Call-Only   | +15.24 | **+89.10** | -14.64 | **+4.74** |
| Raise-Only  | -82.56 | **-73.57** | -52.38 | **-54.01** |
| TAG         | +12.71 | **+14.23** | -6.79  | **-11.52** |
| LAG         | -3.50  | **+7.20**  | -16.60 | **-12.57** |
| Régulier    | +9.54  | **+8.49**  | -8.84  | **-11.78** |
| **Moyenne** | **-20.32** | **+1.22** | **-24.56** | **-19.94** |

### Δ mutilé → complet (gain Pluribus complet)

| Baseline | Δ V1 (complet−mutilé) | Δ P7 (complet−mutilé) |
|---|---:|---:|
| Aléatoire   | **+35.23**  | +13.65 |
| Call-Only   | **+73.86**  | +19.38 |
| Raise-Only  | +8.99       | -1.63  |
| TAG         | +1.52       | -4.73  |
| LAG         | +10.70      | +4.03  |
| Régulier    | -1.05       | -2.94  |
| **Moy**     | **+21.54**  | **+4.62** |

**V1 gagne ~5x plus de l'activation Pluribus complet que P7.**
Logique mécanique : V1 a 4 continuations, P7 n'en a qu'une. La rotation
k=4 contribue clairement à V1.

### Δ V1 complet − P7 complet (LE chiffre central)

| Baseline | V1 complet | P7 complet | **Δ (V1 − P7)** |
|---|---:|---:|---:|
| Aléatoire   | -38.11 | -34.48 | **-3.63** (P7 marginal) |
| Call-Only   | +89.10 | +4.74  | **+84.36** (V1 énorme) |
| Raise-Only  | -73.57 | -54.01 | **-19.56** (P7 mieux) |
| TAG         | +14.23 | -11.52 | **+25.75** (V1 mieux) |
| LAG         | +7.20  | -12.57 | **+19.77** (V1 mieux) |
| Régulier    | +8.49  | -11.78 | **+20.27** (V1 mieux) |
| **Moyenne** | **+1.22** | **-19.94** | **+21.16** (V1 mieux global) |

---

## 2. Vérification critère décision

L'utilisateur a fixé : *"Si l'écart V1 vs P7 sur les baselines structurées
(TAG, LAG, Régulier) est >15 bb/100 : signal clair."*

| Baseline structurée | Δ V1 − P7 | > 15 bb/100 ? |
|---|---:|---|
| TAG       | +25.75 | ✓ |
| LAG       | +19.77 | ✓ |
| Régulier  | +20.27 | ✓ |

**Les 3 critères sont tous > 15 bb/100. Signal clair selon le critère défini.**

Pas besoin d'étape 3b (seeds 123 + 2026 supplémentaires).

---

## 3. Observations factuelles brutes

1. **V1 complet est globalement positif** (winrate moyen +1.22 bb/100) — c'est
   le premier régime testé qui passe au-dessus de zéro en moyenne. Les 4
   bots structurés/passifs (Call, TAG, LAG, Régulier) sont tous **positifs**.

2. **P7 complet reste globalement négatif** (-19.94 bb/100). Les 3 bots
   structurés (TAG, LAG, Régulier) sont tous **négatifs**. Seul Call-Only
   est marginalement positif (+4.74).

3. **L'effet "Pluribus complet" booste V1 ×21.54 bb/100 vs +4.62 bb/100 pour
   P7**. La majorité du gain V1 vient des continuations k=4 (Call-Only +73.86,
   Aléatoire +35.23 — bots où la rotation aléatoire des stratégies biaisées
   est la plus dévastatrice contre des règles statiques).

4. **P7 reste meilleur que V1 contre les bots mécaniques** :
   - Aléatoire : P7 -34.48 vs V1 -38.11
   - Raise-Only : P7 -54.01 vs V1 -73.57
   - L'avantage abstraction V2+P7 sur les bots mécaniques persiste.

5. **P7 reste pire que V1 contre les bots structurés**, et l'écart est
   significatif (>15 bb/100) sur TOUS les 3 bots GTO-like.

6. **Variance inter-baseline énorme** : Call-Only +89.10 vs Régulier +8.49
   en V1 complet. Le winrate moyen masque des dynamiques très différentes
   selon le profil adversaire.

---

## 4. Implications pour décision pre-p7 vs continuer P7

### Faits acquis

1. V1 + Pluribus complet **fonctionne** (+1.22 bb/100 en moyenne, positif sur
   4/6 baselines).
2. P7 + Pluribus complet **ne fonctionne pas** (-19.94 bb/100 en moyenne,
   positif sur 1/6 baselines).
3. L'écart V1−P7 sur bots structurés est ≥ 15 bb/100 sur les 3 (TAG, LAG,
   Régulier).
4. L'asymétrie continuations (V1 a k=4, P7 a k=1) explique partiellement
   mais pas totalement l'écart.

### Pistes possibles (à arbitrer ensemble — pas par moi)

**A — Revert pre-p7 (revenir à V1)**
- Pour : V1 a un track record positif en mode complet ; pre-p7 = état stable
- Contre : on perd les gains de cardinalité P7 (÷55× hist) et la propreté
  conceptuelle Spin & Rush
- Coût : `git reset --hard pre-p7` + sync axiom-public, 0€

**B — Re-train continuations P7 cloud (P10)**
- Pour : neutraliser l'asymétrie continuations ; permettrait une comparaison
  équitable
- Contre : ~50€ cloud (4 variantes × 4M iter), ETA ~30h cumul, et même si
  ça comble une partie du gap, l'écart sur Régulier (+20.27) suggère que
  l'abstraction P7 est aussi en cause
- Décision conditionnelle : faire seulement si l'utilisateur veut tenter
  P7 jusqu'au bout

**C — Tester variante C 4-sizings (P10 alternatif)**
- Recoder `_abstraire_sizing` avec 4 buckets (S/M/L_pot/L_over) pour
  préserver la distinction r3 (pot) vs r4 (overbet) — voir spec P7 §2
- Re-train cloud P7 baseline avec nouvelle abstraction
- Plus invasif, ~10€ cloud
- Si ça résout, l'écart sur structurés vient bien de la sur-agrégation
  sizing

**D — Hybride : revert pre-p7 + extraire les gains cardinalité P7**
- Garder le format clé V1 mais appliquer la cap=4 sur hist seulement
- Pas le formalisme Spin & Rush mais cardinalité ÷ N intermédiaire
- Demande re-implémentation propre

### Mon avis (à valider)

L'écart sur bots structurés est important mais l'asymétrie continuations est
réelle. **B** (re-train continuations P7) tranche définitivement, mais coûte
50€. **A** est sûr et économe mais abandonne ~6 mois de travail P7.

Question à clarifier avant la décision : l'objectif final est-il de battre
TAG/LAG/Régulier (~bots structurés) ou des humains réels sur Betclic ?
Les humains Spin & Rush sont plus proches de mécaniques + tilt que de bots
GTO-like. Si oui, P7 pourrait avoir un avantage non capturé par les
baselines actuelles.

---

## 5. État technique

| Item | État |
|---|---|
| Branche actuelle | `main` |
| Tests `pytest tests/ -q` | 208 GREEN (pré-vérifié post TA1) |
| Blueprints intacts (md5) | ✓ V1, P7-4M identiques au backup |
| Backup `/tmp/axiom_blueprints_safe/` | ✓ |
| Stash main restored | ✓ |
| Résultats sauvegardés | `data/strategy/p9_eval_v1_pluribus_results.json`, `p9_eval_p7_pluribus_results.json` |

---

## 6. STOP

Tableau 4 colonnes livré, deltas calculés, asymétrie documentée en tête.
Critère >15 bb/100 sur structurés franchi sur les 3 baselines.

**À toi pour décider** entre options A / B / C / D ci-dessus, ou autre.
Pas d'interprétation finale ni de recommandation imposée par moi.
