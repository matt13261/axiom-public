# Brainstorm P6 — Méthode bucketing postflop 8 → 50 buckets
_Session 2026-04-26 — 3-max NLHE Spin & Go 15€_

---

## Contexte

**Méthode actuelle :**
- MC 100 simulations, `seed(42)` → équité brute déterministe
- `bisect_right` sur 7 seuils → 8 buckets calibrés par percentiles
- Calibration : 4000 situations × 500 MC = 2M sims
- **Défaut fondamental** : flush draw (FD) avec 35% d'équité et paire faible
  avec 35% d'équité → même bucket → stratégie identique → fuite EV massive

**Opportunité** :
- `DIM_CARD = 4` dans `network.py` = scalaires de texture (non one-hot)
- Changer NB_BUCKETS ne casse PAS `DIM_INPUT = 52`
- `AbstractionCartes` est instanciée localement → migration progressive possible

**Objectif** : 50 buckets postflop (×6.25 granularité), qualité stratégique maximale,
budget 50€ cloud, 1-2 semaines solo.

---

## Méthode A — E[HS²] (Hand Strength Squared)

### Principe
HS = rang normalisé de la main parmi toutes les combinaisons possibles (0→1).
E[HS²] = espérance de HS² sur les runouts futurs, calculée par Monte Carlo.

La magie : E[HS²] ≈ E[HS]² + Var(HS). Une main avec **potentiel** (draw)
a une variance élevée → E[HS²] > E[HS]². Une paire faible "statique" a Var ≈ 0.

Ainsi le **vecteur de features est (E[HS], E[HS²])** → 2 dimensions → bucketing
2D, ou bucketing 1D sur E[HS²] seul, ou combiné `alpha*E[HS] + (1-alpha)*E[HS²]`.

### Formule
```
HS(main, board) = rang(main) / total_combinaisons_adverses  # 0..1
E[HS](main, board) = mean(HS sur N runouts aléatoires)
E[HS²](main, board) = mean(HS² sur N runouts aléatoires)
Potentiel = E[HS²] - E[HS]²  # variance = potentiel de la main
```

### Exemple concret
| Main | Board (flop) | E[HS] | E[HS²] | Potentiel | Bucket actuel |
|------|-------------|-------|--------|-----------|---------------|
| J♠T♠ | Q♥8♣3♦ (OESD) | 0.35 | 0.22 | 0.10 | bucket 4 |
| 6♥6♦ | Q♥8♣3♦ (PP) | 0.35 | 0.14 | 0.03 | bucket 4 ← MÊME |
| A♦K♦ | Q♥8♣3♦ (FD+overs) | 0.38 | 0.26 | 0.12 | bucket 4 ← MÊME |

Avec E[HS²] : ces 3 mains seraient dans des buckets différents ✅

### Coût compute
- **Calibration** : 10 000 situations × 500 simulations MC = 5M sims
  → CPU local : ~3-4h sur 8 cœurs
  → Pas besoin de cloud pour la calibration seule
- **Inference** : même coût que méthode actuelle (100 MC) → ~0.5ms
  Si on précalcule E[HS²] avec plus de sims (500) pour calibration
  et 100 pour inference : acceptable

### Qualité stratégique
| Critère | Évaluation |
|---------|------------|
| FD vs paire faible | ✅ Distingués (potentiel élevé vs nul) |
| Value vs bluff catcher | ✅ Partiellement (HS élevé = value, HS bas + potentiel = bluff) |
| Ranges adverses | ❌ Ignorées (pas de modèle adversaire) |
| Monotonie buckets | ✅ Buckets 2D cohérents |

### Complexité d'implémentation
- **Lignes** : ~120 nouvelles lignes dans `card_abstraction.py`
- **Dépendances** : numpy uniquement (déjà présent)
- **Modification** : remplacer `calculer_equite` par `calculer_hs_carre`
  dans `engine/hand_evaluator.py` ou nouvelle fonction
- **Risques** : coordonner (E[HS], E[HS²]) en 2D → k-means 2D pour trouver
  50 centroides → ajouter sklearn MAIS seulement pour calibration

### Compatibilité
- OFT : ✅ Sans impact (tracker opère sur actions)
- Migration progressive : ✅ (`AbstractionCartesV2` en parallèle)
- Checkpoints Deep CFR : ✅ (`DIM_CARD=4` inchangé, bucket = scalaire infoset uniquement)

### Scores
| Axe | Score /10 |
|-----|-----------|
| Qualité stratégique | **8/10** |
| Coût compute | **9/10** (CPU local suffisant) |
| Faisabilité 1-2 semaines | **8/10** |
| Robustesse production | **8/10** |
| **TOTAL** | **33/40** |

---

## Méthode B — OCHS (Opponent Cluster Hand Strength)

### Principe
Au lieu de calculer l'équité vs adversaires **aléatoires**, on calcule l'équité
vs K **clusters** de ranges adverses prédéfinis.

Pour chaque main, on obtient un vecteur de K équités : `[eq_vs_cluster_1, ..., eq_vs_cluster_K]`.
On fait ensuite un k-means sur ces vecteurs K-dimensionnels → 50 buckets.

Les ranges adverses typiques en Spin 15€ 3-max :
- Cluster 1 : top 10% des mains (AA-TT, AK, AQs)
- Cluster 2 : top 10-25% (paires moyennes, broadways)
- Cluster 3 : top 25-50% (suited connectors, Ax)
- Cluster 4 : top 50-100% (n'importe quoi)

### Formule
```
ranges_adverses = [top_10pct, top_25pct, top_50pct, any]  # K=4
pour chaque (main, board):
    features[main] = [MC_equity(main, board, range_k) for k in 1..K]
clusters = kmeans(features, n=50)
```

### Coût compute
- **Calibration** : 10 000 situations × K ranges × 300 MC = **12M sims**
  → Cloud nécessaire : ~2h sur 96 cœurs = ~2€
- **Inference** : K lookups MC (K×100 sims) → ~2-3ms par bucket
  ⚠️ 2-3× plus lent que la méthode actuelle pour l'inference live
- **Précalcul possible** : table de lookup pour mains connues, MC pour
  situations inédites

### Qualité stratégique
| Critère | Évaluation |
|---------|------------|
| FD vs paire faible | ✅ Distingués (FD bat les ranges faibles, PP bat les ranges moyen) |
| Value vs bluff catcher | ✅ Excellent (un bluff catcher bat les bluffs = cluster faible, perd vs value) |
| Ranges adverses | ✅ Prise en compte directe |
| Monotonie buckets | ⚠️ Pas garantie (vecteur K-dim, non ordonné naturellement) |

### Complexité d'implémentation
- **Lignes** : ~200 nouvelles lignes + définition des 4 ranges adverses
- **Dépendances** : sklearn (kmeans) pour calibration, déjà probable
- **Risques** :
  - Définir les K ranges adverses est **subjectif** → mauvais ranges = mauvaise abstraction
  - En 3-max, les ranges adverses sont hétérogènes et position-dépendantes
  - K-means sur vecteur 4D peut être instable (initialisation sensible)
  - Inference 3× plus lente : problème si latence < 3s requise en live

### Compatibilité
- OFT : ✅ (les ranges OCHS sont fixes à la calibration, pas liés à OFT)
- Migration progressive : ✅
- Checkpoints Deep CFR : ✅

### Scores
| Axe | Score /10 |
|-----|-----------|
| Qualité stratégique | **9/10** |
| Coût compute | **5/10** (cloud nécessaire, inference 3× plus lente) |
| Faisabilité 1-2 semaines | **5/10** (définir les ranges + k-means + validation) |
| Robustesse production | **6/10** (sensible aux ranges choisies) |
| **TOTAL** | **25/40** |

---

## Méthode C — K-means sur features composites

### Principe
Calculer un vecteur de N features pour chaque situation (main, board), puis
appliquer un k-means (n=50) pour trouver les 50 centroides.

Features candidates :
```
f1 = E[HS]        # équité brute Monte Carlo (déjà calculée)
f2 = E[HS²]       # variance d'équité (potentiel)
f3 = nb_outs      # nombre d'outs estimés (draws)
f4 = made_hand    # indicateur main faite (paire+, 0/1)
f5 = nut_potential # proximité des nuts (flush/straight potentiels)
f6 = street       # flop=0.33, turn=0.66, river=1.0
```

Le k-means génère 50 centroides dans R⁶. L'inference = chercher le centroïde
le plus proche du vecteur de la main actuelle.

### Formule
```
features(main, board) = [HS, HS², outs, made, nut_pot, street_norm]
centroides = kmeans(features_train_10000, n_clusters=50, seed=42)
bucket(main, board) = argmin(||features(main,board) - centroides||)
```

### Coût compute
- **Calibration** : 10 000 situations × 500 MC = 5M sims + k-means ~1min
  → CPU local : 3-4h, pas de cloud nécessaire
- **Inference** : calculer 6 features (~2ms MC) + `argmin` sur 50 centroides
  (~0.01ms) → total ~2ms par bucket
- **Stockage** : 50 centroides × 6 features = 300 floats = négligeable

### Qualité stratégique
| Critère | Évaluation |
|---------|------------|
| FD vs paire faible | ✅ Distingués (f2, f3, f4 séparent clairement) |
| Value vs bluff catcher | ✅ f1 (HS) + f5 (nut_pot) séparent bien |
| Ranges adverses | ❌ Ignorées (features intrinsèques seulement) |
| Monotonie buckets | ✅ Les clusters sont naturellement ordonnables par f1 |

### Complexité d'implémentation
- **Lignes** : ~180 lignes (features + k-means calibration + inference)
- **Dépendances** : sklearn (kmeans), numpy
- **Risques** :
  - Calculer `outs` proprement est subtil (flush draws, straight draws, overcards)
  - k-means sensible à l'initialisation (→ utiliser `n_init=10`, `random_state=42`)
  - Features non normalisées → biais du k-means vers les grandes valeurs
  - 6 features = curse of dimensionality modéré (50 clusters sur 6D est OK)

### Compatibilité
- OFT : ✅
- Migration progressive : ✅
- Checkpoints Deep CFR : ✅

### Scores
| Axe | Score /10 |
|-----|-----------|
| Qualité stratégique | **9/10** |
| Coût compute | **8/10** (CPU local suffisant, inference ~2ms) |
| Faisabilité 1-2 semaines | **7/10** (features à implémenter, mais modulaire) |
| Robustesse production | **8/10** (k-means stable avec seed fixe) |
| **TOTAL** | **32/40** |

---

## Méthode D — Distribution Histogram Bucketing

### Principe
Au lieu de résumer la distribution d'équité par sa moyenne (E[HS]) ou variance
(E[HS²]), on encode la **distribution entière** sous forme d'histogramme à M bins.

Pour chaque (main, board), lancer N MC et construire un histogramme des équités
obtenues → vecteur de M bins normalisé (somme=1). Puis k-means sur ces
vecteurs M-dimensionnels → 50 buckets.

Intuition : une main "bimodale" (soit gagne fort soit perd tout = draw) a un
histogramme en U, une main "unimodale" (paire solide) a un pic centré.

### Formule
```
sims = [HS(main, board, runout_i) for i in 1..N]  # N=200
histogram = np.histogram(sims, bins=10, range=(0,1))[0] / N  # vecteur 10D
bucket = argmin(||histogram - centroides||)
```

### Coût compute
- **Calibration** : 10 000 situations × 200 MC = 2M sims + k-means sur 10D
  → CPU local : ~1-2h
- **Inference** : 200 MC par bucket → **2× plus lent** que méthode actuelle
  à précision comparable
- **Précalcul** : impossible en live (board change à chaque main)

### Qualité stratégique
| Critère | Évaluation |
|---------|------------|
| FD vs paire faible | ✅ Excellent (distribution bimodale vs unimodale) |
| Value vs bluff catcher | ✅ Très bon |
| Ranges adverses | ❌ Ignorées |
| Monotonie buckets | ❌ Buckets 10D non ordonnables intuitivement |

### Complexité d'implémentation
- **Lignes** : ~100 lignes (histogramme + k-means)
- **Dépendances** : sklearn, numpy
- **Risques** :
  - 200 MC = inference lente pour 3-max live (agent doit décider en <3s)
  - Histogramme 10D → k-means difficile à interpréter / debugger
  - Moins intuitif pour calibration future

### Compatibilité
- OFT : ✅
- Migration progressive : ✅
- Checkpoints Deep CFR : ✅

### Scores
| Axe | Score /10 |
|-----|-----------|
| Qualité stratégique | **8/10** |
| Coût compute | **6/10** (inference 2× plus lente, non précalculable) |
| Faisabilité 1-2 semaines | **7/10** |
| Robustesse production | **6/10** (k-means 10D instable) |
| **TOTAL** | **27/40** |

---

## Méthode E — E[HS²] + K-means 2D (hybride recommandée)

### Principe
Combiner la rigueur théorique de E[HS²] (Papp 1998, utilisé dans Libratus)
avec la flexibilité du k-means 2D.

**Features** : uniquement `(E[HS], E[HS²])` → vecteur 2D.
- E[HS] = force actuelle de la main
- E[HS²] - E[HS]² = potentiel résiduel = discriminant clé

**Clusterer** : k-means sur 2D avec n=50 → 50 buckets.
**Inference** : calculer (E[HS], E[HS²]) en 100 MC → lookup centroïde → O(50).

C'est exactement ce qu'utilise Claudico / Libratus pour la card abstraction,
avec k ≈ 50-100 buckets par street.

### Formule
```python
def calculer_hs_carre(main, board, nb_adv=2, n_sims=100):
    hs_vals = []
    for _ in range(n_sims):
        runout = completer_board_aleatoire(main, board)
        hs = rang_normalise(main, runout, nb_adv)  # 0..1
        hs_vals.append(hs)
    return np.mean(hs_vals), np.mean(np.array(hs_vals)**2)

# Calibration :
features_train = [calculer_hs_carre(m, b, n_sims=500)
                  for m, b in situations_train_10000]
centroides = KMeans(n_clusters=50, random_state=42).fit(features_train)

# Inference :
hs, hs2 = calculer_hs_carre(main, board, n_sims=100)
bucket = centroides.predict([[hs, hs2]])[0]
```

### Distribution attendue des 50 buckets
```
Axe Y = E[HS²] (potentiel)
│         •nuts
│       ••
│     ••  ← flush draws haute
│   ••    ← pair + draw
│ ••      ← bluff catchers
│•        ← air total
└────────── Axe X = E[HS] (force actuelle)
```
Les 50 centroides couvrent cette espace 2D naturellement.

### Coût compute
| Étape | Volume | Temps CPU local | Temps Cloud |
|-------|--------|-----------------|-------------|
| Calibration train | 10 000 × 500 MC | 3-4h (8 cores) | ~20min (96 cores) |
| K-means 2D, n=50 | 10 000 × 2 features | <1min | idem |
| Re-entraîne MCCFR (50 buckets) | blueprint complet | ~48h (CPU) | ~2h (96 cores) ≈ 12€ |
| **Total** | — | — | **≈ 14€ sur budget 50€** |

**Inference** : 100 MC → calcul HS → calcul HS² → lookup 50 centroides
→ **~0.5ms** (même que méthode actuelle)

### Qualité stratégique — comparaison détaillée
| Scénario | Méthode actuelle | E[HS²] + K-means |
|----------|-----------------|------------------|
| J♠T♠ sur Q♥8♣3♦ (OESD, 35% eq) | bucket 4 | bucket_draw (E[HS²] élevé) |
| 6♥6♦ sur Q♥8♣3♦ (PP faible, 35% eq) | bucket 4 | bucket_pair (E[HS²] faible) |
| A♦9♦ sur K♦7♦2♣ (FD, 40% eq) | bucket 5 | bucket_strong_draw |
| T♥T♦ sur K♦7♦2♣ (overpair, 60% eq) | bucket 6 | bucket_value |
| 7♣5♣ sur 8♦6♦3♥ (FD+OESD, 55% eq) | bucket 6 | bucket_monster_draw |

**Résultat** : les mains "semi-bluff" avec potentiel élevé ET force actuelle
modérée sont correctement séparées des mains value statiques.

| Critère | Évaluation |
|---------|------------|
| FD vs paire faible | ✅ Excellent |
| Value vs bluff catcher | ✅ Bon |
| Draws forts (OESD, FD) vs draws faibles | ✅ Distingués par E[HS²] |
| Ranges adverses | ❌ Ignorées (compromis assumé) |
| Monotonie buckets | ✅ Clusters ordonnables par E[HS] ≈ force brute |

### Complexité d'implémentation
```
Nouveau fichier : abstraction/card_abstraction_v2.py (~200 lignes)
  - calculer_hs_carre(main, board, n_sims)
  - calibrer_kmeans_50(nb_situations, n_sims_calibration)
  - sauvegarder_centroides(path)
  - charger_centroides(path)
  - class AbstractionCartesV2:
      bucket(cartes, board) → int 0..49
      bucket_et_equite(cartes, board) → (int, float, float)

Script recalibration : recalibrer_3max_v2.py (~80 lignes)
  - Génère data/strategy/centroides_50buckets.npy

Modification minimale : card_abstraction.py
  - Ajouter instance globale abstraction_cartes_v2
  - Laisser abstraction_cartes v1 intacte (migration progressive)
```

- **Dépendances** : sklearn (kmeans) + numpy → **sklearn déjà probable**
- **Risques** :
  - Rang normalisé (HS) nécessite de connaître les 2 adversaires → approximation
    avec nb_adv=2 aléatoires (comme méthode actuelle)
  - K-means peut créer des buckets vides si calibration insuffisante → `n_init=20`
  - Centroides à sauvegarder dans `data/strategy/` pour reproductibilité

### Compatibilité
- OFT : ✅ Aucun impact (tracker/mixer opèrent sur actions, pas sur buckets)
- Migration progressive : ✅ (`AbstractionCartesV2` coexiste avec V1)
- Checkpoints Deep CFR : ✅ (`DIM_CARD=4` scalaire → bucket = scalaire infoset seulement)
- Scaling futur (200 buckets) : ✅ Juste changer `n_clusters=200` et recalibrer

### Scores
| Axe | Score /10 |
|-----|-----------|
| Qualité stratégique | **9/10** |
| Coût compute | **9/10** (CPU local pour calibration, Cloud pour MCCFR) |
| Faisabilité 1-2 semaines | **9/10** (2D k-means = code simple) |
| Robustesse production | **9/10** (centroides persistés, seed fixe) |
| **TOTAL ⭐** | **36/40** |

---

## Tableau comparatif

| Méthode | Qualité | Compute | Faisabilité | Robustesse | **Total** |
|---------|---------|---------|-------------|------------|-----------|
| A — E[HS²] pur (1D bisect) | 8 | 9 | 8 | 8 | 33 |
| B — OCHS | 9 | 5 | 5 | 6 | 25 |
| C — K-means 6D features | 9 | 8 | 7 | 8 | 32 |
| D — Histogram 10D | 8 | 6 | 7 | 6 | 27 |
| **E — E[HS²] + K-means 2D** | **9** | **9** | **9** | **9** | **36 ⭐** |

---

## Recommandation finale

**→ Méthode E : E[HS²] + K-means 2D, 50 clusters**

**Justification sur les 5 critères prioritaires :**

1. **Qualité stratégique maximale** : E[HS²] est la feature minimale suffisante
   pour distinguer les draw hands des made hands à équité égale. C'est la méthode
   utilisée par Claudico et documentée dans la littérature (Papp 1998, Johanson 2013).
   Le k-means 2D sur (E[HS], E[HS²]) produit des clusters naturellement interprétables
   et couvre l'espace (force × potentiel) de manière uniforme.

2. **Faisabilité 1-2 semaines** : 200 lignes de code, dépendance sklearn minimale,
   pas de features complexes à implémenter (pas d'`outs` à calculer, pas de hand
   evaluator customisé). La calibration est un script autonome.

3. **Migration progressive** : `AbstractionCartesV2` coexiste avec V1. On peut
   tester en parallèle sur un blueprint séparé sans toucher au code production.

4. **OFT préservé** : aucun impact. Le bucket est un entier dans la clé infoset ;
   OFT opère sur les actions observées, pas sur les buckets.

5. **Scaling futur** : passer à 100 ou 200 buckets = changer `n_clusters` et
   relancer le script de calibration. Aucune refactorisation architecturale.

**Rejet de OCHS (B)** : les ranges adverses en Spin 15€ 3-max sont trop hétérogènes
et position-dépendantes pour être prédéfinies statiquement. L'inference 3× plus lente
est rédhibitoire pour le live. OCHS est pertinent en cash game avec ranges stables.

**Pourquoi pas C (k-means 6D)** : E[HS²] capture déjà 80% de la valeur des features
composites (f1, f2). Ajouter `outs`, `made_hand`, `nut_potential` ajoute de la complexité
pour un gain marginal. La règle "le modèle le plus simple qui satisfait les critères" s'applique.

---

## Questions ouvertes

**Q1 — Nombre de simulations pour l'inference (trade-off latence/qualité) :**
La méthode actuelle utilise 100 MC pour l'inference postflop. E[HS²] nécessite
le même calcul. Est-ce que 50 MC suffisent pour une variance acceptable sur le
bucket résultant ? (à valider empiriquement par le test de déterminisme).

**Q2 — Street-specific vs street-agnostic :**
Doit-on entraîner 3 sets de centroides séparés (flop/turn/river) ou un seul ?
Les distributions d'équité flop vs river sont très différentes (flop = plus de
variance, river = résolu). 3 sets = meilleure qualité mais ×3 la complexité
de calibration. Recommandation : commencer par 1 set global, valider, puis
séparer si le winrate ne progresse pas.

**Q3 — Préflop : migrer aussi ou garder 8 buckets preflop ?**
La table de lookup preflop actuelle (169 mains → 8 buckets par percentile)
est de bonne qualité pour le preflop 3-max. Passer à 20 buckets preflop
(plus de granularité AA vs KK vs QQ) est une amélioration indépendante.
Recommandation : Phase 2 = postflop uniquement (8→50), preflop reste à 8.
Phase 3 ou 4 : monter preflop à 20 buckets.

---

_Brainstorm généré le 2026-04-26 — à valider avant Étape 3 (spec complète)_
