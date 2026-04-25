# P1 — Plan d'exécution

_Basé sur les hypothèses de `hypotheses.md`. Budget : 3 jours max._
_Point d'arrêt global : winrate vs random ≥ -40 bb/100 (gain ≥ +30 bb/100 vs baseline -68.4)._

---

## Métadonnées

- **Baseline** : -68.4 bb/100 vs random | -78.4 vs raise-only | +13.5 vs TAG (3 runs, seeds=[42,123,2026])
- **Objectif** : ≥ -40 bb/100 vs random, ≥ -50 bb/100 vs raise-only
- **Kill-switch** : si après Exp 01+02, winrate vs random < -63.4 bb/100 (= baseline + 5) → sauter Exp 03, aller directement Exp 04
- **Structure runs** : 3 runs indépendants par expérience, seeds=[42, 123, 2026], 500 mains × 6 bots
- **Branche git par expérience** : `exp/P1-hX-nom`
- **Commit format** : `exp(P1-hX): description`
- **Résultats** : `docs/investigations/P1-winrates-negatifs/experiments/XX-nom/result.md`

---

## Expérience 00 — Baseline (DONE)

**Statut :** ✅ Complétée — référence zéro établie

| Adversaire | Run 1 | Run 2 | Run 3 | Moyenne | Écart-type |
|------------|------:|------:|------:|--------:|-----------:|
| aleatoire  | -68.2 | -66.6 | -70.5 |   -68.4 |        2.0 |
| call       | +21.5 | -45.9 | +29.4 |    +1.6 |       41.4 |
| raise      | -86.7 | -66.0 | -82.7 |   -78.4 |       11.0 |
| tag        | +17.7 | +19.3 |  +3.5 |   +13.5 |        8.7 |
| lag        | -15.0 |  +4.7 |  -7.2 |    -5.8 |        9.9 |
| regulier   | +10.7 | +30.2 | +11.3 |   +17.4 |       11.1 |

**Seuil kill-switch : -63.4 bb/100 vs random** (baseline -68.4 + 5)

---

## Expérience 01 — H3 : Seed fixée + 1000 simulations MC (Option A)

**Durée estimée :** 30 min dev + 30 min eval
**Branche :** `exp/P1-h3-seed-fixee`

### Correctif (Option A)

- Fixer `random.seed(42)` avant chaque appel MC dans `AbstractionCartes.bucket()`
- Augmenter `NB_SIMULATIONS` de 100 → 1000 (réduction variance estimateur)
- **Ne PAS recalibrer les seuils de bucket** (hors périmètre H3)

### Checklist

- [ ] Créer branche `exp/P1-h3-seed-fixee` depuis `main`
- [ ] Écrire le test TDD :
  ```python
  def test_bucket_deterministe_avec_seed():
      from treys import Card
      cartes = [Card.new('As'), Card.new('Kh')]
      board  = [Card.new('2c'), Card.new('7d'), Card.new('Jh')]
      b1 = abstraction_cartes.bucket(cartes, board)
      b2 = abstraction_cartes.bucket(cartes, board)
      assert b1 == b2, f"Buckets non reproductibles : {b1} != {b2}"
  ```
- [ ] Vérifier RED
- [ ] Modifier `NB_SIMULATIONS = 1000` et fixer `random.seed(42)` dans `bucket()`
- [ ] Vérifier GREEN
- [ ] `pytest tests/ -q` → tous green
- [ ] Lancer eval 3 runs (seeds=[42,123,2026]) × 500 mains × 6 bots
- [ ] Rédiger `experiments/01-H3/result.md` (tableau moyennes ± écart-type)

### Critère de décision

| Condition | Action |
|-----------|--------|
| Variance inter-runs réduite ET gain ≥ +5 bb/100 vs random | Merger, continuer H1 |
| Gain < +5 bb/100 mais variance réduite | Garder le fix (qualité), continuer H1 |
| Aucun effet | Abandonner, continuer H1 sans ce fix |

---

## Expérience 02 — H1+H6 : Heuristique fallback différenciée face aux raises

**Durée estimée :** 1h dev + 30 min eval
**Branche :** `exp/P1-h1-heuristique-agressive`
**Dépend de :** exp 01 terminée

### Problème identifié

`agent.py:936-941` — branche `else` (BTN + situations sans contrainte) :
```python
vec[_IDX_FOLD]        = 0.10  # trop peu de fold
vec[_IDX_CHECK]       = 0.60  # passif
vec[_IDX_CALL]        = 0.60  # passif
vec[idx_raise_milieu] = 0.30  # insuffisant
```

Face à raise-only, BTN finit dans cette branche avec face_a_raise=True → normalisé :
FOLD=10%, CALL=60%, RAISE=30%. Trop passif.

### Correctif proposé

```python
if face_a_raise:
    if bucket >= 5:                       # main forte
        vec[_IDX_FOLD]        = 0.10
        vec[_IDX_CALL]        = 0.25
        vec[idx_raise_fort]   = 0.65     # 3-bet fort
    elif bucket >= 3:                     # main moyenne
        vec[_IDX_FOLD]        = 0.40
        vec[_IDX_CALL]        = 0.40
        vec[idx_raise_milieu] = 0.20
    else:                                 # main faible
        vec[_IDX_FOLD]        = 0.75
        vec[_IDX_CALL]        = 0.20
        vec[idx_raise_petit]  = 0.05
else:
    if bucket >= 5:
        vec[_IDX_CHECK]       = 0.20
        vec[_IDX_RAISE_BASE]  = 0.50
        vec[_IDX_FOLD]        = 0.05
        vec[_IDX_CALL]        = 0.25
    elif bucket >= 3:
        vec[_IDX_CHECK]       = 0.45
        vec[_IDX_RAISE_BASE]  = 0.35
        vec[_IDX_FOLD]        = 0.05
        vec[_IDX_CALL]        = 0.15
    else:
        vec[_IDX_FOLD]        = 0.10
        vec[_IDX_CHECK]       = 0.70
        vec[_IDX_CALL]        = 0.10
        vec[idx_raise_petit]  = 0.10
```

### Checklist

- [ ] Créer branche `exp/P1-h1-heuristique-agressive`
- [ ] Écrire test TDD (BTN face raise, bucket faible → fold > 50%)
- [ ] Vérifier RED
- [ ] Implémenter le correctif
- [ ] Vérifier GREEN
- [ ] `pytest tests/ -q` → tous green
- [ ] **Vérifier : exploitabilité Kuhn < 0.005** (si test disponible)
- [ ] Eval 3 runs (seeds=[42,123,2026]) × 500 mains × 6 bots
- [ ] `experiments/02-H1/result.md`

### Critère de décision (STRICTS)

| Condition | Action |
|-----------|--------|
| Gain ≥ +30 bb/100 vs random | STOP — objectif atteint, merger, capitaliser |
| Gain +10 à +30 bb/100 ET TAG > 0 | Merger, continuer H2+H5 |
| TAG winrate < 0 | **REVERTER immédiatement** — réajuster les poids |
| Gain < +10 bb/100 | Garder ou reverter selon impact TAG |
| Kuhn exploitabilité > 0.005 | **REVERTER** — invariant violé |

### Kill-switch post Exp 01+02

Si winrate vs random après Exp 01+02 < **-63.4 bb/100** (baseline + 5) :
→ **Sauter Exp 03, aller directement à Exp 04** (module d'exploitation minimal)

---

## Expérience 03 — H2+H5 : Instrumentation du hit rate par adversaire

**Durée estimée :** 1h dev + 15 min analyse
**Branche :** `exp/P1-h2-hitrate-logging`
**Dépend de :** exp 01 et 02 terminées ET kill-switch NON déclenché

### Objectif

Outil de diagnostic — pas un correctif. Même si H1+H3 ont déjà amélioré le winrate,
utile pour comprendre les limitations structurelles restantes.

### Checklist

- [ ] Créer branche `exp/P1-h2-hitrate-logging`
- [ ] Ajouter dans `AgentAXIOM.stats` :
  - `blueprint_hit_par_adversaire` : dict adversaire → (hits, lookups)
  - `bucket_distribution` : Counter(bucket) par main
  - `hist_profondeur` : Counter(len(hist)) — profondeur des séquences
- [ ] Logger ces stats dans `training/evaluator.py` et sauver dans CSV
- [ ] Lancer 200 mains vs raise-only, 200 vs TAG, comparer les deux
- [ ] `experiments/03-H2H5/result.md` avec les distributions

### Critère de décision

| Condition | Action |
|-----------|--------|
| Hit rate raise-only < 40% | Confirme H2 → ajouter en backlog entraînement |
| Distribution buckets biaisée vs raise-only | Confirme H5 → backlog recalibrage |
| Hit rate raise-only ≈ hit rate TAG | H2/H5 ne sont pas des causes → fermer |

---

## Expérience 04 — H4 : Module d'exploitation minimal

**Durée estimée :** 8-12h dev + 2h eval
**Branche :** `exp/P1-h4-exploitation-minimal`
**Dépend de :** exp 01, 02, 03 terminées ET gain total < objectif  
**OU** : kill-switch déclenché après Exp 01+02

### Portée limitée (ne pas scope-creep)

Compter les actions adverses (fold_freq, call_freq, raise_freq) sur les 20 dernières
mains et ajuster les poids de la heuristique en temps réel :

```python
if adversaire.raise_freq > 0.7:   # raise-only détecté
    # utiliser poids h1-h3 agressifs
elif adversaire.fold_freq > 0.5:  # fold-heavy détecté
    # augmenter bet-value, réduire bluff
```

### Critère de décision

| Condition | Action |
|-----------|--------|
| Gain ≥ +20 bb/100 supplémentaires | Merger — module d'exploitation opérationnel |
| Gain < +10 bb/100 | Abandonner — complexité non justifiée |

---

## Planning estimé

| Jour | Expérience | Durée | Livrable |
|------|------------|-------|---------|
| Jour 0 | Exp 00 (baseline) | 1h | ✅ Référence zéro |
| Jour 1 matin | Exp 01 (seed+1000 MC) | 1h | Bucketing déterministe |
| Jour 1 après-midi | Exp 02 (heuristique) | 2h | Fallback agressif BTN |
| Jour 2 matin | Eval + analyse | 2h | result.md exp 01+02, kill-switch check |
| Jour 2 après-midi | Exp 03 (logging) | 2h | Diagnostic hit rate |
| Jour 3 | Exp 04 si nécessaire | 4h | Exploitation minimal |

**Total estimé : 11-15h** (3 jours confort, 2 jours intensif)

---

## Invariants à surveiller (poker-regression-check)

Avant chaque merge :
- `pytest tests/ -q` → tous green ✓
- Clés infoset toujours 7 segments ✓
- DIM_INPUT = 52 ✓ (heuristique ne touche pas les réseaux)
- Somme stacks stable ✓ (pas de modif engine)
- **TAG winrate > 0** ✓ (nouveau — seuil strict Exp 02)
- **Kuhn exploitabilité < 0.005** ✓ (si test disponible)
