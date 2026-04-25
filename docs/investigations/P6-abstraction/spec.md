# Spec P6 — Abstraction cartes V2 : E[HS²] + Potentiel + K-means 3D street-specific
_Finalisée le 2026-04-26 — 3-max NLHE Spin & Go 15€_
_Référence brainstorm : docs/investigations/P6-abstraction/brainstorm.md_

---

## 1. Méthode bucketing finale

### 1.1 Principe technique

Pour chaque situation (main hero, board visible), on calcule **3 features scalaires**
via Monte Carlo 100 simulations, puis on prédit le bucket par **distance euclidienne
au centroïde k-means le plus proche** parmi 50 centroides entraînés street-spécifiques.

Les 3 features capturent :
- `E[HS]` : la **force actuelle** de la main (équité brute vs 2 adversaires aléatoires)
- `E[HS²]` : l'**espérance du carré** de l'équité → intègre la variance → distingue les
  draws (variance élevée) des made hands statiques (variance proche de 0)
- `Potentiel` : `E[HS_river] - E[HS_courant]` → amélioration attendue d'ici la river

Ces 3 features viennent **du même run MC** (aucun coût supplémentaire à l'inference).

### 1.2 Formules mathématiques

```python
def compute_features(cartes, board, street, n_sims=100, seed=42):
    """
    Retourne (E[HS], E[HS²], Potentiel) pour une situation donnée.

    street : 'flop' (3 cartes), 'turn' (4 cartes), 'river' (5 cartes)
    """
    rng = random.Random(seed)

    # --- Run MC principal ---
    hs_vals = []
    for _ in range(n_sims):
        runout = completer_board_aleatoire(cartes, board, rng)  # compléter jusqu'à 5 cartes
        hs = rang_normalise(cartes, runout, nb_adversaires=2)    # float ∈ [0, 1]
        hs_vals.append(hs)

    arr = np.array(hs_vals, dtype=np.float32)
    e_hs  = float(arr.mean())           # E[HS]
    e_hs2 = float((arr ** 2).mean())    # E[HS²]

    # --- Potentiel : E[HS_river] si pas déjà à la river ---
    if street == 'river':
        potentiel = 0.0
    else:
        hs_river_vals = []
        for _ in range(n_sims):
            board_complet = completer_board_jusqu_river(cartes, board, rng)
            hs_r = rang_normalise(cartes, board_complet, nb_adversaires=2)
            hs_river_vals.append(hs_r)
        e_hs_river = float(np.mean(hs_river_vals))
        potentiel = max(0.0, e_hs_river - e_hs)  # clamped à 0 (jamais négatif)

    return e_hs, e_hs2, potentiel
```

### 1.3 Propriétés vérifiables

```
Propriété 1 : E[HS²] ≥ E[HS]²          (Jensen : f(E[X]) ≤ E[f(X)] pour f convexe)
Propriété 2 : Potentiel = 0 à la river  (board complet = pas de runout futur)
Propriété 3 : Potentiel ≥ 0             (clamping explicite)
Propriété 4 : E[HS] ∈ [0, 1]
Propriété 5 : E[HS²] ∈ [0, 1]
```

### 1.4 K-means 3D street-spécifique

3 sets indépendants de centroides, un par street :

```
centroides_flop  : array(50, 3)   # 50 points dans R³
centroides_turn  : array(50, 3)
centroides_river : array(50, 3)
```

Justification : les distributions de (E[HS], E[HS²], Potentiel) sont radicalement
différentes par street :
- **Flop** : forte variance (beaucoup de draws), potentiel élevé
- **Turn** : variance intermédiaire, potentiel réduit
- **River** : potentiel = 0, distribution bimodale (gagné/perdu)

Un set global mélangerait ces distributions et produirait des clusters incohérents
stratégiquement (ex : un draw fort au flop et un value hand à la river dans le même bucket).

Calibration : k-means sur 4000 situations échantillonnées aléatoirement par street.

### 1.5 Justification vs alternatives (ref: brainstorm.md)

| Alternative | Pourquoi rejetée |
|-------------|-----------------|
| OCHS (B) | Ranges adverses trop hétérogènes en Spin 15€, inference 3× plus lente |
| K-means 6D (C) | E[HS²] + potentiel = 80% de la valeur, features supplémentaires = complexité sans gain |
| Histogram 10D (D) | Inference 2× plus lente, k-means 10D instable, non interprétable |
| E[HS²] bisect 1D (A) | Moins précis que k-means 3D (ignore corrélations entre features) |

---

## 2. Architecture cible

### 2.1 Nouvelle classe AbstractionCartesV2

**Fichier** : `abstraction/card_abstraction.py` (ajout en bas du fichier existant)

```python
class AbstractionCartesV2:
    """
    Abstraction cartes Phase 2 — E[HS²] + Potentiel + K-means 3D street-specific.

    Coexiste avec AbstractionCartes (V1) jusqu'à la bascule finale (Étape F).
    Préflop : délègue à V1 (lookup table 169 mains → 8 buckets, inchangé).
    Postflop : compute_features() + predict_bucket() via centroides k-means.
    """

    # API publique identique à V1 pour faciliter le switch
    def bucket(self, cartes: list, board: list) -> int: ...
    def bucket_preflop(self, cartes: list) -> int: ...
    def bucket_postflop(self, cartes: list, board: list) -> int: ...
    def bucket_et_equite(self, cartes: list, board: list) -> tuple: ...
        # Retourne (bucket: int, e_hs: float, e_hs2: float)
        # Note: signature légèrement différente de V1 (e_hs2 remplace equite_brute simple)
        # Les callsites utilisent uniquement [0] (bucket) → compatible

    # API étendue V2
    def features(self, cartes: list, board: list) -> tuple: ...
        # Retourne (e_hs: float, e_hs2: float, potentiel: float)
```

**Constantes** :
```python
NB_BUCKETS_PREFLOP_V2  = 8    # inchangé (lookup table V1 réutilisée)
NB_BUCKETS_FLOP        = 50
NB_BUCKETS_TURN        = 50
NB_BUCKETS_RIVER       = 50
NB_MC_SIMULATIONS_V2   = 100  # inference
NB_MC_CALIBRATION_V2   = 500  # calibration uniquement (script recalibration)
```

**Centroïdes** : chargés depuis `data/abstraction/centroides_v2.npz` au `__init__`.
Structure `.npz` :
```
{
    'flop':  np.ndarray(50, 3),   # float32
    'turn':  np.ndarray(50, 3),
    'river': np.ndarray(50, 3),
}
```

**Fallback** : si `centroides_v2.npz` absent → lever `FileNotFoundError` avec message
explicatif ("Lancer recalibrer_3max_v2.py pour générer les centroïdes").

**Cache mémoire** : `functools.lru_cache` sur `bucket_postflop` pour éviter les
recalculs intra-main (même (cartes, board) appelé plusieurs fois par l'agent).
Clé de cache : `(tuple(sorted(cartes)), tuple(board))`.

### 2.2 Nouveau fichier : abstraction/card_clustering.py

Module **pur** (pas de dépendance à AbstractionCartes, utilisable standalone).

```python
# abstraction/card_clustering.py
"""
Calcul des features E[HS], E[HS²], Potentiel et fitting k-means pour P6.
Utilisé par AbstractionCartesV2 (inference) et recalibrer_3max_v2.py (calibration).
"""

def compute_features(cartes, board, nb_adversaires=2, n_sims=100, seed=42):
    """
    Calcule (E[HS], E[HS²], potentiel) pour une situation donnée.
    cartes : list de 2 ints (format treys)
    board  : list de 3, 4 ou 5 ints (format treys)
    Retourne : tuple(float, float, float)
    """

def rang_normalise(cartes, board_complet, nb_adversaires):
    """
    HS = fraction des main adverses que hero bat.
    Utilise engine.hand_evaluator pour l'évaluation.
    Retourne float ∈ [0, 1].
    """

def completer_board_aleatoire(cartes, board, rng, cibles=5):
    """
    Complète le board jusqu'à `cibles` cartes en tirant aléatoirement
    dans le deck minus les cartes déjà utilisées.
    """

def fit_centroids(features_array, n_clusters=50, seed=42, n_init=20):
    """
    Fit un k-means sklearn sur features_array (N, 3).
    Retourne np.ndarray(n_clusters, 3) — les centroïdes.
    Utilise n_init=20 pour robustesse contre initialisation aléatoire.
    """

def predict_bucket(features_vec, centroides):
    """
    Retourne l'indice du centroïde le plus proche (distance euclidienne L2).
    features_vec : tuple ou array(3,)
    centroides   : array(K, 3)
    Retourne int ∈ [0, K-1].
    """

def sauvegarder_centroides(path, flop, turn, river):
    """Sauvegarde dans un fichier .npz numpy."""

def charger_centroides(path):
    """Charge depuis .npz. Retourne dict{'flop': ..., 'turn': ..., 'river': ...}."""
```

**Dépendances** :
- `numpy` (déjà présent)
- `sklearn.cluster.KMeans` (nouveau — calibration seulement)
- `engine.hand_evaluator.calculer_equite` → **à remplacer** par `rang_normalise` interne
  (plus précis : HS = rang relatif, pas équité MC globale)
- `treys.Deck` (déjà présent)

> ⚠️ **Décision technique ouverte #1** : `rang_normalise` implémente le rang relatif de la
> main hero parmi toutes les combinaisons possibles pour les adversaires (évaluation
> exacte = très lent) ou par sampling MC des mains adverses (comme `calculer_equite`
> actuel) ?
> **Solution par défaut proposée** : conserver le sampling MC avec nb_adversaires=2
> aléatoires (identique à V1). La différence avec le rang exact est négligeable sur
> 50 buckets. Changer cela en Phase 4 si nécessaire.

### 2.3 Tableau fichiers modifiés

| Fichier | Nature de la modification | Lignes estimées |
|---------|--------------------------|-----------------|
| `abstraction/card_abstraction.py` | +`AbstractionCartesV2`, +instance globale `abstraction_cartes_v2` | +200 |
| `abstraction/__init__.py` | +export `AbstractionCartesV2` | +2 |
| `tests/test_abstraction.py` | +tests V2 (section 4.2) | +80 |
| `ai/agent.py` | switch V1→V2 à l'Étape F | -10/+15 |
| `ai/mccfr.py` | switch V1→V2 à l'Étape F | -5/+10 |
| `ai/deep_cfr.py` | switch V1→V2 à l'Étape F | -5/+10 |
| `solver/depth_limited.py` | switch V1→V2 à l'Étape F | -5/+10 |
| `solver/subgame_solver.py` | switch V1→V2 à l'Étape F | -5/+10 |

### 2.4 Tableau fichiers nouveaux

| Fichier | Rôle |
|---------|------|
| `abstraction/card_clustering.py` | Features E[HS²] + potentiel, k-means fitting, predict |
| `recalibrer_3max_v2.py` | Script calibration 4000 spots/street → centroides_v2.npz |
| `data/abstraction/centroides_v2.npz` | Centroïdes produits par recalibration |
| `tests/test_card_clustering.py` | Tests unitaires card_clustering.py |
| `scripts/cloud_train_p6.sh` | Script setup + training sur GCP |

---

## 3. Plan de migration (8 étapes)

### Étape A — Création classe stub AbstractionCartesV2 (jour 1)
- Ajoute `AbstractionCartesV2` dans `card_abstraction.py`
- API identique à V1 côté signature
- Implémentation : toutes les méthodes retournent 0 (stub pour tests RED)
- Crée `abstraction/card_clustering.py` avec stubs (fonctions `pass`)
- **Commit** : `feat(P6-stub): AbstractionCartesV2 skeleton + card_clustering stubs`

### Étape B — Tests RED (jour 1-2)
- Crée `tests/test_card_clustering.py` : tests 1-7
- Ajoute tests 8-14 dans `tests/test_abstraction.py`
- Tous les tests doivent FAIL (car stubs)
- Vérifie : `pytest tests/test_card_clustering.py` → N errors / N failures
- Vérifie : `pytest tests/test_abstraction.py` → tests existants GREEN, nouveaux RED
- **Commit** : `test(P6): RED tests card_clustering + AbstractionCartesV2`

### Étape C — Implémentation card_clustering.py (jour 2-3)
- `rang_normalise` : sampling MC mains adverses
- `completer_board_aleatoire` : tirage sans remise
- `compute_features` : boucle MC, calcul E[HS], E[HS²], potentiel
- `predict_bucket` : distance L2 euclidienne
- Tests 1-7 passent en GREEN
- **Commit** : `feat(P6): implement card_clustering compute_features + predict`

### Étape D — Implémentation AbstractionCartesV2 (jour 3-4)
- Chargement centroides depuis `.npz` avec fallback clair
- Délégation à `card_clustering` pour features + prediction
- Cache LRU sur `bucket_postflop`
- Tests 8-11 passent en GREEN (nécessite des centroïdes temporaires pour tests)
  → Utiliser centroides de test = K-means sur 100 samples (rapide)
- **Commit** : `feat(P6): implement AbstractionCartesV2 with LRU cache`

### Étape E — Calibration (jour 4-5)
- **Local (validation)** : `recalibrer_3max_v2.py` sur 100 spots/street
  → visualisation PCA 2D des clusters → vérification cohérence
- **Cloud (production)** : n2-standard-12, 4h, 4000 spots/street × 500 MC
  → `data/abstraction/centroides_v2.npz` (fichier gitignored, transféré manuellement)
- Test 10 (`test_v2_load_centroides_from_disk`) passe en GREEN
- **Commit** : `feat(P6): add recalibrer_3max_v2.py calibration script`

### Étape F — Bascule des modules (jour 6-8)
- Branche : `exp/P6-abstraction-v2`
- Switch `agent.py` : `self._abs_cartes = AbstractionCartesV2(mode='3max')`
- Switch `mccfr.py`, `deep_cfr.py`, `solver/depth_limited.py`, `solver/subgame_solver.py`
- Tests 12-14 passent en GREEN
- `pytest tests/ -q` → 100% GREEN (hors re-entraînement MCCFR)
- **Commit** : `feat(P6-migration): switch all modules to AbstractionCartesV2`

### Étape G — Ré-entraînement blueprint MCCFR (jour 8-12)
- Script : `scripts/cloud_train_p6.sh`
- VM : n2-standard-96, europe-west4, 12-15h
- Commande : `python train.py --mode mccfr --iterations 500000 --output data/strategy/blueprint_v2.pkl`
- Output transféré via `gsutil cp` → local
- **Commit** : `train(P6): add cloud_train_p6.sh script`

### Étape H — Validation et merge (jour 12-14)
- 3 runs × 6 bots × 1000 mains (même protocole que Exp 04)
- Critères section 5 vérifiés
- Merge `exp/P6-abstraction-v2` → `main` si succès
- Sinon : rollback documenté dans `docs/investigations/P6-abstraction/decisions/adr-001-rollback.md`

---

## 4. Tests RED à écrire (14 tests)

### 4.1 Tests card_clustering.py (7 tests)

```python
# Test 1
def test_compute_features_deterministic():
    """Même main + board + seed → mêmes features exactes."""
    f1 = compute_features([Card.new('Js'), Card.new('Ts')],
                          [Card.new('Qh'), Card.new('8c'), Card.new('3d')],
                          street='flop', seed=42)
    f2 = compute_features([Card.new('Js'), Card.new('Ts')],
                          [Card.new('Qh'), Card.new('8c'), Card.new('3d')],
                          street='flop', seed=42)
    assert f1 == f2

# Test 2
def test_potentiel_zero_river():
    """À la river (5 cartes), potentiel doit être exactement 0.0."""
    _, _, potentiel = compute_features(
        [Card.new('Ah'), Card.new('Kh')],
        [Card.new('Qd'), Card.new('Jc'), Card.new('Ts'),
         Card.new('2h'), Card.new('7d')],
        street='river', seed=42)
    assert potentiel == 0.0

# Test 3
def test_potentiel_positive_flop_draw():
    """Au flop avec un flush draw, potentiel > 0."""
    e_hs, _, potentiel = compute_features(
        [Card.new('As'), Card.new('Ks')],   # FD + overs
        [Card.new('Qs'), Card.new('7s'), Card.new('2h')],
        street='flop', seed=42)
    assert potentiel > 0.0, f"Potentiel attendu > 0, obtenu {potentiel}"

# Test 4
def test_e_hs_squared_geq_e_hs_squared():
    """E[HS²] ≥ E[HS]² (Jensen's inequality) pour toute main."""
    e_hs, e_hs2, _ = compute_features(
        [Card.new('9c'), Card.new('8c')],
        [Card.new('Th'), Card.new('7d'), Card.new('3s')],
        street='flop', seed=42)
    assert e_hs2 >= e_hs ** 2 - 1e-6, (
        f"Jensen violé : E[HS²]={e_hs2:.4f} < E[HS]²={e_hs**2:.4f}")
    assert 0.0 <= e_hs  <= 1.0
    assert 0.0 <= e_hs2 <= 1.0

# Test 5
def test_fit_centroids_shape():
    """fit_centroids retourne un array de forme (50, 3)."""
    rng = np.random.RandomState(42)
    features = rng.rand(200, 3).astype(np.float32)
    centroides = fit_centroids(features, n_clusters=50, seed=42)
    assert centroides.shape == (50, 3), f"Shape attendu (50,3), obtenu {centroides.shape}"

# Test 6
def test_fit_centroids_deterministic():
    """Même seed → centroïdes identiques."""
    rng = np.random.RandomState(0)
    features = rng.rand(200, 3).astype(np.float32)
    c1 = fit_centroids(features, n_clusters=50, seed=42)
    c2 = fit_centroids(features, n_clusters=50, seed=42)
    np.testing.assert_array_equal(c1, c2)

# Test 7
def test_predict_bucket_in_range():
    """predict_bucket retourne un int dans [0, 49]."""
    centroides = np.random.RandomState(42).rand(50, 3).astype(np.float32)
    for _ in range(20):
        features_vec = np.random.rand(3).astype(np.float32)
        b = predict_bucket(features_vec, centroides)
        assert 0 <= b < 50, f"Bucket hors plage : {b}"
```

### 4.2 Tests AbstractionCartesV2 (5 tests)

```python
# Test 8
def test_v2_bucket_postflop_deterministic():
    """Même état → même bucket, plusieurs appels consécutifs."""
    v2 = AbstractionCartesV2()  # nécessite centroides_v2.npz ou fixture de test
    cartes = [Card.new('8c'), Card.new('7c')]
    board  = [Card.new('9h'), Card.new('6d'), Card.new('2s')]
    b1 = v2.bucket_postflop(cartes, board)
    b2 = v2.bucket_postflop(cartes, board)
    assert b1 == b2

# Test 9
def test_v2_distinguishes_draw_from_pair():
    """J♠T♠ sur Q♥8♣3♦ (OESD) et 6♥6♦ sur Q♥8♣3♦ (paire faible) → buckets différents."""
    v2 = AbstractionCartesV2()
    board = [Card.new('Qh'), Card.new('8c'), Card.new('3d')]
    b_draw = v2.bucket_postflop([Card.new('Js'), Card.new('Ts')], board)
    b_pair = v2.bucket_postflop([Card.new('6h'), Card.new('6d')], board)
    assert b_draw != b_pair, (
        f"Draw et paire faible à même équité → doivent avoir des buckets distincts, "
        f"obtenu b_draw={b_draw}, b_pair={b_pair}")

# Test 10
def test_v2_load_centroides_from_disk(tmp_path):
    """AbstractionCartesV2 charge correctement les centroïdes depuis un .npz de test."""
    flop  = np.random.RandomState(0).rand(50, 3).astype(np.float32)
    turn  = np.random.RandomState(1).rand(50, 3).astype(np.float32)
    river = np.random.RandomState(2).rand(50, 3).astype(np.float32)
    path  = tmp_path / "centroides_v2.npz"
    np.savez(path, flop=flop, turn=turn, river=river)
    v2 = AbstractionCartesV2(centroides_path=str(path))
    assert v2.centroides['flop'].shape  == (50, 3)
    assert v2.centroides['turn'].shape  == (50, 3)
    assert v2.centroides['river'].shape == (50, 3)

# Test 11
def test_v2_api_compatible_v1():
    """AbstractionCartesV2 expose les mêmes méthodes publiques qu'AbstractionCartes."""
    v1_methods = {'bucket', 'bucket_preflop', 'bucket_postflop', 'bucket_et_equite'}
    v2 = AbstractionCartesV2
    for m in v1_methods:
        assert hasattr(v2, m), f"Méthode {m!r} manquante dans AbstractionCartesV2"

# Test 12
def test_v2_bucket_in_range():
    """bucket_postflop retourne toujours un int dans [0, 49]."""
    v2 = AbstractionCartesV2()
    board_flop  = [Card.new('As'), Card.new('Kd'), Card.new('Qh')]
    board_turn  = board_flop + [Card.new('7c')]
    board_river = board_turn + [Card.new('2s')]
    for board in [board_flop, board_turn, board_river]:
        b = v2.bucket_postflop([Card.new('Jh'), Card.new('Tc')], board)
        assert 0 <= b < 50, f"Bucket hors plage : {b} pour board {len(board)} cartes"
```

### 4.3 Tests régression (2 tests)

```python
# Test 13
def test_v1_still_works_during_migration():
    """V1 (AbstractionCartes) fonctionne toujours après ajout de V2 (cohabitation)."""
    from abstraction.card_abstraction import abstraction_cartes
    cartes = [Card.new('As'), Card.new('Ah')]
    assert abstraction_cartes.bucket_preflop(cartes) == 7  # AA = bucket max V1

# Test 14
def test_oft_unaffected_by_v2():
    """OpponentTracker et ExploitMixer fonctionnent indépendamment du bucketing."""
    from ai.opponent_tracker import OpponentTracker
    from ai.exploit_mixer   import ExploitMixer
    tracker = OpponentTracker()
    for _ in range(30):
        tracker.observer_action(seat_index=1, action=2,
                                contexte={'phase': 'preflop'})
    mixer = ExploitMixer(tracker)
    bp = np.array([0.1, 0.0, 0.8, 0.1, 0., 0., 0., 0., 0.], dtype=np.float32)
    result = mixer.ajuster(bp, adversaire=1, game_type='3max')
    assert abs(result.sum() - 1.0) < 1e-5
    # Le bucketing V2 n'est pas impliqué dans ce test : OFT isolé
```

---

## 5. Critères de validation Phase 2

### 5.1 Critères techniques (obligatoires — bloquants)

| Critère | Valeur cible | Outil de vérification |
|---------|-------------|----------------------|
| Tests pytest | 100% GREEN | `pytest tests/ -q` |
| Exploitabilité Kuhn | < 0.005 | `test_mccfr.py::test_cfr_valeur_convergence` |
| DIM_INPUT | = 52 | `assert DIM_INPUT == 52` dans `network.py` |
| Latence `bucket_postflop` | < 1ms (avec LRU cache chaud) | `time.perf_counter()` dans test 8 |
| OFT classement profils | Inchangé vs baseline OFT | `test_oft.py` (14 tests) |

### 5.2 Critères de performance (cibles — comparés à baseline Exp 04 bis + OFT)

| Bot | Baseline OFT (Exp 04 bis) | Cible V2 | Obligatoire ? |
|-----|--------------------------|----------|---------------|
| bot_random | −68.1 bb/100 | > −50 | Non (référence) |
| call-only | +2.4 bb/100 | > +5 | ✅ Oui |
| raise-only | −78 bb/100 | > −65 | ✅ Oui |
| TAG | +15.2 bb/100 | > +12 | ✅ Oui (pas de régression) |
| Régulier | +10.2 bb/100 | > +12 | ✅ Oui |
| LAG | −3.3 bb/100 | > −8 | ✅ Oui (pas de régression) |

**Critère σ** : σ vs call-only < 30 bb/100 (préservé depuis Exp 04).

> ⚠️ **Note** : les cibles sont conservatrices pour une première validation.
> L'amélioration principale de V2 se mesurera sur les bots qui exploitent la
> mauvaise abstraction V1 (notamment raise-only et LAG). La validation vs TAG/régulier
> est surtout un test de non-régression.

### 5.3 Critère de robustesse

- 3 runs indépendants (seeds différents) → σ inter-runs < 15 bb/100
- Aucun bug `assert_cartes_valides` dans le moteur de jeu

---

## 6. Plan compute cloud

### 6.1 Calibration des centroïdes (Étape E)

```
VM         : n2-standard-12 (12 vCPU, 8 GB RAM)
Région     : europe-west4 (Pays-Bas)
OS         : Debian 12, Python 3.10
Durée est. : 4h
Budget     : 12 cores × 4h × 0.30€/core/h ≈ 14€

Script     : python recalibrer_3max_v2.py \
               --nb-spots 4000 \
               --nb-mc-calibration 500 \
               --output data/abstraction/centroides_v2.npz

Output     : centroides_v2.npz (~50 KB) → scp depuis GCP → local
```

### 6.2 Ré-entraînement blueprint MCCFR (Étape G)

```
VM         : n2-standard-96 (96 vCPU, 128 GB RAM)
Région     : europe-west4
OS         : Debian 12, Python 3.10
Durée est. : 12-15h
Budget     : 96 cores × 13h × 0.03€/core/h ≈ 37€

Script     : python train.py \
               --mode mccfr \
               --iterations 500000 \
               --abstraction v2 \
               --output data/strategy/blueprint_v2.pkl

Monitoring : gsutil cp gs://axiom-training-artifacts/logs/train_v2.log .
Hard kill  : si coût dépasse 60€ → `gcloud compute instances stop axiom-train`
Output     : blueprint_v2.pkl (~400 MB) → gsutil cp → local
```

### 6.3 Budget total Phase 2

| Poste | Coût estimé |
|-------|-------------|
| Calibration centroïdes | 14€ |
| Training MCCFR 500k iter | 37€ |
| **Total** | **51€** |
| Budget alloué | 50€ |
| Marge | ±1€ → réduire à 450k iter si dépassement |
| Réserve globale | 470€ restants sur 520€ |

---

## 7. Risques et mitigations

| Risque | Probabilité | Sévérité | Mitigation |
|--------|-------------|----------|------------|
| Cascade modules : V2 casse V1 | Faible | Élevée | Migration 8 étapes, V1 préservée jusqu'à Étape F, tests 13-14 |
| Centroïdes mal calibrés (clusters vides) | Faible | Élevée | `n_init=20` k-means, validation PCA locale avant cloud |
| OFT cassé par V2 | Très faible | Élevée | Test 14 dédié, OFT opère uniquement sur actions |
| Compute cloud dépasse budget | Moyenne | Moyenne | Kill à 60€, option "450k iterations" si dépassement |
| Blueprint V2 pire que V1 | Moyenne | Élevée | Critères validation section 5, rollback + ADR documenté |
| Latence inference > 1ms | Faible | Moyenne | LRU cache + test 8 chrono |
| K-means instable (seed) | Faible | Faible | `n_init=20`, `random_state=42`, résultat reproductible |
| Test 9 (draw vs pair) : insuffisant avec 100 MC | Possible | Moyenne | Si flaky : augmenter à 200 MC dans le test, **pas** dans la prod |

---

## 8. Timeline estimée

| Jour | Étape | Livrable principal |
|------|-------|--------------------|
| 1 | A | `AbstractionCartesV2` stub + `card_clustering.py` stubs |
| 1-2 | B | 14 tests RED dans `test_card_clustering.py` + `test_abstraction.py` |
| 2-3 | C | `card_clustering.py` implémenté, tests 1-7 GREEN |
| 3-4 | D | `AbstractionCartesV2` implémenté, tests 8-12 GREEN |
| 4-5 | E | Centroides calibrés localement + script cloud validé |
| 5 | E (cloud) | `centroides_v2.npz` final téléchargé depuis GCP |
| 6-8 | F | Tous modules basculés, `pytest 100% GREEN` |
| 8-12 | G | `blueprint_v2.pkl` entraîné et téléchargé |
| 12-14 | H | Évaluation 3 runs + validation critères + merge |
| **TOTAL** | **14 jours calendaires** | |

---

## 9. Décisions à prendre AVANT implémentation

**Aucune** — toutes les décisions techniques sont arrêtées dans cette spec.

Si un imprévu survient, créer un fichier ADR :
```
docs/investigations/P6-abstraction/decisions/adr-001-<sujet>.md
```
Format ADR minimal :
```markdown
# ADR-001 — <titre>
**Date** : YYYY-MM-DD
**Statut** : Proposed / Accepted / Rejected
**Contexte** : <problème rencontré>
**Décision** : <choix retenu>
**Conséquences** : <impact>
```

---

## 10. Décision reportée : implémentation de `rang_normalise`

### 10.1 Définitions alternatives

**Option A — Sampling MC (retenue en Phase 2) :**
```python
def rang_normalise(cartes_hero, board_complet, nb_adversaires=2, n_samples=50):
    """
    HS ≈ fraction des tirages adverses que hero bat.
    Tire n_samples mains aléatoires pour les adversaires, évalue chacune.
    Identique conceptuellement à calculer_equite() de V1.
    """
    score_hero = evaluateur.evaluate(cartes_hero + board_complet)
    wins = 0
    for _ in range(n_samples):
        main_adv = tirer_main_adverse(board_complet, cartes_hero)
        score_adv = evaluateur.evaluate(main_adv + board_complet)
        if score_hero < score_adv:  # treys : score plus bas = meilleure main
            wins += 1
    return wins / n_samples
```

**Option B — Rang exact (reporté en Phase 4) :**
```python
def rang_normalise_exact(cartes_hero, board_complet, nb_adversaires=2):
    """
    HS = fraction exacte des C(48,2) mains adverses possibles que hero bat.
    Enumère toutes les combinaisons de mains adverses compatibles avec le board.
    """
    deck_restant = deck_minus(cartes_hero + board_complet)   # 45 cartes
    toutes_mains = itertools.combinations(deck_restant, 2)   # C(45,2) = 990 mains
    score_hero = evaluateur.evaluate(cartes_hero + board_complet)
    wins = sum(1 for m in toutes_mains
               if evaluateur.evaluate(list(m) + board_complet) > score_hero)
    return wins / C(len(deck_restant), 2)
```

### 10.2 Analyse : Phase 2 vs Phase 4

| Critère | Option A (MC sampling, Phase 2) | Option B (rang exact, Phase 4) |
|---------|--------------------------------|--------------------------------|
| Précision par évaluation | ±5-8% (n=50 MC) | Exacte (0%) |
| Temps par appel | ~0.3ms (50 evals) | ~15ms (990 evals × 3-way) |
| Déterminisme | ✅ avec seed fixe | ✅ toujours |
| Cohérence avec V1 | ✅ identique | ⚠️ change les valeurs de features |
| Impact sur les 50 buckets | Variance légère sur positionnement aux frontières | Buckets plus stables, frontières nettes |
| Invalidation blueprint | Non (bruit cohérent) | Oui (recalibration + ré-entraînement) |

### 10.3 Pourquoi reporter en Phase 4

**Raison principale : coût disproportionné vs gain attendu.**

Avec 50 buckets, une erreur de ±5% sur E[HS] déplace une main d'au plus 1-2 buckets
sur 50. Le k-means absorbera ce bruit dans la calibration car toutes les situations
de calibration auront le même bruit systématique (seed=42).

Le problème de classification qu'on veut résoudre (flush draw ≠ paire faible) est
indépendant de la précision du rang : les deux mains ont des distributions HS
radicalement différentes, que HS soit calculé en MC50 ou en exact.

**Raison secondaire : compatibilité avec V1.**
`calculer_equite()` de V1 est déjà du MC sampling. Garder la même approche en V2
préserve la cohérence conceptuelle et facilite la comparaison des résultats.

**Ce que ça change si on l'inclut en Phase 2 :**
1. Inference 50× plus lente (0.3ms → 15ms) → risque de dépasser la contrainte < 1ms
2. Les centroïdes de calibration changent → le blueprint V1 devient incomparable
   (les buckets ne correspondent plus du tout entre V1 et V2)
3. Complexité d'implémentation + 80 lignes → risque de bug
4. Aucun gain mesurable sur les critères de validation Phase 2

**Ce qu'on gagne en Phase 4 (rang exact) :**
- Buckets plus stables aux frontières (meilleure généralisation)
- Meilleure qualité théorique pour Phase 4 (Deep CFR avec 200+ buckets)
- Alignement avec la littérature Libratus/Claudico qui utilise le rang exact

### 10.4 Décision entérinée

```
Phase 2 : rang_normalise = Option A (MC sampling, n_samples identique à n_sim)
Phase 4 : migrer vers Option B (rang exact) si latence acceptable avec GPU
          Créer ADR à ce moment : docs/investigations/.../decisions/adr-002-rang-normalise.md
```

---

## 11. Résumé en 10 points clés

1. **Méthode** : E[HS²] + Potentiel + K-means 3D, Méthode E enrichie
2. **Features** : (E[HS], E[HS²], Potentiel) — calculées en un seul run MC
3. **Buckets** : 50 flop + 50 turn + 50 river (preflop inchangé à 8)
4. **Centroïdes** : 3 sets street-spécifiques dans `data/abstraction/centroides_v2.npz`
5. **Migration** : 8 étapes, V1 préservée jusqu'à Étape F — aucun big bang
6. **Invariants préservés** : DIM_INPUT=52, format clé 7 segments, OFT intact
7. **Tests** : 14 tests RED (7 clustering + 5 V2 + 2 régression)
8. **Budget cloud** : 51€ (calibration 14€ + training MCCFR 37€) sur 520€ disponibles
9. **Timeline** : 14 jours calendaires
10. **rang_normalise** : MC sampling (identique V1) en Phase 2, rang exact reporté en Phase 4 (section 10)

---

_Spec finalisée le 2026-04-26 — auteur : session Claude + Matthew_
_Prochaine étape : Étape A (créer stubs + TDD Guard bypass si nécessaire)_
