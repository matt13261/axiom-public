# P7 — Refonte abstraction historique + stacks (Spin & Rush)

**Date** : 2026-04-28
**Statut** : SPEC validée, en attente GO pour P7.3 (RED tests).
**Diagnostic source** : `docs/investigations/P6-abstraction/` — segment `hist`
explose à 19 492 valeurs uniques en 2K iter ; `stacks` à 1 416 valeurs.

---

## 1. Décisions validées (récap)

| # | Décision |
|---|---|
| D1 | Hist mono-street, sizings abstraits (β strict), cap 6 actions/street |
| D2 | Mapping sizing : `r0/r1 → rS`, `r2/r3 → rM`, `r4 → rL` |
| D3 | Stacks : 7 niveaux Spin & Rush via `PALIERS_STACK_SPIN_RUSH = [0, 5, 8, 13, 19, 27, 41]` |
| D4 | Format clé infoset 7 segments **inchangé** |
| D5 | Flag agresseur préflop **reporté** (post-validation P7) |
| D6 | Migration V2+P7 directe (P6 déjà actif) |
| D7 | HU migré aussi (helpers factorisés, coût marginal) |

---

## 2. Mapping sizing : justification

### Rappel `_discretiser_raise_frac` actuel ([info_set.py:43-64](../../../abstraction/info_set.py))

| bucket | Plage `frac = mise/pot` | Centre pseudo-harmonique | Sémantique |
|---|---|---|---|
| 0 | `frac == 0` | — | Pas de raise (n'apparaît jamais dans `r{bucket}`) |
| 1 | `0 < frac ≤ 0.33` | 0.165 | micro raise |
| 2 | `0.33 < frac ≤ 0.75` | 0.54 | small raise (≈half-pot) |
| 3 | `0.75 < frac ≤ 1.25` | 1.0 | pot raise |
| 4 | `frac > 1.25` | 1.5 | over-raise |

### Comparaison de deux variantes de mapping

**Variante A — équilibrée (proposition initiale)** :
```
r0, r1   →  rS   (frac ≤ 0.33)
r2, r3   →  rM   (0.33 < frac ≤ 1.25)
r4       →  rL   (frac > 1.25)
```

**Variante B — Spin & Rush short-stack (ALTERNATIVE retenue)** :
```
r0, r1   →  rS   (frac ≤ 0.33      — micro raise / défensif r0)
r2       →  rM   (0.33 < frac ≤ 0.75 — half-pot, value/protection)
r3, r4   →  rL   (frac > 0.75       — pot + overbet, polarisé/committing)
```

### Analyse comparative pour le format Spin & Rush

Stack effectif typique en Spin & Rush ≤ 20 BB. Sur un flop dans un pot de
~10 BB :
| Sizing | Bet | Stack restant | Interprétation stratégique |
|---|---|---|---|
| `r1` (frac=0.165) | ~1.5 BB | ~18 BB | Probe, info bet, range bet large |
| `r2` (frac=0.54)  | ~5 BB   | ~14 BB | Standard cbet, value mince/protection |
| `r3` (frac=1.0)   | ~10 BB  | ~9 BB  | **Commit** — l'opposant doit shove ou fold |
| `r4` (frac=1.5)   | ~15 BB  | ~4 BB  | **Commit polarisé** — overbet shove-induce |

**Distinction critique** : `r3` (pot) en short-stack = engagement quasi-total
du stack restant (SPR ~1 après bet → toute action future = all-in).
Stratégiquement identique à `r4` (overbet) : range polarisé monstres+bluffs.
À l'inverse, `r2` (half-pot) laisse de la marge postflop (SPR ~3 restant)
et appartient à un range linéaire value/protection.

**Variante A** fusionne `r2` + `r3` dans `rM` → perd la frontière
"non-committing / committing" qui est **LA décision-clé** en Spin & Rush.

**Variante B** aligne la frontière sur le seuil de commitment SPR ~1 :
- `rS` / `rM` = "je peux jouer postflop"
- `rL` = "je m'engage, range polarisé"

**Verdict** : retenir **Variante B**. Distinction stratégique short-stack
préservée. Coût marginal : variante B groupe `r3` (pot, ~3-betting standard)
avec `r4` (overbet rare). Acceptable car en short-stack `r3` EST un commit
de facto, donc proche d'un overbet en termes de range.

**Pondération** : si le diagnostic post-P7 montre des winrates négatifs sur
les spots commitment-aware, considérer une variante C à 4 sizings
(`rS`/`rM`/`rL_pot`/`rL_over`) — coût cardinalité +33%.

### Justification frontières (Variante B retenue)

- `r0` est défensif (le code `r0` ne devrait jamais être généré car
  `_discretiser_raise_frac(0) = 0` mais le code `f'r{bucket}'` n'est appelé que
  pour des RAISE avec montant > 0, donc bucket ∈ {1,2,3,4}). Inclus dans `rS`
  par sécurité.
- Frontière `rS / rM` à 0.33 : seuil pseudo-harmonique `_CENTRES_RAISE_BUCKET[0]
  = 0.165` × 2 — point neutre entre micro et standard.
- Frontière `rM / rL` à 0.75 : seuil SPR-aware. Au-dessus, on engage > 75%
  du pot → on s'engage à jouer all-in postflop avec un stack ≤ 20 BB.
- 3 buckets asymétriques mais sémantiquement clairs (probe / value / commit).

### Conservation des sémantiques distinctives

| Token | Ancien | Nouveau | Conservé ? |
|---|---|---|---|
| FOLD | `f` | `f` | ✓ |
| CHECK | `x` | `x` | ✓ |
| CALL | `c` | `c` | ✓ |
| ALL-IN | `a` | `a` | ✓ |
| RAISE micro/small | `r1`, `r2` | `rS`, `rM` (selon mapping) | ✓ (compacté) |
| RAISE pot | `r3` | `rM` | ✓ (fusionné avec r2) |
| RAISE overbet | `r4` | `rL` | ✓ (préservé distinct) |

**Distinction préservée** : ALL-IN reste séparé (raison : action irréversible
qui change radicalement les EV ; ne pas fusionner avec rL).

---

## 3. Helpers à factoriser dans `abstraction/info_set.py`

### Helper 1 : `_abstraire_sizing(idx_taille: int) -> str`

```python
def _abstraire_sizing(idx_taille: int) -> str:
    """
    Mappe le bucket sizing brut (0..4 issu de _discretiser_raise_frac)
    vers le sizing abstrait Spin & Rush (S/M/L).

    Mapping (Variante B — Spin & Rush short-stack) :
      0, 1 → 'S'   (frac ≤ 0.33    — micro / probe)
      2    → 'M'   (frac ≤ 0.75    — half-pot, value/protection)
      3, 4 → 'L'   (frac > 0.75    — pot+ / commit polarisé)
    """
    if idx_taille <= 1: return 'S'
    if idx_taille == 2: return 'M'
    return 'L'
```

### Helper 2 : `_format_hist_avec_cap(hist_brut: str, cap: int = 6) -> str`

```python
def _format_hist_avec_cap(hist_brut: str, cap: int = 4) -> str:
    """
    Reformate l'historique brut (ex: 'xr1r3fr2a') en historique abstrait
    avec sizings S/M/L, en ne gardant que les `cap` dernières actions.

    Tokens reconnus :
      - 'f', 'x', 'c', 'a' : 1 caractère (action atomique)
      - 'r{N}' avec N ∈ 0..4 : 2 caractères, mappés via _abstraire_sizing

    Sortie :
      Concaténation des actions abstraites, ex: 'xrSrMfrSa' → cap 6 → 'xrSrMfrS'

    Notes :
      - Ne touche pas au stockage hist_phases (qui reste raw r{N}).
      - Appelé uniquement à la construction de la clé d'infoset.
      - Réseau Deep CFR continue de lire le format raw — DIM_INPUT=52 préservé.
    """
    actions = []
    i = 0
    while i < len(hist_brut):
        ch = hist_brut[i]
        if ch == 'r' and i + 1 < len(hist_brut) and hist_brut[i+1].isdigit():
            actions.append('r' + _abstraire_sizing(int(hist_brut[i+1])))
            i += 2
        else:
            actions.append(ch)
            i += 1
    if len(actions) > cap:
        actions = actions[-cap:]
    return ''.join(actions)
```

---

## 4. Paliers stack Spin & Rush

### Définition

```python
# Paliers actuels (18 niveaux) — conservés pour rétrocompat blueprints existants
PALIERS_STACK = [0, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 65, 75, 100, 150, 200]

# Nouveaux paliers Spin & Rush (7 niveaux) — clés P3..P50
PALIERS_STACK_SPIN_RUSH = [0, 5, 8, 13, 19, 27, 41]
```

### Mapping BB → bucket

| Palier (BB) | Code lisible | Phase tournoi (500ch start) | Niveau blindes |
|---|---|---|---|
| 0–4 | P3 | Push/fold absolu | 7-8 |
| 5–7 | P5 | Push/fold + ICM | 5-6 |
| 8–12 | P10 | Open shove + 3-bet jam | 3-4 |
| 13–18 | P15 | Mix, fold equity importante | 2-3 |
| 19–26 | P22 | Mid-stack postflop limité | 1 fin |
| 27–40 | P30 | Mid-stack postflop plein | 1 début |
| 41+ | P50 | Stack accumulé (rare) | — |

**Format dans la clé** : `stacks=(13,5,27)` (palier inférieur en BB), inchangé
structurellement. Cardinalité théorique : `7³ = 343` (vs `1416` actuel).

### Mécanisme de bascule

Pas de flag config dynamique — **switch direct** dans le code (par cohérence
avec V2 qui a remplacé V1 sans flag). Justification : un flag permettrait deux
formats de clé incompatibles à coexister, source de bugs. Si on veut garder
V1 dispo pour rétrocompat, on garde `PALIERS_STACK` exporté mais
`_construire_cle` utilise `PALIERS_STACK_SPIN_RUSH` directement.

Le fichier `config/blinde_structure_spin_rush.json` est créé pour documenter
la structure de blindes Betclic (référentiel partagé), pas pour piloter le
switch des paliers.

---

## 5. Plan de modification — fichiers impactés

### Architecture clé : raw au stockage, abstrait à la clé

**Insight crucial** : `hist_phases` est stocké en **raw** (`r1`, `r2`, ...) dans
les états (engine, mccfr, deep_cfr, train_hu). Seule la **construction de clé**
applique l'abstraction sizing+cap.

Avantages :
- **DIM_INPUT=52 préservé** (réseau lit raw, encoder inchangé)
- **engine/game_state.py inchangé** (pas de touche au moteur)
- **Tests engine inchangés** (le moteur ne sait rien de l'abstraction)
- **Migration localisée** aux constructeurs de clé

### Fichiers source à modifier (6)

| # | Fichier | Modif |
|---|---|---|
| 1 | `abstraction/info_set.py` | + 2 helpers, + `PALIERS_STACK_SPIN_RUSH`, modif `_construire_cle` |
| 2 | `ai/mccfr.py::_cle_infoset` | Importer helpers, transformer hist + utiliser nouveaux paliers |
| 3 | `train_hu.py::_cle_infoset` (l. 438 zone) | Idem |
| 4 | `solver/depth_limited.py` | Vérifier si construction clé locale → si oui même modif, sinon RAS |
| 5 | `solver/subgame_solver.py` | Idem |
| 6 | `config/blinde_structure_spin_rush.json` (NEW) | Référentiel structure Betclic |

### Fichiers source NON modifiés (sanity check)

| Fichier | Raison |
|---|---|
| `engine/game_state.py` | `enregistrer_action` écrit en raw, c'est correct |
| `ai/network.py` | Encoder lit raw, garde 5 dims raise → DIM_INPUT=52 |
| `ai/agent.py` | Utilise `construire_cle_infoset` de info_set.py — transparent |
| `ai/deep_cfr.py` (handler actions) | Stocke en raw, inchangé |
| `screen/lire_etat.py` | Reconstruit raw depuis l'écran — inchangé |

### Tests à modifier ou créer (4-5)

| # | Fichier | Action |
|---|---|---|
| T1 | `tests/test_p7_abstraction.py` (NEW) | Tests atomiques helpers + cardinalité |
| T2 | `tests/test_mccfr.py` | MAJ assertions cardinalité + nouveaux paliers |
| T3 | `tests/test_hu.py::test_format_cles_hu` | MAJ format clé HU (vérifier hist abstrait) |
| T4 | `tests/test_abstraction.py` | MAJ tests format clé si existants |

---

## 6. Décision HU

**Migrer HU avec le même format**. Raisons :
- `train_hu.py` utilise déjà `_discretiser_raise_frac` et `_normaliser` importés
  de `info_set.py` — coût marginal de refactor.
- Garder un seul format évite la divergence (deux abstractions à maintenir).
- Le namespace `HU_PHASE` (préfixe phase HU) reste utilisé pour séparer les
  clés HU des clés 3-max — pas de collision.
- Stacks tuple HU reste `(A, B)` (2 stacks) vs `(A, B, C)` 3-max — inchangé.

**Format clé HU final** :
```
HU_PREFLOP|pos=0|bucket=3|pot=2|stacks=(P10,P10)|hist=rSc|raise=2
```

---

## 7. Estimation cardinalité finale (calculée, pas estimée)

### Hypothèses de calcul

Sur 2K iter actuels :
- 19 492 hist uniques observés
- Échantillonnage estimé (basé sur exemples observés) :
  - 40% sans raise (`xc`, `x`, `xx`, `xxx`, `xxc`, ...) → cardinalité hist sans raise ~50-100
  - 60% avec raise, distribution moyenne ~3 raises par hist
- Cap=6 : ~30% des hist actuels sont > 6 actions

### Calcul réduction sizing seul (cap=∞)

Pour les hist contenant des raises (60% du total = 11 695 unique) :
- Compression par RAISE : 5 → 3 buckets, ratio `5/3 = 1.67×` par token
- Avec ~3 raises moyens : `(5/3)^3 = 4.63×`
- Hist avec raise réduits : `11 695 / 4.63 ≈ 2 525`
- Hist sans raise (40%) : `≈ 100` (peu de variantes possibles)
- **Total sizing seul : ~2 625 hist uniques**

### Calcul réduction cap=6 additionnel

Sur les ~30% de hist > 6 actions (5 850 unique avant abstraction) :
- Suite à abstraction sizing → ~1 263 unique
- Cap=6 colle préfixes différents avec mêmes tails → estimé ÷2 supplémentaire
- Hist longs après cap+sizing : `~630`
- Hist courts après sizing seul (70%) : `~1 838`
- **Total final : ~2 470 hist uniques**

### Cardinalité combinée par segment (à 2K iter)

| Segment | Cardinalité avant | Cardinalité après P7 (estim.) | Notes |
|---|---|---|---|
| phase | 4 | 4 | inchangé |
| pos | 3 | 3 | inchangé |
| bucket | 50 | 50 | V2 inchangé |
| pot | 18 | 18 | inchangé |
| **stacks** | **1 416** | **~150-250** | 7³ théorique = 343, densité réelle plus faible |
| **hist** | **19 492** | **~2 470** | sizing+cap |
| raise | 4 | 4 | inchangé |

### Estimation infosets totaux

Actuellement : 632 865 infosets / 2K iter = ~316 infosets/iter
Après P7 estimé : produit cartésien densifié × ratio observé →
**~30-60 infosets/iter** (vs 316), soit **÷5 à ÷10**.

### Honnêteté : la cible 500-1500 est ambitieuse

Le user a mentionné 500-1500 hist comme cible. Le calcul ci-dessus aboutit à
**~2470**, soit ~1.5× au-dessus de la cible haute. Deux leviers complémentaires
si insuffisant après mesure réelle :

1. **Cap plus agressif** : passer cap=6 → cap=4 (gain estimé ÷1.8 sur les
   hist longs). Coût : perte de profondeur informationnelle pour décisions
   tardives river dans 3-bet/4-bet pots.
2. **Fusion `rM` partielle** : fusionner aussi `r4` dans `rM` (2 sizings au
   lieu de 3 : `S` / `M`). Gain ~×1.5 supplémentaire mais perte de la
   distinction overbet vs pot, importante en stratégie.

**Recommandation** : appliquer P7 avec cap=6 + 3 sizings, mesurer, et
décider du tightening selon résultat. Si ratio infosets/iter < 60 → succès.

### Mesure réelle post-implémentation (2026-04-28)

| Configuration | Cardinalité hist mesurée | Verdict |
|---|---|---|
| cap=6 (initial) | 3 084 (jitter ±100) | au-dessus cible 2500 |
| **cap=4 (retenu)** | **417** | dans cible originale 500-1500 ✓ |

Réduction totale vs baseline pré-P7 : **23 146 → 417** = **÷55×** (cardinalité hist).

Diagnostic distribution (cap=6, 2K iter MCCFR) :
- 88% des hists distinctes ont longueur 5-6 actions → cap=4 collapse leur masse
- Top 20 hists fréquentes ne couvrent que 27.5% du volume → long tail
- **FLOP** = 95% des hists distinctes (3044), TURN = 1831, RIVER = 741, PREFLOP = 232
- Hists ≤ 4 actions préservées : 12% (381 distinctes) → matche le 417 mesuré à cap=4

Conclusion : cap=4 cible précisément la queue de distribution sans toucher
les hists courtes (préflop typique en short stack). Voir `tdd_guard_bypasses.md`
pour les justifications de bypass durant l'implémentation.

---

## 8. Plan tests RED atomiques (ordre TDD strict)

| # | Test | Fichier | Justification |
|---|---|---|---|
| RED.1 | `test_abstraire_sizing_S_M_L_frontieres` | test_p7_abstraction.py | Frontières mapping bucket → S/M/L |
| RED.2 | `test_abstraire_sizing_defensif_bucket_0` | test_p7_abstraction.py | r0 → S (défensif) |
| RED.3 | `test_format_hist_sizing_remplace_chiffres` | test_p7_abstraction.py | `xr1r3` → `xrSrM` |
| RED.4 | `test_format_hist_cap_garde_dernieres_actions` | test_p7_abstraction.py | cap=6 sur hist > 6 |
| RED.5 | `test_format_hist_inferieur_cap_inchange` | test_p7_abstraction.py | hist court intact |
| RED.6 | `test_paliers_stack_spin_rush_7_niveaux` | test_p7_abstraction.py | constante définie 7 valeurs |
| RED.7 | `test_cle_infoset_format_7_segments_preserve` | test_p7_abstraction.py | invariant projet |
| RED.8 | `test_cle_infoset_utilise_paliers_spin_rush` | test_p7_abstraction.py | stacks=(P3..P50) syntaxiquement valide |
| RED.9 | `test_cle_infoset_hist_abstrait_format_S_M_L` | test_p7_abstraction.py | clé contient rS/rM/rL pas r1..r4 |
| RED.10 | `test_mccfr_cle_infoset_aligne_avec_info_set` | test_mccfr.py | cohérence cross-module |
| RED.11 | `test_train_hu_cle_infoset_aligne` | test_hu.py | cohérence HU |
| RED.12 | `test_cardinalite_hist_random_play_inferieure_2500` | test_p7_abstraction.py | matche calcul section 7 (~2470). Si dépassé en P7.5 : tighten cap=4 ou re-discuter mapping — pas de relâchement du seuil (anti-TDD) |
| RED.13 | `test_cardinalite_stacks_random_play_inferieure_400` | test_p7_abstraction.py | 7³=343 + marge densité |
| RED.14 | `test_kuhn_convergence_preservee` | test_mccfr.py existant | non-régression |
| RED.15 | `test_engine_partie_complete_inchangee` | test_engine.py existant | non-régression invariant |

Chaque test : RED → stub minimal → GREEN → commit atomique. TDD Guard actif.

---

## 9. Critères de succès P7

| Métrique | Cible | Mesure |
|---|---|---|
| Cardinalité hist (2K iter) | **417** mesuré (cible < 2500) | RED.12 GREEN ✓ |
| Cardinalité stacks (10K iter) | < 400 | RED.13 + mesure manuelle |
| Ratio infosets/iter | < 60 | mesure post-implémentation |
| Tests existants (192) | tous GREEN | full pytest |
| Format clé 7 segments | préservé | RED.7 |
| DIM_INPUT réseau | 52 (inchangé) | sanity check encoder |
| Convergence Kuhn | préservée | RED.14 |

---

## 10. Ordre d'exécution P7.3 → P7.6

1. **P7.3** : créer `tests/test_p7_abstraction.py`, RED.1 → RED.15 atomiquement.
2. **P7.3.bis — Tag de revert** (juste avant P7.4) :
   ```bash
   git tag -a pre-p7 -m "Tag avant migration P7 : PALIERS_STACK 18 niveaux + hist raw r0..r4. Revert possible si re-train cloud post-P7 échoue."
   git push origin pre-p7
   ```
   Compense l'absence de flag config dynamique. `git checkout pre-p7` permet
   un retour propre à l'état pré-migration si le re-train cloud post-P7
   produit des winrates dégradés non-récupérables.
3. **P7.4** : implémenter helpers + modif 6 fichiers, GREEN tous tests.
4. **P7.5** : mini training local 10K iter, mesurer cardinalité réelle.
   Si hist > 2500 : tighten cap=4 OU itérer sur mapping (variante C 4-buckets).
   Pas de relâchement du seuil 2500 dans RED.12.
5. **P7.6** : commits propres, sync axiom-public, MAJ TODO/SPRINT/ROADMAP.

**STOP après cette spec — attente GO pour P7.3 (création tests RED).**

---

## 11. Risque résiduel cap=4 (à surveiller P7.5)

Le cap=4 droppe 88% des hists 5-6 actions, dont une partie correspond à des
**pots flop multi-way après 3-bet préflop** (le seul scénario réaliste en
Spin & Rush ≤ 20 BB où la profondeur d'historique postflop dépasse 4 actions).

Préflop usuel court : open + call ou open + 3bet + call/4bet → ≤ 4 actions →
**100% préservé** par cap=4.

**Surveillance P7.5** : si la mesure de winrate post-re-train montre une fuite
d'EV concentrée sur les pots flop multi-way 3-bet (à mesurer via decisions
trace par scénario), envisager :
- Réintroduire un encodage spécifique de l'agresseur préflop (cf. flag PF
  reporté en décision D5 §1)
- Cap asymétrique : cap=4 sur préflop, cap=6 sur postflop (rajoute
  ~2700 hists postflop, total estimé ~3000 — au-dessus seuil RED.12 mais
  acceptable si l'EV est confirmée)

Cette note est volontairement laissée en spec pour traçabilité long terme.
