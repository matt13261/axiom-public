# Audit architecture AgentAXIOM — décalage Pluribus / mode éval

**Date** : 2026-04-30
**Contexte** : suite au diagnostic P8 (V1 sans DCFR -121, P7 sans DCFR -172),
hypothèse que l'agent est mal architecturé pour Pluribus. Ce rapport audite
l'architecture réelle vs la "recette Pluribus" et identifie les trous.

**STOP après livraison** — pas d'interprétation finale, pas de modif code.

---

## 1. Composants présents dans `ai/agent.py::AgentAXIOM`

L'agent expose **6 sources de stratégie** :

| # | Composant | Attribut | Méthode de chargement | Présent dans Pluribus ? |
|---|---|---|---|---|
| 1 | Blueprint MCCFR 3-joueurs | `_blueprint` | `charger_blueprint(chemin)` | ✓ (cœur Pluribus) |
| 2 | Blueprint MCCFR Heads-Up | `_blueprint_hu` | `charger_blueprint_hu(chemin)` | partiel (Pluribus joue 6-max, pas HU) |
| 3 | **Continuation Strategies k=4** | `_blueprints_k` (dict 4 variantes) | `charger_blueprints_continuations(dossier, base_nom)` | **✓ (cœur Pluribus)** |
| 4 | **Solveur depth-limited (FLOP)** | `_solveur` | `activer_solveur(...)` | **✓ (real-time search Pluribus)** |
| 5 | **Solveur subgame (TURN/RIVER)** | `_solveur_subgame` | `activer_solveur(...)` | **✓ (real-time search Pluribus)** |
| 6 | Deep CFR (réseaux) | `_reseaux_strategie` + `_reseaux_valeur` | `charger_deep_cfr(chemin)` + `charger_reseaux_valeur(chemin)` | ✗ (extension AXIOM, pas dans Pluribus) |

Et **2 modules transverses** :
| # | Module | Rôle | Présent dans Pluribus ? |
|---|---|---|---|
| 7 | OFT — `OpponentTracker` + `ExploitMixer` | exploit adversaire faible | ✗ (extension AXIOM) |
| 8 | Heuristique fallback | si tout miss | ✗ (Pluribus garantit couverture par MCCFR + search) |

---

## 2. Composants chargés par `creer_agent()` (mode éval / self_play)

`creer_agent()` (l. 1110-1177) appelle uniquement :
- `charger_blueprint_hu(chemin_hu)` ✓
- `charger_blueprint(chemin_bp)` ✓
- `charger_deep_cfr(chemin_s)` ✓
- `charger_reseaux_valeur(base_valeur)` ✓

Il **NE charge PAS** :
- `charger_blueprints_continuations(...)` ✗
- `activer_solveur(...)` ✗

### Conséquence directe

**En P8 eval (et toutes les évals existantes via `creer_agent`), l'agent
testé n'a PAS Pluribus k=4 ni real-time search.** Le mix réel est :

```
blueprint baseline + Deep CFR (fallback) + OFT (post-mix) + heuristique (fallback final)
```

C'est un Pluribus mutilé, augmenté de Deep CFR et OFT.

---

## 3. Composants utilisés à chaque décision (`_obtenir_distribution`, l. 431-503)

Flux exact d'une décision (3-max, joueur AXIOM doit agir) :

```
choisir_action(etat, joueur, legales)
│
├── stats['total'] += 1
├── _rafraichir_variante_si_nouvelle_main(etat)        ← Pluribus k=4
│   └── Si len(self._blueprints_k) >= 2 :
│       fingerprint cartes → tirage variante uniforme parmi {baseline, fold, call, raise}
│       self._blueprint = self._blueprints_k[variante]
│       (si k=1 — cas par défaut creer_agent — RIEN)
│
├── distribution = _obtenir_distribution(etat, joueur)
│   │
│   ├── 1. Real-time search ───────────────────────────  ← Pluribus search
│   │   ├── if FLOP and self._solveur :
│   │   │       distribution = solveur_dl.resoudre(etat, joueur, self)
│   │   │       (si self._solveur is None — cas par défaut — saut)
│   │   ├── elif TURN/RIVER and self._solveur_subgame :
│   │   │       distribution = solveur_sg.resoudre(etat, joueur, self)
│   │   │       (si None — cas par défaut — saut)
│   │   └── (si self._solveur* None : skip → tentative blueprint)
│   │
│   ├── 2a. Blueprint HU ──────────────────────────────  ← MCCFR HU
│   │   if mode_hu and self._blueprint_hu : lookup
│   │
│   ├── 2b. Blueprint 3-joueurs ───────────────────────  ← MCCFR 3J ★ utilisé
│   │   if not mode_hu and self._blueprint :
│   │       cle = construire_cle_infoset(etat, joueur)
│   │       vec = _lookup_blueprint_blende(blueprint, cle, frac)
│   │       (pseudo-harmonic blending sur frontières raise)
│   │       if hit → return vec
│   │
│   ├── 3. Deep CFR ───────────────────────────────────  ← AXIOM extension ★ utilisé
│   │   if self._reseaux_strategie :
│   │       forward 3 réseaux → strat
│   │       return strat
│   │
│   └── 4. Heuristique ────────────────────────────────  ← fallback ultime
│       return _heuristique(etat, joueur)
│
├── adv = _identifier_adversaire_actif(etat, joueur)
├── if adv :
│   distribution = mixer.ajuster(distribution, adv, game_type)  ← OFT exploit
│
├── probas_legales = _mapper_sur_legales(distribution, legales, etat)
├── action = _selectionner(legales, probas_legales)
├── action = _perturber_sizing(action, etat, joueur)            ← Point 9 anti-pattern
└── return action
```

★ = composants effectivement actifs dans `creer_agent` (P8 eval).

---

## 4. Continuation strategies (Pluribus k=4) — état réel

### Fichiers présents sur disque

```
data/strategy/blueprint_v1.pkl          (2.58M infosets) ← baseline V1
data/strategy/blueprint_v1_call.pkl     (3.60M infosets) ← variante "biais call"
data/strategy/blueprint_v1_fold.pkl     (3.26M infosets) ← variante "biais fold"
data/strategy/blueprint_v1_raise.pkl    (2.78M infosets) ← variante "biais raise"
```

Les 4 fichiers existent depuis le 20-21 avril 2026. Ils ont été **entraînés**
mais ne sont **jamais chargés** en mode éval.

### Pourquoi pas chargés ?

`creer_agent()` n'appelle pas `charger_blueprints_continuations()`. Cette
méthode n'a qu'un seul appelant dans tout le repo :

| Fichier | Ligne | Contexte |
|---|---|---|
| `screen/jouer.py` | 98 | mode prod scraper Betclic |

`grep -rn "charger_blueprints_continuations" --include='*.py'` → 1 seul match
non-définitionnel. **Trou architectural** : la fonctionnalité est codée mais
non câblée dans le pipeline d'éval.

### Côté P7

Les variantes `blueprint_v1_*.pkl` sont au format V1 (incompatibles P7).
**Aucun équivalent P7 n'existe** : pas de `blueprint_p7_4m_cloud_call.pkl` etc.

---

## 5. Real-time search — état réel

### Code présent

- `solver/depth_limited.py` → `SolveurProfondeurLimitee` (FLOP)
- `solver/subgame_solver.py` → `SolveurSousJeu` (TURN/RIVER)

Tous deux migrés P7 (utilisent `PALIERS_STACK_SPIN_RUSH` + `_format_hist_avec_cap`).

### Activation

`activer_solveur()` instancie les deux. **Non appelé** dans :
- `creer_agent()` (eval / self_play)
- `training/evaluator.py`
- `training/self_play.py`

**Seul appel** : `screen/jouer.py:104` (mode prod).

### Conséquence

**Aucune des évaluations baseline n'utilise les solveurs real-time.** Pluribus
sans real-time search se réduit au blueprint MCCFR pur, ce qui matche
exactement l'expérience "P7/V1 sans DCFR catastrophique" du diag P8.

---

## 6. Comparaison "recette Pluribus" attendue vs AXIOM réel

| Composant Pluribus (Brown & Sandholm 2019) | Code présent ? | Utilisé en éval ? |
|---|---|---|
| Blueprint MCCFR (cœur) | ✓ | ✓ |
| Continuation strategies k=4 (méta-stratégie) | ✓ (dispo) | **✗ jamais chargé** |
| Real-time search FLOP (depth-limited) | ✓ (dispo) | **✗ jamais activé** |
| Real-time search TURN/RIVER (subgame) | ✓ (dispo) | **✗ jamais activé** |
| Action abstraction discretization (sizing buckets) | ✓ (`_discretiser_raise_frac` + Variante B P7) | ✓ |
| Pseudo-harmonic mapping aux frontières | ✓ (`_lookup_blueprint_blende`) | ✓ |
| Re-entraînement nightly (offline self-play) | ✗ pas implémenté | — |
| Deep CFR / réseau (PAS dans Pluribus original) | ✓ (extension AXIOM) | ✓ |
| Opponent tracking + exploit (PAS dans Pluribus original) | ✓ (OFT, extension AXIOM) | ✓ |

### Score Pluribus en éval

3 / 6 composants Pluribus chargés, dont 0 / 3 des éléments "real-time"
(continuation strategies + 2 solveurs).

**L'agent évalué en P8 est un Pluribus à 50%** — le blueprint MCCFR seul, sans
les couches qui font sa robustesse face à des adversaires structurés.

---

## 7. Trous architecturaux identifiés

### TA1 — `creer_agent` n'active pas les composants Pluribus

**Sévérité** : **majeure**. Toutes les évaluations baseline (P1, P8, futures)
utilisent un agent tronqué. Les chiffres P8 (V1 +20.32, P7 -24.56) ne reflètent
pas la performance "AXIOM en mode prod".

**Impact** : sous-estimation systématique de la performance MCCFR pur ;
incompatibilité entre régression d'éval et comportement réel sur Betclic.

**Fix** : ajouter à `creer_agent` les appels `charger_blueprints_continuations`
(si fichiers présents) et `activer_solveur` (avec params raisonnables).
Ou créer un mode "creer_agent_complet" explicite.

### TA2 — Pas de continuation strategies P7

**Sévérité** : moyenne (n'affecte pas l'agent V1 en mode prod, mais si on
veut tester P7 dans des conditions Pluribus complètes, il faut générer
`blueprint_p7_*_call.pkl`, etc).

**Impact** : aucune éval P7 dans des conditions Pluribus complètes possible
aujourd'hui, même en activant TA1.

**Fix** : re-train cloud P7 avec biais Continuation Strategies (4 variantes
× 4M iter = ~50€ cloud, faisable mais coûteux).

### TA3 — Solveurs real-time peut-être lents pour 60K mains d'éval

**Sévérité** : faible mais bloquante en pratique. `activer_solveur` use 3s
budget par décision. Sur 60K mains × ~10 décisions/main = 600K décisions →
~500h de wall clock. **Inutilisable pour les évals automatisées.**

**Impact** : même en activant TA1, on ne peut pas faire des évals 60K mains
avec real-time search à 3s/décision.

**Fix** : créer un mode "solveur rapide" (budget 0.1-0.3s) pour les évals,
et garder le mode 3s pour la prod. Ou désactiver les solveurs en éval (ce
qui est le cas actuel) — auquel cas on est conscient qu'on évalue un agent
réduit.

### TA4 — Deep CFR appelé en fallback uniquement

**Sévérité** : faible (architectural). Deep CFR est positionné comme
"plan B" du blueprint, pas comme oracle de valeur indépendant. Pas une
critique, juste à noter.

---

## 8. Recommandations concrètes (sans implémenter)

### Priorité 1 — Fixer le trou TA1

Modifier `creer_agent` pour activer les composants Pluribus si fichiers
disponibles. Ou bien créer une fonction parallèle `creer_agent_pluribus`.
Conserver `creer_agent` minimal pour rétrocompat.

Coût : ~20 min code + tests. **Aucun cloud.**

### Priorité 2 — Re-tester P8 avec agent Pluribus complet

Une fois TA1 fixé, relancer l'éval V1 + P7 :
- V1 + continuations (`blueprint_v1_*.pkl`) + solveur rapide
- P7 (sans continuations, on n'en a pas) + solveur rapide

Comparer aux chiffres P8 actuels. Cela nous dira si le déficit P7 vient
réellement de l'abstraction ou du fait qu'on évaluait un agent tronqué
des deux côtés.

Coût : 0€, ~2h éval.

### Priorité 3 — Si P7 reste mauvais → re-train P7 avec continuations (TA2)

Cloud re-train 4 variantes P7 × 4M iter chacune = ~50€, ETA ~30h cumul (les
4 peuvent tourner en parallèle si VM séparées). Ne faire que si Priorité 2
montre que les continuations comblent l'écart pour V1 et qu'on en a besoin.

### Priorité 4 — Si nécessaire, re-train Deep CFR sous P7

Pluribus n'utilise pas Deep CFR. Si on veut reproduire la "recette pure
Pluribus", on peut DÉSACTIVER Deep CFR par défaut et compter sur
continuations + solveurs. Mais le diag P8 montre que désactiver DCFR
catastrophique → on garde DCFR comme filet de sécurité, en parallèle.

---

## 9. État technique

| Item | État |
|---|---|
| Branche actuelle | `main` |
| Commits | aucun (audit lecture seule) |
| Fichiers modifiés | aucun |
| Tests | non touchés |
| Blueprints | intacts |

---

## 10. STOP

Audit livré. Trois trous architecturaux dont un majeur (TA1 — `creer_agent`
incomplet). Aucune décision implémentée.

À toi pour arbitrer entre :
- (a) Fixer TA1, re-éval, et voir si P7 reste en déficit
- (b) Lancer cloud re-train P7 avec continuations directement
- (c) Autre piste

Je n'implémente rien jusqu'à ton GO.
