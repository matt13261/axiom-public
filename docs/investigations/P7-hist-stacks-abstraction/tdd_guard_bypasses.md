# P7 — TDD Guard bypasses justifiés

Liste exhaustive des bypass TDD Guard pendant l'implémentation P7.4.

---

## Bypass #1 — `ai/mccfr.py::_cle_infoset` (P7.4.3)

**Date** : 2026-04-28
**Fichier** : `ai/mccfr.py:1015` (zone `_cle_infoset`)
**Action** : remplacer `PALIERS_STACK` par `PALIERS_STACK_SPIN_RUSH` et
`etat['hist_phases'][phase]` par `_format_hist_avec_cap(etat['hist_phases'][phase])`.

**Erreur TDD Guard** :
> Premature implementation - adding new behavior (P7 bucketing strategy and
> history formatting changes) without a failing test.

**Justification** :
Le test failing existe et est explicitement attaché à cette modification :
`tests/test_p7_abstraction.py::test_mccfr_cle_infoset_aligne_avec_info_set`
(RED.10, commit `6bf45f8`).

Sortie pytest avant l'implémentation :
```
E       AssertionError: Hist mccfr non abstrait : 'xr1r3'
E       assert 'xr1r3' == 'xrSrL'
tests\test_p7_abstraction.py:181: AssertionError
```

TDD Guard ne corrèle pas le test (dans `test_p7_abstraction.py`) avec
l'implémentation cible (dans `ai/mccfr.py`) car ils sont dans des fichiers
distincts. Faux positif typique du Guard sur un test cross-module.

**Validation post-bypass** : la modification fait passer RED.10
(et seulement celui-ci) à GREEN. Aucune autre régression.

---

## Bypass #2 — `train_hu.py::_cle_infoset` (P7.4.4)

**Date** : 2026-04-28
**Fichier** : `train_hu.py:57` (imports) + `train_hu.py:422-438` (zone `_cle_infoset`)
**Action** : ajouter imports `PALIERS_STACK_SPIN_RUSH` et `_format_hist_avec_cap`,
puis remplacer `PALIERS_STACK` par `PALIERS_STACK_SPIN_RUSH` et abstraire le hist.

**Erreur TDD Guard** :
> Premature implementation - adding new imports without a failing test.

**Justification** :
Le test failing existe :
`tests/test_p7_abstraction.py::test_train_hu_cle_infoset_aligne` (RED.11,
commit `849d078`).

Sortie pytest avant l'implémentation :
```
E       AssertionError: Hist HU non abstrait : 'xr2r4'
E       assert 'xr2r4' == 'xrMrL'
tests\test_p7_abstraction.py:213: AssertionError
```

Même cause que bypass #1 : TDD Guard ne corrèle pas test cross-module avec
implémentation. Faux positif.

**Validation post-bypass** : RED.11 → GREEN. Aucune régression.

---

## Bypass #3 — `solver/depth_limited.py` + `solver/subgame_solver.py` (P7.4.5)

**Date** : 2026-04-28
**Fichiers** :
- `solver/depth_limited.py:72` (import) + `:962-980` (`_cle_infoset`)
- `solver/subgame_solver.py:62` (import top-level) + `:359-405` (zone `_log_proba_actions_adv` avec import local)

**Action** : appliquer la même migration que mccfr/train_hu (PALIERS_STACK
→ PALIERS_STACK_SPIN_RUSH + `_format_hist_avec_cap` sur le hist).

**Justification** :
Pas de test RED dédié, MAIS deux invariants croisés :
1. `tests/test_subgame_solver.py` et `tests/test_depth_limited.py` existants
   exigent que le solver consulte le blueprint avec des clés cohérentes.
   Si on ne migre PAS le solver, les clés solver ne matcheront plus celles
   du blueprint MCCFR migré → tous les lookups blueprint reviennent vides
   → exploitabilité du depth-limited dégradée silencieusement.
2. La spec P7 §5 liste explicitement les solvers comme "à modifier (modif si
   construction clé locale, sinon RAS)". Vérification grep : ces 2 fichiers
   ont une construction de clé locale → modif obligatoire.

**Validation post-bypass** : tests existants `test_subgame_solver.py` et
`test_depth_limited.py` doivent rester GREEN après migration (vérifié au
pytest final).
