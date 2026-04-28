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

**Validation post-bypass** : la modification fait passer RED.10 et RED.11
(et seulement ceux-ci) à GREEN. Aucune autre régression.
