# TODO — Session 2026-04-26

## En cours (Phase 2 — Abstraction V2)

### Immédiat (prochaine session)
- [ ] **Étape A** : créer `AbstractionCartesV2` stub dans `card_abstraction.py`
      + créer `abstraction/card_clustering.py` avec stubs
      (nécessite probablement TDD Guard bypass #5 — multi-fichiers bootstrap)
- [ ] **Étape B** : écrire les 14 tests RED
      - `tests/test_card_clustering.py` (7 tests)
      - ajouter tests 8-14 dans `tests/test_abstraction.py`
      Confirmer : nouveaux tests RED + 141 existants GREEN

### Suite (dans l'ordre)
- [ ] Étape C : implémenter `card_clustering.py` (compute_features, fit_centroids, predict)
- [ ] Étape D : implémenter `AbstractionCartesV2` complet + LRU cache
- [ ] Étape E : script `recalibrer_3max_v2.py` + run local validation + run cloud

## Backlog (hors Phase 2)
- [ ] Créer compte Google Cloud (guide dans done/2026-04-25.md)
- [ ] Lancer Continuation Strategies k=4 (infrastructure prête, voir TODO.txt)

## Terminé cette session
- [x] Étape 1 : analyse code existant (rapport synthèse abstraction V1)
- [x] Étape 2 : brainstorm méthodes bucketing → sauvegardé brainstorm.md
- [x] Étape 3 : spec P6 complète → sauvegardée spec.md
- [x] GitHub hybride : axiom-private + axiom-public + sync hook ✅
- [x] Décision Blueprint HU : HU hors scope, documenté SPRINT.md
