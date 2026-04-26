# TODO — Session 2026-04-26

## Immédiat (prochaine session — Étape D)
- [x] **Étape C** : implémenter `card_clustering.py` — TERMINÉ (4 commits, 160 tests GREEN)
      - C.1 : `compute_features` MC réelle (B.13 ✓)
      - C.2 : `fit_centroids` sklearn KMeans (B.14 ✓)
      - C.3 : `predict_bucket` argmin L2 (B.15 ✓)
      - C.4 : tests intégration pipeline (B.16/B.17 ✓)

- [ ] **Étape D** : implémenter `AbstractionCartesV2` complète
      - Charger centroides, LRU cache, bucket() unifié, bucket_et_equite()
      - Test B.8/B.9 doivent passer avec vraie logique (pas rank-stub)
      - bucket_postflop : compute_features + predict_bucket (+ centroides depuis fichier)
      - bucket_preflop  : logique Chen/bisect existante (réutiliser V1)

## Suite (dans l'ordre)
- [ ] Étape E : script `recalibrer_3max_v2.py` + calibration locale + cloud
- [ ] Étape F : bascule modules (agent, mccfr, deep_cfr, solver)
- [ ] Étape G : training blueprint cloud (37€)
- [ ] Étape H : validation + merge

## Backlog
- [ ] Créer compte Google Cloud (guide dans done/2026-04-25.md)
- [ ] Continuation Strategies k=4 (infrastructure prête, voir TODO.txt)

## Terminé cette session
- [x] Section 10 spec.md : décision rang_normalise documentée (MC vs exact)
- [x] Étape A.1 : stub compute_features + 1 test (142 total)
- [x] Étape A.2 : stub AbstractionCartesV2 + 1 test (143 total)
- [x] Étape B.1-B.12 : 12 tests atomiques TDD (155 total)
      - 6 GREEN immédiats (B.1/B.2/B.4/B.6/B.8*/B.12)
      - 5 nécessitaient stub + TDD Guard cycles (B.3/B.5/B.7/B.9/B.10/B.11)
      - 1 correction test (B.12 : adversaire= → seat_index=)
- [x] Étape C.1-C.4 : vraie implémentation card_clustering.py (160 tests GREEN)
      - compute_features MC réelle : E[HS], E[HS²], Potentiel via treys Evaluator
      - fit_centroids : sklearn KMeans(n_init=10, random_state=seed)
      - predict_bucket : argmin(linalg.norm(centroids - vec, axis=1))
      - 5 nouveaux tests B.13-B.17 (dont B.16/B.17 GREEN immédiatement)
- [x] Zéro bypass TDD Guard cette session
