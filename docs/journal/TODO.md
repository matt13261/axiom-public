# TODO — Session 2026-04-26

## Immédiat (prochaine session — Étape E)
- [x] **Étape C** : implémenter `card_clustering.py` — TERMINÉ (4 commits, 161 tests GREEN)
      - C.1 : `compute_features` MC réelle (B.13 ✓)
      - C.2 : `fit_centroids` sklearn KMeans (B.14 ✓)
      - C.3 : `predict_bucket` argmin L2 (B.15 ✓)
      - C.4 : tests intégration pipeline (B.16/B.17 ✓)
      - fix : potentiel négatif (B.18 ✓)

- [x] **Étape D** : implémenter `AbstractionCartesV2` complète — TERMINÉ (5 commits, 166 tests GREEN)
      - D.1 : __init__ avec centroides= dict direct ✓
      - D.2 : bucket_postflop réelle (compute_features + predict_bucket) ✓
      - D.3 : bucket_preflop délègue à V1 ✓
      - D.4 : cache dict (26000x speedup sur cache hit) ✓
      - D.5 : régression OFT + V2 réelle ✓

- [ ] **Étape E** : script `recalibrer_3max_v2.py` + calibration locale + cloud
      - Générer dataset features (N mains × 3 streets × n_sim=200)
      - fit_centroids flop/turn/river → 50 centroïdes chacun
      - Sauvegarder data/abstraction/centroides_v2.npz
      - Budget cloud : ~37€ (Étape G)

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
- [x] Étape C.1-C.4 + fix : vraie implémentation card_clustering.py (161 tests GREEN)
      - compute_features MC réelle : E[HS], E[HS²], Potentiel via treys Evaluator
      - Correction critique : potentiel peut être négatif (overpair perd de la valeur)
      - fit_centroids : sklearn KMeans(n_init=10, random_state=seed)
      - predict_bucket : argmin(linalg.norm(centroids - vec, axis=1))
      - 6 nouveaux tests B.13-B.18
- [x] Étape D.1-D.5 : AbstractionCartesV2 complète (166 tests GREEN)
      - bucket_postflop : MC + K-means + cache dict (26000x speedup hit vs miss)
      - bucket_preflop : délégation V1 (préflop inchangé)
      - 5 nouveaux tests D.1-D.5
- [x] Zéro bypass TDD Guard cette session (2 corrections de cycle acceptées)
