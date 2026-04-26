# TODO — Session 2026-04-26

## Immédiat (prochaine session — Étape C)
- [ ] **Étape C** : implémenter `card_clustering.py` pour rendre les tests réels GREEN
      Fonctions à implémenter (dans l'ordre TDD atomique) :
      1. `rang_normalise(cartes, board_complet, nb_adversaires)` — MC sampling
      2. `completer_board_aleatoire(cartes, board, rng, cibles=5)`
      3. `completer_board_jusqu_river(cartes, board, rng)`
      4. `compute_features` — vrai calcul E[HS], E[HS²], Potentiel
      5. `fit_centroids` — vrai k-means sklearn (n_init=20, random_state)
      6. `predict_bucket` — argmin distance L2 euclidienne
      Objectif final : tests B.2/B.3/B.4 passent avec vraie implémentation,
      B.1/B.6 restent GREEN (déterminisme toujours satisfait)

- [ ] **Étape D** : implémenter `AbstractionCartesV2` complète
      - Charger centroides, LRU cache, bucket() unifié, bucket_et_equite()
      - Test B.8/B.9 doivent passer avec vraie logique (pas rank-stub)

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
- [x] Zéro bypass TDD Guard cette session
