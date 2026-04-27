# TODO — Session 2026-04-26

## Immédiat (prochaine session — Étape F)
- [x] **Étape C** : implémenter `card_clustering.py` — TERMINÉ (4 commits, 161 tests GREEN)
      - C.1 : `compute_features` MC réelle (B.13)
      - C.2 : `fit_centroids` sklearn KMeans (B.14)
      - C.3 : `predict_bucket` argmin L2 (B.15)
      - C.4 : tests intégration pipeline (B.16/B.17)
      - fix : potentiel négatif (B.18)

- [x] **Étape D** : implémenter `AbstractionCartesV2` complète — TERMINÉ (5 commits, 166 tests GREEN)
      - D.1 : __init__ avec centroides= dict direct
      - D.2 : bucket_postflop réelle (compute_features + predict_bucket)
      - D.3 : bucket_preflop délègue à V1
      - D.4 : cache dict (26000x speedup sur cache hit)
      - D.5 : régression OFT + V2 réelle
      - fix : RuntimeError si centroides=None (plus de fallback silencieux)

- [x] **Étape E** : calibration centroides — TERMINÉ en LOCAL (171 tests GREEN, 0€)
      - E.1 : script recalibrer_3max_v2.py + multiprocessing (tests E.1/E.1b)
      - E.2 : validation n=200 (5.1s, discrimination OK)
      - E.3 : calibration finale n=4000, n_workers=4 (15.9s total)
      - centroides_v2.npz : 2 KB, flop/turn/river (50,3)
      - Tests E.3 + B.9v2 GREEN avec vrais centroïdes
      - Cloud NON utilisé — budget 14€ épargné

- [x] **Étape F** : bascule modules vers AbstractionCartesV2 — TERMINÉ (181 tests GREEN)
      - F.0 : bucket() + bucket_et_equite() sur V2 + auto-chargement npz
      - F.1 : solver/subgame_solver.py migré
      - F.2 : solver/depth_limited.py migré
      - F.3 : ai/deep_cfr.py migré
      - F.4 : ai/agent.py migré (2 instances 3max+hu)
      - F.5 : ai/mccfr.py migré
      - F.6 : tests mis à jour (plage bucket V2), merge main, tag p6-bascule-complete

## Suite (dans l'ordre)
- [ ] **Étape G** : training blueprint cloud (37€) — post Étape F ✅
      - G.1 : préparer script d'entraînement V2 (namespace clés infoset changé)
      - G.2 : valider run local rapide (N=100 iter) avant cloud
      - G.3 : lancer training cloud (budget 37€ max)
      - G.4 : sauvegarder blueprint_v2.pkl + valider
- [ ] Étape H : validation + merge

## Backlog
- [ ] Créer compte Google Cloud (guide dans done/2026-04-25.md)
- [ ] Continuation Strategies k=4 (infrastructure prête, voir TODO.txt)

## Terminé cette session
- [x] Section 10 spec.md : décision rang_normalise documentée (MC vs exact)
- [x] Étapes A-B : 14 tests atomiques TDD, stubs V2 (155 tests GREEN)
- [x] Étapes C-D : vraie implémentation card_clustering + AbstractionCartesV2
      - 11 nouveaux tests (B.13-B.18, D.1-D.5)
      - Corrections : potentiel négatif, fallback RuntimeError
- [x] Étape E : calibration locale n=4000, 15.9s, 0€ cloud
      - 4 nouveaux tests (E.1, E.1b, E.3, B.9v2)
- [x] 171 tests GREEN — zéro bypass TDD Guard
