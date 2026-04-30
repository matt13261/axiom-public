# Sprint actuel — Phase 2 : Abstraction V2 (Semaine 2026-04-26)

## Objectif sprint
Implémenter et valider l'abstraction cartes V2 (E[HS²] + Potentiel + K-means 3D).

## Tâches critiques
- [x] Système journal opérationnel + hook git ✅ 2026-04-25
- [x] Architecture GitHub hybride (privé + public) ✅ 2026-04-25
- [ ] Compte Google Cloud créé (guide prêt, action manuelle)
- [x] Décision blueprint HU → HU hors scope ✅ 2026-04-25
- [x] Spec P6 finalisée ✅ 2026-04-26

## Phase 2 — Abstraction V2 (14 jours)
- [x] Étape A : stubs AbstractionCartesV2 + card_clustering ✅
- [x] Étape B : 14 tests RED ✅
- [x] Étape C : implémentation card_clustering.py ✅
- [x] Étape D : implémentation AbstractionCartesV2 ✅
- [x] Étape E : calibration cloud → centroides_v2.npz ✅
- [x] Étape F : bascule modules vers V2 ✅
- [x] Étape G : ré-entraînement blueprint MCCFR (cloud, 37€) — partiel, voir P7
- [ ] Étape H : validation 3×6×1000 mains + merge

## Phase P7 — Refonte abstraction hist + stacks (Spin & Rush) — **ARCHIVED 2026-04-30**
- [x] P7.1-7 : implémentation + cloud pilot 4M (24€) — voir CONCLUSION.md
- [x] P8 : eval baseline (V1 +20.32 vs P7 -24.56 — agent mutilé) ✅
- [x] P8.diag : Deep CFR confound réfuté ✅
- [x] **TA1 audit majeur : creer_agent loupait continuations + solveurs** ✅ 2026-04-30
- [x] **TA1 fix : creer_agent active Pluribus complet** ✅ commit `de0c573`
- [x] P9 : eval Pluribus complet (V1 +1.22 vs P7 -19.94, écart structurés > 15) ✅ 2026-04-30
- [x] **REVERT P7 → pre-p7 (TA1 préservé)** ✅ 2026-04-30 commit `56e218d`
      Tag archive : `p7-complete-archived` (commit `c371e21`)

## Phase P10 — Validation V1 + Pluribus complet (NEXT)
Objectif : confirmer les chiffres V1 Pluribus complet sur 3 seeds × 10K mains
(P9 mesuré 1 seed × 5K), focus sur amélioration des baselines difficiles
(Aléatoire, Raise-Only, LAG).
- [ ] P10.1 : Eval V1 Pluribus complet 3 seeds × 10K mains (~10h local)
- [ ] P10.2 : Analyse focus LAG (-3.50 baseline → +7.20 Pluribus, marginal)
- [ ] P10.3 : Diag pourquoi Aléatoire/Raise-Only restent négatifs malgré
      continuations (potentiel exploit OFT mismatch ?)

## Parking lot (post-P10 si pertinent)
- [ ] **P11** : Continuations P7 cloud (~50€) si on reprend P7 abstraction
- [ ] **P12** : Re-train Deep CFR sous abstraction post-P10

## Important
- [ ] Créer compte Google Cloud (guide disponible : docs/journal/done/2026-04-25.md)
- [ ] Mini-prototype k-means 1000 mains (Étape E locale)
- [ ] Monitoring compute cloud (kill à 60€)

## Définition done (Phase 2)
- [x] Spec P6 reviewée et finalisée ✅
- [ ] 14 tests RED écrits et confirmés en FAIL
- [ ] Tests GREEN après implémentation
- [ ] Centroïdes V2 calibrés et validés
- [ ] Blueprint V2 entraîné sur cloud
- [ ] Validation : critères section 5 de la spec satisfaits
- [ ] 2 repos GitHub synchronisés ✅

## Journal décisions

### 2026-04-25 — Blueprint HU
**Décision : HU hors scope — 3-max strict.**

Diagnostic :
- Aucun `blueprint_hu.pkl` dans `data/strategy/` → aucun entraînement HU MCCFR jamais lancé.
- `data/checkpoints/` (iter 001-109) = Deep CFR 3-max (regret/strategy/valeur nets × 3 joueurs), pas du HU.
- Aucun processus Python actif.

Conséquence : rien à arrêter, rien à archiver.
Focus exclusif 3-max Spin 15€.

### 2026-04-25 — Architecture GitHub hybride
**axiom-private** (privé, code complet) + **axiom-public** (public, docs+tests).
Hook post-commit auto-sync axiom-public si modifs dans docs/ ou tests/.
Script : `scripts/sync_public.sh`.

### 2026-04-26 — Méthode abstraction P6
**E[HS²] + Potentiel + K-means 3D street-specific, 50 buckets postflop.**

- 3 features : (E[HS], E[HS²], Potentiel) calculées en 1 run MC
- 3 sets de centroïdes : flop / turn / river (50 chacun)
- Préflop : inchangé (8 buckets, lookup table V1)
- Budget cloud : 51€ (calibration 14€ + training 37€)
- Référence : docs/investigations/P6-abstraction/spec.md
