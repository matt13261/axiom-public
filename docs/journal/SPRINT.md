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

## Phase P7 — Refonte abstraction hist + stacks (Spin & Rush)
- [x] P7.1 : analyse code existant ✅
- [x] P7.2 : spec écrite + amendements (Variante B mapping) ✅
- [x] P7.3 : 15 tests RED (12 RED + 3 GUARDRAIL) ✅
- [x] P7.3.bis : tag `pre-p7` posé sur `edefa7f` ✅
- [x] P7.4 : implémentation 6 fichiers, cap=4, 207/207 GREEN ✅
- [x] **P7.5 : saturation curve validée (α=0.667, ratio<60 dès 50K)** ✅ 2026-04-29
- [x] **P7.6 : pilot cloud 500K — ratio 4.53 (mieux que prédit 6.0), 1.80€** ✅ 2026-04-29
- [x] **P7.7 : pilot cloud 4M (interrompu réseau VM à batch 9/10) — 2.99M infosets, ~24€** ✅ 2026-04-30
- [ ] **P8 : évaluation blueprint P7 vs baselines + comparaison pré-P7** ← NEXT
- [ ] P7.7 : pilot 5M cloud (~10€)
- [ ] P7.8 : validation winrates + merge

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
