# Sprint actuel — Setup architecture (Semaine 2026-04-25)

## Objectif sprint
Boucler setup complet et démarrer Phase 2.

## Tâches critiques
- [ ] Système journal opérationnel + hook git
- [ ] Architecture GitHub hybride (privé + public)
- [ ] Compte Google Cloud créé
- [ ] Décision blueprint HU
- [ ] Spec P6 démarrée

## Important
- [ ] Étude littérature : E[HS²], OCHS, k-means
- [ ] Mini-prototype k-means 1000 mains
- [ ] Profiling CPU bottlenecks

## Définition done
- [ ] Spec P6 reviewée
- [ ] 1 expérience prototype lancée
- [ ] Cloud opérationnel
- [ ] 2 repos GitHub synchronisés

## Journal décisions

### 2026-04-25 — Blueprint HU
**Décision : HU hors scope — 3-max strict.**

Diagnostic :
- Aucun `blueprint_hu.pkl` dans `data/strategy/` → aucun entraînement HU MCCFR jamais lancé.
- `data/checkpoints/` (iter 001-109) = Deep CFR 3-max (regret/strategy/valeur nets × 3 joueurs), pas du HU.
- Aucun processus Python actif.

Conséquence : rien à arrêter, rien à archiver.
Focus exclusif 3-max Spin 15€. Toute ref au "blueprint_hu" dans le code reste en place
comme fallback architectural, mais aucun entraînement HU ne sera lancé en Phase 2.

### 2026-04-25 — Architecture GitHub
_(à compléter lors de la Phase C)_
