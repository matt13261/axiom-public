# Roadmap AXIOM — 3-max Spin & Go 15€

## Mission
Agent autonome capable de battre régulièrement les Spin & Go 15€
sur Betclic, Winamax ou équivalent. Winrate cible : +3 à +6 bb/100.

## Stratégie technique cible
- Format : 3-max No-Limit Hold'em
- Stack depth : 25-100 BB
- Abstraction : 50-100 buckets postflop, 12-15 actions abstraites
- Algorithme : MCCFR blueprint + Deep CFR + subgame solving + OFT v2
- Compute : CPU local (dev/test) + Google Cloud (training intensif)

## Phases (6-9 mois)

### Phase 1 — Stabilisation (semaine 1) ✅
- [x] Git, tests, skills Claude Code
- [x] P1 résolu (OFT v1)
- [ ] Système journal opérationnel
- [ ] Architecture GitHub hybride
- [ ] Setup Google Cloud

### Phase 2 — Refonte abstraction P6 (semaines 2-5)
Passage de 8 à 50 buckets postflop, métrique E[HS²] + clustering.
- [ ] Spec P6 complète
- [ ] Implémentation TDD
- [ ] Recalibration cartes 3-max
- [ ] Ré-entraînement blueprint cloud (50€ budget)
- [ ] Validation : winrate vs random > -30, vs reg > +20

### Phase 3 — Action abstraction (semaines 5-7)
Passage de 7 à 15 actions abstraites.
- [ ] Spec sizings (0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0)
- [ ] Re-entraînement (50€ cloud)
- [ ] Validation contre PioSolver 3-max

### Phase 4 — Deep CFR sérieux (semaines 7-10)
- [ ] Spec architecture réseaux
- [ ] Training intensif cloud (200€)
- [ ] Exploitabilité < 10 mbb/g vs solveur référence

### Phase 5 — Subgame solving avancé (semaines 10-13)
- [ ] Refactor depth_limited.py performance
- [ ] Cache sous-jeux fréquents
- [ ] Latence décision < 3s

### Phase 6 — OFT v2 et intégration (semaines 13-16)
- [ ] OFT v2 multi-street
- [ ] Identité persistante par pseudo
- [ ] Hooks HUD personnalisé
- [ ] Screen reader live

### Phase 7 — Validation et déploiement (semaines 16-24)
- [ ] Test dataset historique (10000+ mains)
- [ ] Benchmark vs PioSolver, GTOWizard
- [ ] Mode play money 10000 mains
- [ ] Validation Spin 15€ play money
- [ ] Décision GO/NO-GO réel

## Compute budget Google Cloud (520€)
- Phase 2 : 50€
- Phase 3 : 50€
- Phase 4 : 200€
- Phase 7 : 100€
- Réserve : 120€

## Métriques succès
| Phase | Winrate cible | Technique |
|-------|---------------|-----------|
| 2 | vs random > -30 | Exploitabilité Kuhn < 0.005 |
| 3 | vs reg > +25 | Distribution sizings cohérente |
| 4 | vs reg > +30 | Loss < 0.01 |
| 5 | Tous | Latence < 3s |
| 6 | Tous | OFT calibré < 30 mains |
| 7 | Spin 15€ +3 bb/100 | GO/NO-GO |

## Risques
1. Compute insuffisant → mitigation : réduire abstraction
2. Plafond technique avant Phase 7 → fallback Spin 7€
3. Burn-out → milestones hebdomadaires
4. Évolution metagame → focus GTO + OFT adaptatif

## Limites assumées
- 3-max strict (pas HU, pas 6-max)
- Spin 15€ cible primaire
- Solo dev sans équipe
