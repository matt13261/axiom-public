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

### Phase 2 — Refonte abstraction P6 (semaines 2-5) ✅
Passage de 8 à 50 buckets postflop, métrique E[HS²] + clustering.
- [x] Spec P6 complète
- [x] Implémentation TDD
- [x] Recalibration cartes 3-max
- [x] Ré-entraînement blueprint cloud (~30€ cumul P6+P7)
- [x] Validation préliminaire (V2 active)

### Phase P7 — Refonte hist + stacks Spin & Rush — **ARCHIVED 2026-04-30**
Tentative de réduire cardinalité hist/stacks (cap=4 + Variante B + paliers
Spin & Rush 7 niveaux). **Reverté** car écart V1 vs P7 sur bots structurés
> 15 bb/100 (TAG +25.75, LAG +19.77, Régulier +20.27).

**Le vrai gain** de P7 a été **TA1** : audit architectural révélant que
`creer_agent` n'activait pas continuations + solveurs. Fix préservé
post-revert, +21.54 bb/100 moyen sur V1.

Détails : `docs/investigations/P7-hist-stacks-abstraction/CONCLUSION.md`
Tag archive : `p7-complete-archived` pour reprendre P7 plus tard si besoin.

### Phase P10 — Validation V1 + Pluribus complet (NEXT)
- [ ] Eval V1 Pluribus complet 3 seeds × 10K mains
- [ ] Diag baselines difficiles (Aléatoire, Raise-Only, LAG)
- [ ] Décision suite : continuer optimisations sur V1 ou redémarrer P7

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
