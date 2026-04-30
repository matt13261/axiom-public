# P7 — Rapport final revert

**Date** : 2026-04-30
**Décision** : Option A — revert vers `pre-p7` avec préservation TA1
**Statut** : ✅ exécuté avec succès

---

## Étapes effectuées

### Phase 1 — Archive

| Action | Statut | Détail |
|---|---|---|
| Tag `p7-complete-archived` | ✅ | Posé sur `c371e21` (HEAD avant revert) + push origin |
| Backup blueprints `/tmp/axiom_blueprints_safe/` | ✅ | md5 vérifiés post-revert |
| Préservation `docs/investigations/P7-*` | ✅ | 10 rapports + spec présents post-revert |
| Préservation `data/strategy/blueprint_p7_*.pkl` | ✅ | 3 fichiers (10K + 500K + 4M) |

### Phase 2 — Branche revert

| Action | Statut |
|---|---|
| Création branche `revert-to-pre-p7` depuis main | ✅ |
| `git checkout pre-p7 -- <code applicatif>` | ✅ |
| Cherry-pick `de0c573` (TA1) sur `ai/agent.py` + `tests/test_agent_pluribus_complet.py` | ✅ |
| Suppression `tests/test_p7_abstraction.py` (testait impl P7 révoquée) | ✅ |
| Suppression `config/blinde_structure_spin_rush.json` (artefact P7) | ✅ |
| Préservation `docs/`, `data/`, `screen/`, `scripts/sync_public.sh` | ✅ |
| Commit `56e218d` `revert: P7 abstraction reverted to pre-p7. TA1 fix preserved.` | ✅ |

### Phase 3 — Merge dans main

| Action | Statut |
|---|---|
| `git checkout main && git merge --ff-only revert-to-pre-p7` | ✅ ff-only OK |
| Suppression branche temporaire `revert-to-pre-p7` | ✅ |
| `git push origin main` | ✅ `aca1118..56e218d` |
| Sync axiom-public auto via hook | ✅ |

### Phase 4 — Vérifications

| Vérification | Résultat |
|---|---|
| Tests `pytest tests/ --tb=no -q` | **193 passed, 1 warning** (192 pré-P7 + 1 TA1) |
| Sanity 1K mains × 6 baselines (V1 + Pluribus complet) | OK |
| Continuations chargées par `creer_agent` | `['baseline', 'call', 'fold', 'raise']` ✓ |
| Solveurs FLOP + SG actifs | ✓ |
| Blueprints intacts (md5 V1 + P7 4M) | ✓ identiques au backup |
| Branche actuelle | `main` |
| HEAD | `56e218d` revert commit |

### Phase 5 — Documentation

| Document | État |
|---|---|
| `docs/investigations/P7-hist-stacks-abstraction/CONCLUSION.md` | ✅ créé |
| `docs/investigations/P7-hist-stacks-abstraction/final_revert.md` | ✅ ce rapport |
| `docs/journal/SPRINT.md` | (à mettre à jour) |
| `docs/journal/ROADMAP.md` | (à mettre à jour) |

### Phase 6 — Cloud

| Action | Statut |
|---|---|
| VM `axiom-training-24` | STOPPED (déjà arrêtée post-P7.7) |
| Disque persistant | conservé (~0.05€/jour) |
| Coût total cumul P5+P6+P7 | ~30€ |

---

## Résultat sanity 1K mains post-revert

```
Continuations: ['baseline', 'call', 'fold', 'raise']
Solveur FLOP: True
Solveur SG  : True
Sanity 1K mains x 6 baselines:
  WR aleat=-30.26 call=+117.58 raise=-75.17
     TAG=+12.14 LAG=-5.74 reg=+13.78
Durée: 972s
```

Cohérent avec V1 Pluribus complet de P9 (Call +89, TAG +14, Régulier +8.5
sur 5K mains × seed 42). Les WR positifs sur Call-Only / TAG / Régulier
confirment que le revert a réussi sans régression.

---

## Tags git de référence

```
pre-p7                  edefa7f  Tag avant migration P7
pre-p7-cloud-5m         a4dd37c  Tag avant pilot 5M cloud
p7-complete-archived    c371e21  Archive HEAD complet P7 avant revert
```

Pour reprendre P7 :
```bash
git checkout p7-complete-archived
```

Pour vérifier état pre-P7 pur (sans TA1) :
```bash
git checkout pre-p7
```

---

## URLs axiom-public

- CONCLUSION P7 : https://github.com/matt13261/axiom-public/blob/main/docs/investigations/P7-hist-stacks-abstraction/CONCLUSION.md
- Revert final : https://github.com/matt13261/axiom-public/blob/main/docs/investigations/P7-hist-stacks-abstraction/final_revert.md
- Architecture audit : https://github.com/matt13261/axiom-public/blob/main/docs/investigations/P7-hist-stacks-abstraction/architecture_audit.md

---

## Suite proposée (P10)

Voir mise à jour `SPRINT.md` :
- **P10** : Validation V1 complet sur 3 seeds × 10K mains (chiffres
  définitifs), focus sur amélioration baselines difficiles (LAG, Aléatoire,
  Raise-Only)
- **P11 (parking)** : Continuations P7 si on reprend l'abstraction P7

---

## État final post-revert

✅ Branche main propre, 193 tests verts, blueprints intacts, archives P7
préservées, axiom-public synchronisé. **Revert complété avec succès.**
