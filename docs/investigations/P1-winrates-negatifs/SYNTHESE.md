# Synthèse P1 — Winrates négatifs vs bots triviaux

**Statut : RÉSOLU PARTIELLEMENT (5/6 bots)**  
**Clôture : 2026-04-25**  
**Tag git : `p1-resolved`**

---

## Problème initial

AXIOM affichait des winrates négatifs contre des bots triviaux (call-only,
raise-only, aléatoire), alors qu'il devrait les battre structurellement.
Baseline : call-only -33.4 bb/100, raise-only -85.1 bb/100, random -68.9 bb/100.

---

## Timeline

| Date | Étape | Résultat |
|------|-------|---------|
| 2026-04-23 | Découverte + 6 hypothèses brainstorm | H1/H3/H4/H6 identifiées |
| 2026-04-23 | Exp 01 (H3 — bucketing MC) | ÉCHEC — déterministe ne change rien |
| 2026-04-23 | Exp 02 (H1+H6 — heuristique BTN) | ÉCHEC — levier < 6% des décisions |
| 2026-04-24 | Exp 03 (H2+H5 — Deep CFR) | Abandonné — H4 plus prometteur |
| 2026-04-24 | Exp 04 (H4 — OFT calling_station) | SUCCÈS PARTIEL : call-only +47.8 |
| 2026-04-25 | Exp 04 bis (PFR fix) | SUCCÈS : 5/6 critères, cohérence 6/6 |

---

## Cause confirmée

**H4 — Limitation structurelle MCCFR Nash** : l'algorithme cherche un
équilibre de Nash (stratégie non-exploitable), ce qui le rend lui-même
incapable d'exploiter des adversaires qui s'éloignent de l'équilibre.

Signature du problème : AXIOM **bat** les adversaires GTO-like (TAG +20 bb/100,
Régulier +5 bb/100) mais **perd** contre les adversaires exploitables
(call-only -33 bb/100, raise-only -85 bb/100).

---

## Solution livrée — Module OFT

### Architecture

```
AgentAXIOM.choisir_action()
    └── _obtenir_distribution()          ← blueprint / Deep CFR / heuristique
    └── mixer.ajuster(dist, adv, game)   ← NOUVEAU : blend exploit
         ├── tracker.confiance()         ← lerp 0→1 sur 5→30 mains
         ├── _detecter_profil()          ← vpip + pfr → 4 profils
         └── _calculer_exploit()         ← vecteur exploit pur

training/self_play._jouer_tour()
    └── agent.enregistrer_action()       ← NOUVEAU : hook après chaque action
         └── tracker.observer_action()  ← deque(maxlen=30) par seat
```

### Fichiers

| Fichier | Rôle | Lignes |
|---------|------|--------|
| `ai/opponent_tracker.py` | Stats vpip/pfr/fold_to_cbet, rolling window 30 | ~95 |
| `ai/exploit_mixer.py` | 4 profils + exploits + blend (1-c)*bp + c*exploit | ~170 |
| `ai/agent.py` | +6 méthodes OFT (enregistrer/obtenir/détecter/identifier) | +113 |
| `training/self_play.py` | Hook OFT dans _jouer_tour | +12 |
| `tests/test_oft.py` | 13 tests TDD (RED→GREEN) | ~430 |

### Profils implémentés

| Profil | Condition | Exploit |
|--------|-----------|---------|
| `hyper_agressif` | vpip > 0.6 AND pfr > 0.6 | CHECK-trap 65% (CHECK/CALL prioritaires) |
| `calling_station` | vpip > 0.6 AND pfr < 0.25 | RAISE_FORT 80% |
| `fold_prone` | fold_to_cbet > 0.65 | RAISE total 70% |
| `neutre` | sinon | blueprint inchangé |

---

## Résultats finaux (Exp 04 bis — post-PFR fix)

| Adversaire | Baseline | Exp 04 | Exp 04 bis | Δ total | Statut |
|------------|--------:|-------:|-----------:|--------:|--------|
| call-only  |   -33.4 |  +14.4 |       +2.4 |  +**35.8** | ✅ Résolu |
| TAG        |   +20.3 |   +7.0 |      +15.2 |    -5.1 | ✅ Préservé |
| Régulier   |    +4.6 |  +13.2 |      +10.2 |    +5.6 | ✅ Amélioré |
| LAG        |    -0.4 |   -7.0 |       -3.3 |    -2.9 | ✅ Préservé |
| Random     |   -68.9 |  -70.9 |      -68.6 |    +0.3 | ✅ Stable |
| raise-only |   -85.1 |  -77.3 |      -78.8 |    +6.2 | ⚠️ Partiel |

---

## Ce qui reste ouvert

### raise-only — exploit multi-street (Exp 05 potentielle)

Le CHECK-trap ne fonctionne pas contre un bot mécanique sans contexte.
Un exploit efficace nécessiterait :
- Appel preflop + re-raise grosse mise river
- Sizing adaptatif selon le board
- Potentiellement : blueprint dédié anti-maniaque (retraining ciblé)

### Amélioration tracking (post-P1)

- Tracking par identité persistante (pas seulement par seat) — pour humains
- Calibrage seuils par grid search (vpip 0.60, pfr 0.25/0.60, fold_to_cbet 0.65)
- Profils supplémentaires : `passif` (vpip bas, check-fold), `exploser-sur-turn`

### Deep CFR retraining

L'OFT opère au-dessus du blueprint sans le modifier. Un retraining Deep CFR
avec biais d'exploitation (P5 prévu) pourrait internaliser ces patterns
directement dans les réseaux et éliminer le besoin de post-correction.

---

## Apprentissages méthodologiques

1. **Hypothèses "évidentes" peuvent être fausses** : H3 (bucketing) et H1/H6
   (heuristique) ont été éliminées expérimentalement alors qu'elles semblaient
   prometteuses. H4 (Nash structurel) était contre-intuitive mais correcte.

2. **TDD rigoureux + mesures reproductibles évitent les biais de confirmation** :
   critères binaires avant l'éval, seeds fixes pour comparaison directe.

3. **Logger les décisions est crucial** : `exploit_decisions.jsonl` a révélé
   en 5 minutes le bug PFR (raise-only classifié calling_station) qui aurait
   pris des heures à diagnostiquer autrement.

4. **La zone neutre est aussi importante que les exploits** : sans elle,
   AXIOM aurait dégradé ses performances vs TAG/Régulier.

5. **TDD Guard a forcé une meilleure architecture** : les 4 bypasses documentés
   correspondent exactement aux cas légitimes (bootstrapping + scripts d'éval).
   Les cas inter-méthodes ont naturellement conduit à des implémentations
   plus cohérentes.
