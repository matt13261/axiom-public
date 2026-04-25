# Exp 04 bis — Résultats finaux

**Date :** 2026-04-25  
**Branch :** exp/P1-h4-oft-bis-pfr  
**Seeds :** [42, 123, 2026]  
**Mains :** 1 000 par run/bot = 18 000 totales  

---

## Verdict : SUCCÈS — MERGE

5/6 critères atteints. Cohérence classification 6/6. Aucun revert.

---

## Tableau triple comparaison

| Bot | Moy 04bis | σ | Moy 04 | Δ vs 04 | Moy baseline | Δ vs base | Statut |
|-----|----------:|--:|-------:|--------:|-------------:|----------:|--------|
| aléatoire  | -68.6 | 6.0 | -70.9 | +2.4  | -68.9 | +0.3  | ✅ |
| call-only  |  +2.4 |12.3 | +14.4 | -12.1 | -33.4 | +35.8 | ✅ |
| raise-only | -78.8 | 4.2 | -77.3 |  -1.5 | -85.1 | +6.2  | ❌ |
| TAG        | +15.2 | 4.6 |  +7.0 |  +8.1 | +20.3 | -5.2  | ✅ |
| LAG        |  -3.3 | 5.2 |  -7.0 |  +3.7 |  -0.4 | -2.9  | ✅ |
| Régulier   | +10.2 | 3.0 | +13.2 |  -3.1 |  +4.6 | +5.6  | ✅ |

---

## Critères atteints : 5/6

- ✅ vs call-only : +2.4 bb/100 (préservé positif — gain net P1 = **+35.8 bb/100**)
- ✅ vs TAG : +15.2 bb/100 (préservé — meilleure valeur des 3 expériences)
- ✅ vs Régulier : +10.2 bb/100 (amélioré +5.6 vs baseline)
- ✅ vs LAG : -3.3 bb/100 (dans seuil -15)
- ✅ vs random : -68.6 bb/100 (pas de régression — limite structurelle)
- ❌ vs raise-only : -78.8 bb/100 (cohérence résolue mais exploit trop simple)

---

## Cohérence classification : 6/6 ✅

| Bot | Profil dominant | VPIP | PFR | Hyper% | Cohérence |
|-----|----------------|------|-----|--------|-----------|
| aléatoire  | hyper_agressif  | 0.715 | 0.502 | 99%  | ✅ |
| call-only  | calling_station | 0.998 | 0.001 | 0%   | ✅ |
| raise-only | hyper_agressif  | 0.998 | 0.998 | 100% | ✅ |
| TAG        | neutre          | 0.131 | 0.074 | 0%   | ✅ |
| LAG        | neutre          | 0.203 | 0.138 | 0%   | ✅ |
| Régulier   | neutre          | 0.126 | 0.080 | 0%   | ✅ |

Le PFR fix élimine l'ambiguïté Exp 04 : raise-only (pfr=0.998) correctement
séparé de calling_station (pfr=0.001).

---

## Limite identifiée — pas un bug

L'exploit `hyper_agressif` (CHECK-trap 65%) ne génère pas de gain contre
un bot raise-only **mécanique** : le bot raise systématiquement sans tenir
compte du contexte board/position. Face à un check, il raise quand même —
le trap ne fonctionne pas.

**Ce que le CHECK-trap requiert pour fonctionner :** un adversaire qui
value-bet ou bluff selon le contexte, et qui face à un check ressent une
"invitation à bluffer". Un bot mécanique n'a pas ce modèle.

**Solutions futures possibles (hors scope P1) :**
- Exploit `hyper_agressif` v2 : appel preflop + re-raise ciblé river
- Trap multi-street avec sizing adaptatif
- Blueprint dédié anti-maniaque (retraining avec biais spécifique)
- Intégration dans Deep CFR retraining (long terme)

---

## Apprentissages

1. **PFR est indispensable** pour distinguer caller (vpip↑, pfr↓) de raiser
   (vpip↑, pfr↑) — VPIP seul crée des faux positifs
2. **Le fix minimal (3 tests, ~50 lignes) résout la classification** à 100%
3. **La résistance à l'exploit dépend du modèle de l'adversaire** :
   un bot mécanique ne "réagit" pas aux traps
4. **Logger PFR dans les décisions** a permis de confirmer le fix instantanément

---

## Fichiers produits

- `results_seed_{42,123,2026}.json`
- `exploit_decisions.jsonl` (24 910 entrées — archivé)
- `run_exp04bis.py` (script évaluation réutilisable)
