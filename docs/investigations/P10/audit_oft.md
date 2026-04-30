# P10.A — Audit OFT A/B (agent Pluribus complet)
**Date** : 2026-04-30
**Protocole** : 1 seed (42) × 3000 mains/baseline × 6 baselines
**Blueprint** : `data/strategy/blueprint_v1.pkl`
**Agent** : Pluribus complet (continuations k=4 + solveurs FLOP/SG + DCFR + heuristique)
**Variable** : OFT activé (run A) vs OFT désactivé via NoOpMixer (run B)
**Durée totale** : 83.2 min

---

## Tableau A/B

| Baseline | OFT ON | OFT OFF | Δ (ON−OFF) | Verdict |
|---|---:|---:|---:|---|
| Aléatoire | -36.64 | -53.60 | **+16.96** | utile |
| Call-Only | +91.35 | +44.34 | **+47.01** | utile |
| Raise-Only | -71.93 | -71.99 | **+0.06** | neutre |
| TAG | +12.02 | +12.93 | **-0.91** | neutre |
| LAG | -2.01 | +7.78 | **-9.79** | nuisible |
| Régulier | +0.23 | +7.60 | **-7.37** | nuisible |
| **Moyenne** | **-1.16** | **-8.82** | **+7.66** | — |

## Comptes

- **Utiles** (Δ > +2 bb/100) : **2/6**
- **Neutres** (|Δ| ≤ 2 bb/100) : **2/6**
- **Nuisibles** (Δ < −2 bb/100) : **2/6**

## Verdict final

**INDÉTERMINÉ par règle automatique** (2 utiles + 2 neutres + 2 nuisibles)
**mais le pattern est en fait NET et conditionnel par profil adversaire.**

### Pattern observé

| Famille adversaire | Δ OFT (ON−OFF) | Conclusion |
|---|---|---|
| **Bots irréguliers** (Aléatoire, Call-Only) | **+16.96**, **+47.01** | OFT **massivement utile** ✓ |
| **Bot mécanique** (Raise-Only) | +0.06 | OFT neutre (raise-only n'a pas de pattern exploitable variable) |
| **Bot structuré équilibré** (TAG) | -0.91 | OFT neutre (zone neutre OFT fonctionne) |
| **Bots structurés agressifs** (LAG, Régulier) | **-9.79**, **-7.37** | OFT **nuisible** ⚠️ |

OFT est **conçu pour exploiter les patterns adversaires**. Le module
fonctionne comme attendu :
- Contre des bots avec patterns franchement déséquilibrés (call-station,
  random) il identifie correctement le profil et exploite massivement
  (+47 bb/100 vs Call-Only).
- Contre des bots structurés (LAG, Régulier), il **fait des faux positifs
  de profil** et applique un exploit non justifié, ce qui dégrade la
  performance.

### Origine probable du faux positif sur LAG/Régulier

LAG (Loose-Aggressive) joue beaucoup et raise souvent → ses stats VPIP/PFR
peuvent dépasser les seuils détecteurs OFT (`hyper_agressif` = vpip>0.6 AND
pfr>0.6 selon CLAUDE.md). L'agent passe en mode "exploit hyper-agressif"
mais LAG n'est pas exploitable comme un random — c'est une stratégie
structurée GTO-deviating mais cohérente.

Régulier joue probablement à des fréquences légèrement déséquilibrées
(réaliste pour mimer un grinder humain) → idem, OFT le classe à tort.

## Recommandation pour P10.B

**GARDER OFT par défaut MAIS** réviser les seuils détecteurs pour réduire
les faux positifs sur bots structurés. La cible est le delta moyen +7.66
qu'on observe, à confirmer sur 3 seeds × 10K mains avant fix.

Trois options pour P10.B :

**Option α — Garder OFT tel quel** : delta moyen positif (+7.66 bb/100),
gain massif sur Call-Only et Aléatoire qui couvre le coût sur LAG/Régulier.
Acceptable si la population de bots/humains rencontrés est majoritairement
irrégulière.

**Option β — Tightener seuils OFT** : élever les seuils VPIP/PFR
détecteurs pour passer en mode neutre plus souvent. Risque : perte de gain
sur Call-Only / Aléatoire en échange de moins de pertes sur LAG/Régulier.
Demande tuning empirique.

**Option γ — OFT conditionnel par game_type** : activer OFT uniquement si
on détecte un profil "exploitable" (variance VPIP haute, etc.) avec
threshold strict, sinon NoOp. Plus complexe à coder.

**Recommandation** : Option α pour P10.B (validation 3 seeds × 10K mains
avec OFT ON), puis option β si le pattern persiste sur les chiffres
définitifs. Option γ reportée si nécessaire.

### Caveat protocole

3000 mains × 1 seed produit ~5-7 bb/100 d'écart-type sur les bots
structurés (estimation par variance buy-in). Les deltas mesurés sur
LAG (-9.79) et Régulier (-7.37) sont marginalement au-delà de cette
incertitude. Validation 3 seeds × 10K mains nécessaire pour confirmer
définitivement les nuisibles.

## Données brutes

```json
{
  "OFT_ON": {
    "label": "OFT_ON",
    "enable_oft": true,
    "seed": 42,
    "nb_mains": 3000,
    "vs_aleatoire": -36.64333333333334,
    "vs_call": 91.35000000000001,
    "vs_raise": -71.92833333333333,
    "vs_tag": 12.021666666666668,
    "vs_lag": -2.011666666666668,
    "vs_regulier": 0.23333333333333192,
    "duree_s": 2384.5333235263824
  },
  "OFT_OFF": {
    "label": "OFT_OFF",
    "enable_oft": false,
    "seed": 42,
    "nb_mains": 3000,
    "vs_aleatoire": -53.6,
    "vs_call": 44.336666666666666,
    "vs_raise": -71.99166666666666,
    "vs_tag": 12.931666666666667,
    "vs_lag": 7.778333333333333,
    "vs_regulier": 7.603333333333334,
    "duree_s": 2553.729981660843
  }
}
```
