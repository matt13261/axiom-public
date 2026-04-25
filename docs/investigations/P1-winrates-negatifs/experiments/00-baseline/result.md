# Baseline P1 — Résultats

**Seeds :** [42, 123, 2026]  
**Mains par run :** 500  

| Adversaire | Run 1 | Run 2 | Run 3 | Moyenne | Écart-type |
|------------|------:|------:|------:|--------:|-----------:|
| aleatoire  |  -68.2 |  -66.6 |  -70.5 |   -68.4 |       2.0 |
| call       |  +21.5 |  -45.9 |  +29.4 |    +1.6 |      41.4 |
| raise      |  -86.7 |  -66.0 |  -82.7 |   -78.4 |      11.0 |
| tag        |  +17.7 |  +19.3 |   +3.5 |   +13.5 |       8.7 |
| lag        |  -15.0 |   +4.7 |   -7.2 |    -5.8 |       9.9 |
| regulier   |  +10.7 |  +30.2 |  +11.3 |   +17.4 |      11.1 |

## Critère kill-switch

Après Exp 01 + Exp 02, si winrate vs random < (baseline_mean + 5 bb/100) → passer directement à Exp 04.

**Baseline vs random (moyenne) : -68.4 bb/100**
**Seuil kill-switch : -63.4 bb/100**
