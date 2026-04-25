# Exp 01-H3-final — Post-revert baseline (seed=42, 100 sims)

**Seeds :** [42, 123, 2026]  
**Seuil degradation acceptable :** +/-3.0 bb/100  

| Adversaire | Run 1 | Run 2 | Run 3 | Moyenne | Baseline | Delta | Statut |
|------------|------:|------:|------:|--------:|---------:|------:|:-------|
| aleatoire  |  -77.6 |  -59.2 |  -69.9 |   -68.9 |    -68.4 |  -0.5 | OK |
| call       |  -71.5 |  -10.4 |  -18.4 |   -33.4 |     +1.6 | -35.0 | ALERTE (|-35.0|>3.0) |
| raise      |  -86.7 |  -73.4 |  -95.1 |   -85.0 |    -78.4 |  -6.6 | ALERTE (|-6.6|>3.0) |
| tag        |  +21.1 |  +25.6 |  +14.3 |   +20.3 |    +13.5 |  +6.8 | ALERTE (|+6.8|>3.0) |
| lag        |   -3.6 |   -1.7 |   +4.2 |    -0.4 |     -5.8 |  +5.4 | ALERTE (|+5.4|>3.0) |
| regulier   |   +6.6 |   +8.1 |   -1.0 |    +4.6 |    +17.4 | -12.8 | ALERTE (|-12.8|>3.0) |

**Verdict : ALERTE — reverter aussi la seed**
