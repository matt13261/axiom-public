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

---

## P10.A.bis cloud - variance OFT (tracker frais)

**Date** : 2026-05-01

**Protocole corrige** vs P10.A : agent FRAIS recree pour chaque (oft, seed, baseline) -> tracker OFT vide en debut de chaque baseline. L'audit cross-baseline a confirme que P10.A polluait le tracker.

**Parametres** : 1500 mains x 3 seeds x 6 baselines x 2 conditions = 36 runs

**Cloud** : VM axiom-training-24 (n2-standard-32), 4 workers paralleles

**Duree** : 63.0 min

**Pic RSS observe** : 12051 MB / worker (cap 13000 MB)

### Tableau synthese moyenne, sigma, IC 95%

| Baseline | mu OFT_ON | sigma ON | mu OFT_OFF | sigma OFF | Delta (ON-OFF) | IC 95% | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Aleatoire | -28.28 | 0.75 | -32.13 | 9.62 | **+3.85** | +-10.92 | BRUIT |
| Call-Only | +70.84 | 38.37 | +82.17 | 32.02 | **-11.34** | +-56.55 | BRUIT |
| Raise-Only | -68.87 | 6.26 | -74.23 | 10.28 | **+5.36** | +-13.61 | BRUIT |
| TAG | +12.56 | 1.23 | +9.60 | 1.30 | **+2.96** | +-2.02 | UTILE |
| LAG | -7.31 | 13.65 | -6.00 | 5.42 | **-1.31** | +-16.63 | BRUIT |
| Regulier | +3.83 | 5.66 | +11.84 | 7.45 | **-8.00** | +-10.59 | BRUIT |

### Donnees brutes (36 runs)

```jsonl
{"oft": false, "seed": 42, "baseline": "aleatoire", "wr_bb100": -25.89, "duree_s": 268.0756607055664, "rss_mb": 12005.10546875, "pid": 3622}
{"oft": false, "seed": 42, "baseline": "call", "wr_bb100": 109.66666666666669, "duree_s": 979.225307226181, "rss_mb": 11996.8984375, "pid": 3623}
{"oft": false, "seed": 42, "baseline": "lag", "wr_bb100": -5.553333333333335, "duree_s": 302.1658225059509, "rss_mb": 11996.75390625, "pid": 3624}
{"oft": false, "seed": 42, "baseline": "raise", "wr_bb100": -71.24333333333334, "duree_s": 171.29240918159485, "rss_mb": 11993.52734375, "pid": 3624}
{"oft": false, "seed": 42, "baseline": "regulier", "wr_bb100": 13.860000000000001, "duree_s": 328.4943051338196, "rss_mb": 11994.98828125, "pid": 3621}
{"oft": false, "seed": 42, "baseline": "tag", "wr_bb100": 11.036666666666667, "duree_s": 303.78054332733154, "rss_mb": 12001.95703125, "pid": 3622}
{"oft": false, "seed": 1337, "baseline": "aleatoire", "wr_bb100": -27.283333333333335, "duree_s": 249.29339170455933, "rss_mb": 12001.9375, "pid": 3622}
{"oft": false, "seed": 1337, "baseline": "call", "wr_bb100": 89.83, "duree_s": 984.3452303409576, "rss_mb": 12011.2109375, "pid": 3624}
{"oft": false, "seed": 1337, "baseline": "lag", "wr_bb100": -11.639999999999999, "duree_s": 343.2861876487732, "rss_mb": 11993.453125, "pid": 3621}
{"oft": false, "seed": 1337, "baseline": "raise", "wr_bb100": -65.77333333333333, "duree_s": 150.5227017402649, "rss_mb": 11993.203125, "pid": 3621}
{"oft": false, "seed": 1337, "baseline": "regulier", "wr_bb100": 3.583333333333332, "duree_s": 313.7145674228668, "rss_mb": 11995.08203125, "pid": 3623}
{"oft": false, "seed": 1337, "baseline": "tag", "wr_bb100": 8.506666666666666, "duree_s": 352.9992325305939, "rss_mb": 12050.66015625, "pid": 3622}
{"oft": false, "seed": 2024, "baseline": "aleatoire", "wr_bb100": -43.20666666666667, "duree_s": 242.53538513183594, "rss_mb": 12002.15625, "pid": 3622}
{"oft": false, "seed": 2024, "baseline": "call", "wr_bb100": 47.01999999999999, "duree_s": 951.0177667140961, "rss_mb": 11996.609375, "pid": 3621}
{"oft": false, "seed": 2024, "baseline": "lag", "wr_bb100": -0.8200000000000012, "duree_s": 291.8027546405792, "rss_mb": 12050.6171875, "pid": 3622}
{"oft": false, "seed": 2024, "baseline": "raise", "wr_bb100": -85.66333333333334, "duree_s": 141.41366243362427, "rss_mb": 12050.53125, "pid": 3622}
{"oft": false, "seed": 2024, "baseline": "regulier", "wr_bb100": 18.073333333333338, "duree_s": 335.51189160346985, "rss_mb": 12004.328125, "pid": 3624}
{"oft": false, "seed": 2024, "baseline": "tag", "wr_bb100": 9.26333333333333, "duree_s": 330.3093750476837, "rss_mb": 11993.09765625, "pid": 3623}
{"oft": true, "seed": 42, "baseline": "aleatoire", "wr_bb100": -28.909999999999997, "duree_s": 204.86695289611816, "rss_mb": 12035.6953125, "pid": 3621}
{"oft": true, "seed": 42, "baseline": "call", "wr_bb100": 99.58999999999999, "duree_s": 973.4194982051849, "rss_mb": 12039.9453125, "pid": 3622}
{"oft": true, "seed": 42, "baseline": "lag", "wr_bb100": 0.4833333333333331, "duree_s": 294.37117075920105, "rss_mb": 12040.13671875, "pid": 3624}
{"oft": true, "seed": 42, "baseline": "raise", "wr_bb100": -64.5, "duree_s": 115.36082530021667, "rss_mb": 12032.09375, "pid": 3624}
{"oft": true, "seed": 42, "baseline": "regulier", "wr_bb100": -2.2600000000000016, "duree_s": 301.78554463386536, "rss_mb": 12000.171875, "pid": 3621}
{"oft": true, "seed": 42, "baseline": "tag", "wr_bb100": 12.633333333333333, "duree_s": 280.25406765937805, "rss_mb": 11989.71875, "pid": 3623}
{"oft": true, "seed": 1337, "baseline": "aleatoire", "wr_bb100": -28.476666666666663, "duree_s": 243.1550908088684, "rss_mb": 11993.37890625, "pid": 3623}
{"oft": true, "seed": 1337, "baseline": "call", "wr_bb100": 85.64999999999999, "duree_s": 1012.8868598937988, "rss_mb": 11997.6328125, "pid": 3624}
{"oft": true, "seed": 1337, "baseline": "lag", "wr_bb100": -23.080000000000002, "duree_s": 349.2874925136566, "rss_mb": 11993.72265625, "pid": 3621}
{"oft": true, "seed": 1337, "baseline": "raise", "wr_bb100": -66.06666666666666, "duree_s": 133.49384689331055, "rss_mb": 11992.98046875, "pid": 3621}
{"oft": true, "seed": 1337, "baseline": "regulier", "wr_bb100": 4.846666666666668, "duree_s": 346.2464635372162, "rss_mb": 11992.65234375, "pid": 3623}
{"oft": true, "seed": 1337, "baseline": "tag", "wr_bb100": 11.296666666666667, "duree_s": 314.72762656211853, "rss_mb": 11992.88671875, "pid": 3623}
{"oft": true, "seed": 2024, "baseline": "aleatoire", "wr_bb100": -27.44333333333334, "duree_s": 245.97694849967957, "rss_mb": 12001.328125, "pid": 3622}
{"oft": true, "seed": 2024, "baseline": "call", "wr_bb100": 27.26666666666667, "duree_s": 954.3501105308533, "rss_mb": 11996.40234375, "pid": 3621}
{"oft": true, "seed": 2024, "baseline": "lag", "wr_bb100": 0.6566666666666663, "duree_s": 345.1096475124359, "rss_mb": 11993.4140625, "pid": 3623}
{"oft": true, "seed": 2024, "baseline": "raise", "wr_bb100": -76.03333333333332, "duree_s": 139.45536589622498, "rss_mb": 11993.4375, "pid": 3623}
{"oft": true, "seed": 2024, "baseline": "regulier", "wr_bb100": 8.916666666666666, "duree_s": 310.5166416168213, "rss_mb": 11994.33984375, "pid": 3624}
{"oft": true, "seed": 2024, "baseline": "tag", "wr_bb100": 13.743333333333334, "duree_s": 305.3705542087555, "rss_mb": 12004.68359375, "pid": 3622}
```

### Verdict final P10.A.bis

**Renversement complet de P10.A** : avec tracker frais et IC 95% sur 3 seeds,
**seul TAG** montre un effet OFT statistiquement significatif (+2.96 ± 2.02
bb/100, p<0.05). Tous les autres deltas sont dans le bruit, y compris les
"findings" majeurs de P10.A (Call +47, LAG -9.79, Régulier -7.37) qui
résultaient de la pollution tracker cross-baseline.

**Pourquoi P10.A donnait des chiffres trompeurs** : l'agent réutilisé sur
les 6 baselines en séquence accumulait dans son tracker des stats vpip/pfr
mélangées. Quand l'évaluation arrivait à LAG ou Régulier, le profil détecté
n'était pas "LAG/Régulier pur" mais "moyenne pondérée des 4 baselines
précédentes" → l'exploit appliqué n'était plus pertinent.

**Variance énorme observée** :
- Call-Only σ_ON = 38.37 bb/100 (variance entre seeds : +99, +85, +27)
- LAG σ_ON = 13.65 bb/100 (+0.48, -23.08, +0.66)

Cette variance vient de la stochasticité du blueprint+continuations k=4
(rotation aléatoire selon fingerprint main) et du sampling MCCFR.

### Recommandation P10.B

**Option α retenue : GARDER OFT par défaut.**

Justification :
- TAG : OFT confirmé UTILE (+2.96 bb/100, statistiquement significatif)
- Aléatoire/Call/Raise/LAG/Régulier : effets dans le bruit → OFT n'est pas
  démontré nuisible
- Coût mémoire/CPU OFT négligeable (mixer simple, tracker rolling 30 mains)
- Si on coupe OFT, on perd le gain TAG (le seul significatif)
- Les findings négatifs P10.A sur LAG/Régulier étaient des artefacts protocole

**P10.B passe avec OFT ON par défaut.**

**Note méthodologique** : pour toute future éval comparative, **toujours
recréer l'agent par baseline** (tracker frais) ET utiliser **3+ seeds avec
calcul d'IC 95%**. P10.A est officiellement invalide et son rapport est
conservé uniquement comme contre-exemple méthodologique.
