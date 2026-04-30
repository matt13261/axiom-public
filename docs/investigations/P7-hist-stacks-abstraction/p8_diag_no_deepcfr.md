# P8.diag — Test diagnostique P7 sans Deep CFR

**Date** : 2026-04-30
**Objectif** : isoler la part du déficit P7 imputable au confound "réseaux Deep CFR
entraînés sous V1, lus avec inputs P7".
**Protocole** : éval P7 (1 seed=42 × 6 baselines × 10K mains) avec patch en
mémoire `agent._reseaux_strategie = None` et `agent._reseaux_valeur = None`.

---

## 1. Patch appliqué (sans commit)

```python
agent = creer_agent(chemin_blueprint='data/strategy/blueprint_p7_4m_cloud.pkl', verbose=False)
agent._reseaux_strategie = None    # désactive Deep CFR strategy nets
agent._reseaux_valeur    = None    # désactive value nets
```

Vérification : log montre `Après patch : reseaux_strategie=absent`. Patch
appliqué correctement, agent ne crash pas.

Pas de commit, pas de modification fichier source. Reverté à la fin (non-event).

---

## 2. Résultats — comparaison directe

| Baseline | P7 normal (rapport P8, seed 42) | **P7 sans Deep CFR (seed 42)** | **Δ (sans DCFR − avec DCFR)** |
|---|---:|---:|---:|
| Aléatoire   | -48.11 | **-117.93** | **-69.82** |
| Call-Only   | -14.55 | -10.85      | +3.70   |
| Raise-Only  | -51.62 | **-696.70** | **-645.08** |
| TAG         | -8.97  | -34.14      | -25.17  |
| LAG         | -17.87 | -102.44     | -84.57  |
| Régulier    | -8.45  | -72.97      | -64.52  |
| **Moyenne** | **-24.93** | **-172.51** | **-147.58** |

---

## 3. Verdict — confound "Deep CFR pollué" RÉFUTÉ

**Les chiffres sans Deep CFR sont catastrophiquement pires** que ceux avec
Deep CFR sur 5 baselines sur 6.

- vs Raise-Only : **-696.70 bb/100** sans DCFR = effondrement total
  (l'agent saigne ~7 BB par main contre un bot mécanique)
- Toutes les baselines empirent sauf Call-Only marginalement

**Conclusion** : les réseaux Deep CFR — même entraînés sous V1 — **aident**
les décisions P7 plus qu'ils ne polluent. Probablement parce que :
- Les outputs réseau ne sont pas aléatoires en input P7 ; le réseau a
  appris des features partiellement transférables (cartes, pot, raise_frac)
  qui restent valides même si le format hist change
- Le fallback alternatif (heuristique simple) est encore plus mauvais que
  Deep CFR pollué
- Le blueprint P7 (3M infosets) ne couvre pas suffisamment la totalité des
  scénarios → quand il miss, Deep CFR sauve la mise plus que l'heuristique

---

## 4. Implication — origine réelle du déficit P7 vs V1

Si Deep CFR n'est pas le coupable, le déficit observé en P8 (V1 +20.32 vs
P7 -24.56 en moyenne) vient d'ailleurs. **Hypothèses à investiguer** :

### H1 — Couverture blueprint P7 insuffisante
Le blueprint P7 a +15.6% d'infosets vs V1 (2.99M vs 2.58M) mais distribués
sur un espace différent. La cap=4 + sizing S/M/L pourraient avoir
sur-agrégé des situations stratégiquement distinctes — l'agent prend la
mauvaise décision moyenne sur des nœuds maintenant fusionnés.

### H2 — Mismatch OFT × abstraction
L'OFT (Opponent Frequency Tracker) a été calibré sur les patterns V1.
Avec P7 produisant des distributions différentes (sizing 3 buckets au lieu
de 5, hist tronqué), les profils détectés et les exploits associés
pourraient ne plus correspondre aux situations réelles.

### H3 — Re-train Deep CFR nécessaire (mais bénéfice marginal selon ce diag)
Si Deep CFR-V1 aide déjà bien P7, un re-train sous P7 améliorerait
probablement encore les chiffres, mais pas autant que prévu par le rapport
P8 initial. Bénéfice attendu : peut-être +5-10 bb/100 sur les bots
structurés, pas la résolution complète du déficit.

### H4 — Granularité sizing trop agressive (`r3 → rL`)
Le mapping Variante B fusionne `r3` (pot raise = standard 3-bet) avec
`r4` (overbet = polarisé). Cette fusion est cohérente en short-stack
(SPR ~1) mais perd l'info de standard 3-bet en stacks plus profonds.
Tester variante C (4 sizings : S/M/L_pot/L_over) pourrait restaurer la
distinction.

---

## 5. Recommandation pour la suite

**Priorité haute** :
1. **Profiler la couverture du blueprint P7** : sur 60K mains de jeu réel
   (P8 eval), mesurer le % de décisions où blueprint MISS et fallback Deep
   CFR/heuristique. Si miss-rate > 30% → couverture insuffisante.
2. **Investiguer H1 (sur-agrégation cap=4)** : tester en local un blueprint
   P7-cap=6 vs P7-cap=4 sur même nb d'iter, comparer winrates. Coût : ~6h
   training local par variante.

**Priorité moyenne** :
3. **Tester variante C 4-sizings** (H4) : modif info_set.py + re-train
   blueprint cloud (P10). Coût : ~10€ + 6-8h.

**Priorité basse / hors scope** :
4. **Re-train Deep CFR sous P7** (P9) : ce diagnostic montre que le bénéfice
   attendu est probablement modéré. À faire seulement si toutes les autres
   pistes échouent.

---

## 6. État technique post-diag

| Item | État |
|---|---|
| Patch appliqué | ✓ en mémoire seulement, pas commit |
| Reverté | ✓ (rien à revert, pas de modif fichier) |
| Branche | `main` |
| Tests | non re-vérifiés (rien modifié) |
| Blueprints intacts | ✓ md5 cohérents |
| Résultats sauvegardés | `data/strategy/p8_eval_p7_no_dcfr_results.json` |

---

## 7. STOP

Verdict diag : **le déficit P7 vs V1 ne vient PAS principalement du confound
Deep CFR**. Origine réelle à investiguer (couverture blueprint, sur-agrégation
sizing, OFT mismatch).

À toi pour décider :
- (a) Profiler couverture blueprint avant tout (rapide, ~30 min local)
- (b) Tester variante C 4-sizings ou cap=6 (long, retraining nécessaire)
- (c) Accepter P7 tel quel comme "régression contrôlée" et passer à autre chose
- (d) Autre piste à creuser
