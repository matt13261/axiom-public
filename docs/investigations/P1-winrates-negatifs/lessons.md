# P1 — Lessons apprises

_Mise a jour au fur et a mesure des experiences._

---

## H3 — Bucketing non-deterministe : ELIMINE comme cause de P1

**Date :** 2026-04-24

### Ce qu'on a confirme

Le bucketing MC etait bien stochastique : meme main + board pouvait retourner
des buckets differents entre deux appels (prouve par le test RED [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]).

La cause racine : `Deck()` dans `calculer_equite()` cree son propre `Random(None)`
(independant de `random.seed()` global) et shuffle le deck. L'ordre du pool de cartes
varie a chaque appel → meme seed global = resultats differents.

Fix applique : `Deck.GetFullDeck()` (ordre fixe) + `random.seed(42)` avant MC.

### Ce qu'on a appris sur l'impact

**L'impact sur les winrates est statistiquement nul.**

Post-fix (seed=42, 100 sims) vs baseline :
- vs aleatoire : -68.9 vs -68.4, delta **-0.5 bb/100** (dans bruit de mesure)
- Autres bots : deltas dans 1-1.5 sigma de la variance baseline

Conclusion : H3 n'est PAS une cause du winrate negatif de P1.
Le bucketing stochastique ajoutait du bruit mais pas de biais systematique.

### Augmenter NB_SIMULATIONS casse la compatibilite blueprint

Passer de 100 → 1000 simulations a cause une regression massive (-100 bb/100 vs
call-only, -5 bb/100 vs random). Cause : le blueprint MCCFR a ete entraine avec
100 simulations. Changer le nombre de sims change les attributions de bucket →
l'agent consulte des infosets differents de ceux entraines → strategie erronee.

**Consequence pour P6 (refonte abstraction) : tout changement de bucketing
(seuils OU nombre de simulations) exige un re-entrainement complet du blueprint.**

### Metrique de coherence intra-run : abandon

Le 67-72% de coherence mesure en Exp 01 reflète la stochasticite GTO normale
d'un agent MCCFR (il echantillonne depuis une distribution de strategie, pas une
action pure). Cette metrique ne mesure pas le determinisme du bucketing.

**Ne plus utiliser la coherence intra-run comme proxy du determinisme pour H1/H6.**

### Seuil +-3 bb/100 trop strict pour evaluations 500 mains

La variance naturelle d'une evaluation 500 mains vs 6 bots est de l'ordre de :
- vs call-only : sigma ≈ 41 bb/100 (tres haute variance, peu de mains decident)
- vs raise/random : sigma ≈ 2-11 bb/100

Un seuil de +-3 bb/100 declenche des faux positifs systemiquement.
Recommendation pour les prochaines experiences : utiliser **+-10 bb/100 comme
seuil de regression significative**, sauf pour aleatoire (sigma=2, seuil +-5 OK).

---

## H3-fix : Valeur residuelle conservee

Meme si H3 ne cause pas P1, le fix seed=42 + Deck.GetFullDeck() a de la valeur :

1. **Reproductibilite** : les evaluations avec meme seed produisent maintenant
   des resultats identiques entre runs (bucket deterministe → infoset stable).
2. **Debuggabilite** : plus facile de reproduire un comportement d'agent.
3. **Prerequis pour de futurs tests unitaires** du comportement postflop.

Le fix est merge sur main (2026-04-24).

---

## Methode de comparaison future : seuil relatif, pas absolu

Le seuil absolu +-3 bb/100 declenchait des faux positifs systematiques
(5/6 bots alertes alors que le delta vs random etait -0.5 bb/100).

**Regle adopte pour toutes les experiences suivantes :**
Alerte si |delta| > 1.5 x sigma_baseline du bot concerne.

Exemple : vs call-only (sigma_baseline = 41.4) → alerte si |delta| > 62 bb/100.
Exemple : vs random   (sigma_baseline =  2.0) → alerte si |delta| >  3 bb/100.

Cela evite les faux positifs sur des bots naturellement volatils (call-only)
tout en restant sensible aux regressions reelles sur les bots stables (random).
