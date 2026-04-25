# P1 — Hypothèses (brainstorm structuré)

_Généré 2026-04-24. Classement : coût croissant, probabilité décroissante._

---

## H1 — Heuristique fallback BTN trop passive (37% des décisions)

**Mécanisme causal :**
La branche `else` de `_heuristique()` (agent.py:936-941) couvre BTN **et** toute
situation sans raise, avec ces poids fixes :

```python
vec[_IDX_FOLD]        = 0.10   # 10%
vec[_IDX_CHECK]       = 0.60   # 60%
vec[_IDX_CALL]        = 0.60   # 60%
vec[idx_raise_milieu] = 0.30   # 30%
```

Normalisé : FOLD=6.3 %, CHECK=37.5 %, CALL=37.5 %, RAISE=18.8 %.

Face à un bot raise-only, cette branche s'applique à chaque fois que BTN est actif
sans contrainte de mise, produisant une réponse trop passive (38% call vs un adversaire
qui raise 100% du temps). Le bot raise-only accumule de la valeur sur chaque street.

Problème corollaire : quand l'adversaire raise et que CHECK est illégal, la normalisation
force CALL=60 %, RAISE=30 %, FOLD=10 % — trop de calls passifs face à une agression pure.

**Expérience minimale :**
Modifier les poids de la branche `else` pour augmenter le fold/raise face à agression :

```python
# Proposition : détecter face_a_raise dans la branche else
if face_a_raise:   # BTN face à un raise
    vec[_IDX_FOLD]      = 0.35
    vec[_IDX_CALL]      = 0.35
    vec[idx_raise_fort] = 0.30
else:              # BTN sans mise (opportunity bet/check)
    vec[_IDX_CHECK]       = 0.45
    vec[_IDX_RAISE_BASE]  = 0.40
    vec[_IDX_FOLD]        = 0.15
```

Dev : 1h. Eval : 500 mains × 6 bots ≈ 30 min.

**Signal attendu :**
- vs AgentRaiseOnly : amélioration ≥ +20 bb/100
- vs AgentAleatoire : amélioration ≥ +10 bb/100
- vs AgentTAG : pas de dégradation (< -5 bb/100)

**Correction si validée :** Implémenter les poids différenciés + ajouter un sous-cas
BIG_BLIND optionnel (check gratuit).

---

## H2 — Analyse du hit rate blueprint par type d'adversaire

**Mécanisme causal :**
Le hit rate global est 63%, mais ce chiffre est agrégé sur tous les adversaires. Contre
un bot raise-only, l'historique des actions crée des séquences inhabituelles (r/r/r/r)
qui ne sont jamais apparues lors de l'entraînement MCCFR 3-max standard. Ces infosets
"exotiques" tombent dans le fallback heuristique, amplifiant H1.

Exemple : `PREFLOP|pos=1|bucket=4|pot=5|stacks=(20,20,20)|hist=rrr|raise=2` peut
exister dans le blueprint, mais `hist=rrrr` (4 raises consécutifs) presque jamais.

**Expérience minimale :**
Patch d'instrumentation dans `agent.py:_lookup_blueprint_blende()` pour logger le
hit rate séparément par type d'adversaire pendant une session d'évaluation de 200 mains.

Dev : 1h (logging). Eval : 200 mains vs raise-only ≈ 10 min.

**Signal attendu :**
- Hit rate vs raise-only < 40% (vs 63% global) → confirme l'hypothèse
- Hit rate vs TAG ≈ 70-80% → la chute est adversaire-spécifique

**Correction si validée :**
- Court terme : améliorer H1 (heuristique plus agressive face aux raises répétés)
- Long terme : augmenter la diversité des séquences dans l'entraînement MCCFR

---

## H3 — Bucketing Monte Carlo bruité (100 sims, seed non fixée)

**Mécanisme causal :**
`card_abstraction.py:34` : `NB_SIMULATIONS = 100`, aucune seed fixée.
Pour une même main + board, deux appels successifs à `bucket()` peuvent retourner des
buckets différents (variance de l'estimateur MC). Si pendant l'entraînement MCCFR la
main XY/board ZZZ était toujours en bucket 4, mais qu'en évaluation elle est parfois en
bucket 3, l'agent consulte un infoset incorrect → action sous-optimale.

Impact estimé : un écart de ±1 bucket sur les mains de limite de seuil (~20% des mains).
Chaque mauvais lookup = utilisation de la stratégie d'une main différente.

**Expérience minimale :**
Ajouter `random.seed(42)` avant chaque appel `calculer_equite()` dans
`AbstractionCartes.bucket()`, relancer l'éval.

Dev : 30 min. Eval : 500 mains × 6 bots ≈ 30 min.

**Signal attendu :**
- Réduction de la variance des winrates entre runs (run1 vs run2 < 5 bb/100 d'écart)
- Amélioration du winrate moyen de +5 à +15 bb/100 (estimation conservatrice)

**Correction si validée :**
Fixer `seed=42` dans `calculer_equite()` ou pré-calculer les buckets au début de chaque
main et les cacher jusqu'à la fin.

---

## H4 — Stratégie Nash non-exploitative : cause structurelle (Long terme)

**Mécanisme causal :**
MCCFR converge vers un Nash Equilibrium : une stratégie qui ne peut pas être exploitée
davantage que son exploitabilité résiduelle. Par définition, une stratégie Nash **ne
cherche pas à maximiser les gains contre un adversaire spécifique** — elle cherche à
minimiser le regret maximum.

Un bot call-only est massivement exploitable : il ne fold jamais, donc on devrait le
bluffer 0% du temps et le value-bet 100% du temps. La stratégie Nash ne fait pas ça.

C'est pourquoi AXIOM bat TAG/Régulier (ils ont des fuites standard proches de celles
prévues par Nash) mais perd contre les bots absurdes (leurs fuites sont extrêmes et
hors du modèle Nash).

**Expérience minimale :**
Pas d'expérience courte possible — c'est une propriété structurelle de l'algorithme.
Validation qualitative : examiner la stratégie blueprint pour AA/KK vs call-only :
le blueprint devrait value-bet gros → mesurer si c'est bien le cas avec hit rate ≥ 80%.

Dev : 1h analyse. Eval : N/A.

**Signal attendu :**
N/A — c'est une limite de design, pas un bug.

**Correction (long terme) :**
Implémenter un module d'exploitation : safe-subgame solving adaptatif, ou un simple
"opponent modeling" qui ajuste la stratégie en temps réel selon les patterns adverses
(fréquence de fold, fréquence de call, etc.). Budget : 5-10 jours.

---

## H5 — Mauvais seuils de bucket aux frontières (bruit d'abstraction)

**Mécanisme causal :**
Les seuils des 8 buckets sont calibrés sur une distribution équilibrée, mais les bots
triviaux créent des distributions de boards très différentes :
- Raise-only → pots énormes → boards drawy → équités MC moins stables
- La calibration a été faite sur `recalibrer_3max.py` (seed=42) avec une distribution
  de jeu normale. Sur des bots triviaux, les mains postflop sont sur-représentées
  dans certains buckets (pots géants = très peu d'équité pour la plupart des mains).

Ce déséquilibre peut forcer 60-70% des mains dans bucket 0-2 (très faible équité)
→ l'agent fold massivement des mains qui auraient dû être en bucket 4-5.

**Expérience minimale :**
Logger la distribution des buckets pendant 200 mains vs raise-only et vs TAG.
Comparer la distribution bucket[0-7] entre les deux.

Dev : 30 min (logging). Eval : 200 mains ≈ 5 min.

**Signal attendu :**
- Distribution biaisée vs raise-only (>50% en bucket 0-2) → confirme l'hypothèse
- Distribution équilibrée vs TAG → le biais est adversaire-spécifique

**Correction si validée :**
Recalibrer les seuils sur des matchs vs bots triviaux, ou augmenter le nombre de MC
simulations pour réduire la variance des équités calculées.

---

## H6 — Hit rate + heuristique SB/BB : combinaison toxique

**Mécanisme causal :**
La heuristique SB/BB face à un raise (agent.py:906-934) est plus nuancée que la branche
BTN, mais reste sub-optimale face à un bot qui raise 100% du temps :

- SB, bucket ≥ 5, face raise : FOLD=20%, CALL=40%, RAISE_FORT=40%
- SB, bucket 0-2, face raise : FOLD=65%, CALL=30%, RAISE=5%

Ces poids ont été conçus pour un adversaire "normal". Face à raise-only :
- Bucket 0-2 représente ~40-50% des mains → on fold 65% × 50% = 32.5% des mains SB
  alors qu'un adversaire raise-only a une range très large (bluffs inclus) → on doit
  défendre plus large en EV.

**Expérience minimale :**
Incluse dans H1 (même correctif de l'heuristique). Analyse de l'impact SB spécifiquement
en examinant les statistiques de position dans les résultats d'évaluation.

Dev : inclus dans H1. Eval : lire le CSV de log des évaluations existantes.

**Signal attendu :**
Winrate SB est le plus négatif de toutes les positions → confirme que la défense SB
est sous-optimale.

**Correction :** Intégrée dans H1.

---

## Synthèse priorisée

| # | Hypothèse | Coût dev | Coût compute | P(cause) | Impact estimé |
|---|-----------|----------|--------------|----------|---------------|
| H1 | Heuristique fallback BTN passive | 1h | 30 min | TRÈS ÉLEVÉE | +20 à +40 bb/100 |
| H2 | Hit rate blueprint par adversaire | 1h (logging) | 10 min | HAUTE | Diagnostic |
| H3 | Bucketing MC bruité (seed) | 30 min | 30 min | MOYENNE | +5 à +15 bb/100 |
| H5 | Distribution buckets biaisée | 30 min (logging) | 5 min | MOYENNE | Diagnostic |
| H6 | Heuristique SB/BB sous-défense | inclus H1 | inclus H1 | MOYENNE | inclus H1 |
| H4 | Nash non-exploitatif (structurel) | 5-10 jours | — | CERTAINE | Limite de design |

**Ordre d'exécution recommandé :**
H3 (seed, 30 min) → H1+H6 (heuristique, 1h) → H2+H5 (analyse hit rate, 1h)
→ H4 si budget disponible (nouvelle session, non prioritaire à court terme)

**Point d'arrêt :** Si H1+H3 améliorent le winrate vs random ≥ +30 bb/100, STOP et capitalize.
