# Exp 02 — H1+H6 : Heuristique BTN agressive

_Branche : `exp/P1-h1-heuristique-agressive`_
_Dépend de : Exp 01 terminée (H3-final sur branche dédiée)_

---

## Analyse de la cible : `agent.py:875-946`

### Anatomie de `_heuristique()`

```
NB_ACTIONS_MAX = 9
Indices :
  0 = FOLD
  1 = CHECK
  2 = CALL
  3 = RAISE (taille 1/5 pot)   ← idx_raise_petit
  4 = RAISE (taille 2/5 pot)
  5 = RAISE (taille 3/5 pot)   ← idx_raise_milieu  [= _IDX_RAISE_BASE + 5//2 = 3+2=5]
  6 = RAISE (taille 4/5 pot)
  7 = RAISE (taille 5/5 pot)
  8 = ALL-IN                   ← idx_raise_fort [= _IDX_ALLIN - 1 = 7]
```

> Note : `idx_raise_fort = _IDX_ALLIN - 1 = 8-1 = 7` (RAISE taille 5/5, pas ALL-IN).

### Trois branches actuelles

#### Branche A — SB face à raise (position == 1, face_a_raise)
```python
bucket >= 5 : FOLD=20%, CALL=40%, RAISE_FORT[7]=40%
bucket 3-4  : FOLD=35%, CALL=45%, RAISE_MID[5]=20%
bucket 0-2  : FOLD=65%, CALL=30%, RAISE_PETIT[3]=5%
```
Normalisé identique (somme = 1.0).

#### Branche B — BB face à raise (position == 2, face_a_raise)
```python
bucket >= 5 : FOLD=15%, CALL=50%, RAISE_FORT[7]=35%
bucket 3-4  : FOLD=30%, CALL=55%, RAISE_MID[5]=15%
bucket 0-2  : FOLD=60%, CALL=35%, RAISE_PETIT[3]=5%
```

#### Branche C — `else` (BTN OU toute situation sans raise) ← CIBLE H1
```python
FOLD[0]=0.10, CHECK[1]=0.60, CALL[2]=0.60, RAISE_MID[5]=0.30
→ Normalisé : FOLD=6.3%, CHECK=37.5%, CALL=37.5%, RAISE=18.8%
```

**Problème** : cette branche couvre DEUX cas distincts que le code ne différencie pas :
1. BTN `face_a_raise=True` (adversaire a misé, CHECK illégal) → normalisé CALL=60%, RAISE=30%, FOLD=10%
2. BTN `face_a_raise=False` (opportunité de mise/check)

Contre raise-only, le BTN déclenche la branche C avec `face_a_raise=True` → CALL 60% d'un adversaire qui raise 100% du temps = stratégie très sous-optimale.

### Positions impliquées dans la heuristique

| Position | Code | Branche actuelle | % appels heuristique |
|----------|------|-----------------|---------------------|
| BTN (position 0) | `else` | Toujours branche C | ~33% des décisions heuristique (estimé) |
| SB (position 1) | `face_a_raise and position==1` | Branche A | ~33% |
| BB (position 2) | `face_a_raise and position==2` | Branche B | ~33% |
| BTN sans raise | `else` (face_a_raise=False) | Branche C | inclus BTN |

**Cas manquant** : BTN avec `face_a_raise=True` → traité comme "sans raise" par la branche `else`.

---

## Correctif proposé

Remplacer la branche `else` par une logique différenciée sur `face_a_raise` ET `bucket` :

```python
else:
    if face_a_raise:
        # BTN face à un raise — manquait complètement
        if bucket >= 5:                        # main forte
            vec[_IDX_FOLD]        = 0.10
            vec[_IDX_CALL]        = 0.25
            vec[idx_raise_fort]   = 0.65      # 3-bet value
        elif bucket >= 3:                      # main moyenne
            vec[_IDX_FOLD]        = 0.40
            vec[_IDX_CALL]        = 0.40
            vec[idx_raise_milieu] = 0.20      # float ou 3-bet bluff
        else:                                  # main faible
            vec[_IDX_FOLD]        = 0.75
            vec[_IDX_CALL]        = 0.20
            vec[idx_raise_petit]  = 0.05      # bluff rare
    else:
        # BTN sans raise — opportunity de mise/check
        if bucket >= 5:                        # value bet fort
            vec[_IDX_CHECK]       = 0.20
            vec[_IDX_CALL]        = 0.25
            vec[_IDX_FOLD]        = 0.05
            vec[idx_raise_fort]   = 0.50
        elif bucket >= 3:                      # value bet moyen
            vec[_IDX_CHECK]       = 0.45
            vec[_IDX_CALL]        = 0.15
            vec[_IDX_FOLD]        = 0.05
            vec[idx_raise_milieu] = 0.35
        else:                                  # check/give-up
            vec[_IDX_FOLD]        = 0.10
            vec[_IDX_CHECK]       = 0.70
            vec[_IDX_CALL]        = 0.10
            vec[idx_raise_petit]  = 0.10      # bluff rare
```

### Justification GTO approximative

**BTN face raise, main faible (bucket 0-2) :**
- Raise-only = range infinie → EV(call) < 0 pour mains faibles
- Solution optimale : fold majoritairement, bluff-raise rare
- Poids : FOLD=75%, CALL=20%, RAISE=5% → normalisé FOLD=75%, CALL=20%, RAISE=5%

**BTN face raise, main forte (bucket ≥5) :**
- Contre range large → 3-bet pour isolation et valeur
- Poids : FOLD=10%, CALL=25%, RAISE_FORT=65%

**BTN sans raise, main forte (bucket ≥5) :**
- Opportunité de mise → value bet 50% du temps
- Éviter check trop souvent (laisse la valeur gratuite)

---

## Stack depth et limites de la heuristique

Stack de départ : 25 BB (court — proche push/fold).
À 25 BB, toutes les raises sont proches de l'all-in (pot commits).
`idx_raise_fort = _IDX_ALLIN - 1 = 7` = RAISE taille 5/5 pot ≈ all-in.

**La heuristique ne tient pas compte de la profondeur de stack.**
C'est une simplification acceptable pour ce correctif (H1 cible le fallback,
pas une stratégie push/fold complète).

---

## Critères de succès (stricts, issus du plan.md)

| Critère | Seuil | Action si raté |
|---------|-------|----------------|
| Gain vs random | ≥ +5 bb/100 vs baseline -68.4 | Évaluer si TAG > 0 avant décision |
| TAG winrate | > 0 bb/100 | **REVERTER immédiatement** |
| Kuhn exploitabilité | < 0.005 (si test disponible) | **REVERTER** |
| Dégradation regulier | < -15 bb/100 | Investiguer |

Kill-switch : si après Exp 02, random < -63.4 bb/100 → Exp 04.
