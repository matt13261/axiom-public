# Exp 04 bis — Correction détection profils via PFR

**Date :** 2026-04-25  
**Branch :** exp/P1-h4-oft-bis-pfr  
**Prérequis :** Exp 04 mergé sur main (tag exp-success/P1-h4-oft-v1)

---

## Problème identifié (Exp 04)

`raise-only` classifié `calling_station` car `vpip = 0.998`.

**Cause :** VPIP (Voluntarily Put money In Pot) = taux d'entrée volontaire
preflop, calculé sur `action >= 2` (CALL **ou** RAISE). Un bot raise-only
a `vpip = 1.0` → dépasse le seuil `calling_station` de 0.60.

`OpponentTracker.vpip()` ne distingue pas caller et raiser → confusion structurelle.

---

## Fix

### 1. Ajouter `pfr(seat_index)` dans `OpponentTracker`

```
pfr = nb_raises_preflop / nb_actions_preflop
Raises preflop : action >= 3 (RAISE / ALL_IN — exclu CALL).
Retourne 0.0 si aucune action preflop observee.
```

### 2. Modifier `ExploitMixer._detecter_profil()`

Ordre de priorité (du plus spécifique au plus général) :

```python
if vpip > 0.6 and pfr > 0.6:
    return 'hyper_agressif'
if vpip > 0.6 and pfr < 0.25:
    return 'calling_station'
if fold_to_cbet > 0.65:
    return 'fold_prone'
return 'neutre'
```

`hyper_agressif` passe **devant** `calling_station` car un joueur
hyper-agressif a souvent vpip élevé (il relance beaucoup = entre dans le pot
beaucoup). Sans ce check prioritaire, il tomberait toujours dans `calling_station`.

### 3. Ajouter exploit `hyper_agressif`

Stratégie contre un joueur qui raise tout :
- Réduire les raises (il re-raise)
- Favoriser CALL (pot contrôle) + FOLD si pas de main forte

```python
_TARGET_CHECK_CALL_HYPER_AGRESSIF = 0.65  # CHECK+CALL → 65%
```

Concrètement : redistribuer la masse des RAISE vers CHECK et CALL
en respectant le support (bp[i] == 0 → result[i] == 0).

---

## Constantes à ajouter / modifier

```python
# OpponentTracker (déjà existantes, inchangées)
CONFIANCE_MIN_MAINS = 5
CONFIANCE_MAX_MAINS = 30

# ExploitMixer (nouvelles)
SEUIL_HYPER_AGRESSIF_VPIP = 0.60   # même base que calling_station
SEUIL_HYPER_AGRESSIF_PFR  = 0.60   # pfr > 0.60 → hyper-agressif
SEUIL_CALLING_STATION_PFR = 0.25   # pfr < 0.25 → calling_station confirmé
_TARGET_CHECK_CALL_HYPER_AGRESSIF = 0.65
```

---

## Tests RED à écrire (3 nouveaux tests)

### Test 10 — raise_only détecté comme hyper_agressif
```
vpip = 1.0 (30 RAISE preflop sur 30 obs)
pfr  = 1.0
→ profil = 'hyper_agressif' (pas 'calling_station')
```

### Test 11 — calling_station requiert pfr bas
```
vpip = 0.8 (24 CALL preflop, 6 FOLD sur 30 obs)
pfr  = 0.0 (aucun raise)
→ profil = 'calling_station'
```

### Test 12 — haut vpip + haut pfr → hyper_agressif (pas calling_station)
```
vpip = 0.8 (24 RAISE + CALL sur 30 obs)
pfr  = 0.7 (21 RAISE sur 30 obs)
→ profil = 'hyper_agressif'
```

---

## Critères de succès Exp 04 bis

| Métrique | Seuil | Justification |
|----------|-------|---------------|
| vs raise-only moy | >= -55 bb/100 | Objectif initial Exp 04 non atteint |
| vs call-only delta | <= 5 bb/100 dégradation | Pas de régression sur le gain phare |
| vs TAG | > 0 | Préservation critique |
| vs Régulier | > 0 | Préservation critique |

---

## Risques

- **Exploit hyper_agressif trop agressif** : si le CALL-pot-control ne tient pas
  face à un 3-bet permanent → risque de stack off. Mitiger par confiance lerp.
- **Over-fit raise_only** : le bot raise-only est un extrême. En partie réelle,
  un hyper-agressif a pfr=0.4-0.6, pas 1.0. Le seuil pfr > 0.6 doit rester
  assez bas pour capturer les joueurs réels agressifs.
- **Bug YAGNI** : ne pas implémenter passive maintenant (prévu mais pas testé).

---

## Procédure TDD

1. Écrire tests 10, 11, 12 → RED
2. Si bypass TDD Guard nécessaire (multi-méthodes indissociables) : bypass #4
3. Ajouter `pfr()` dans `OpponentTracker`
4. Modifier `_detecter_profil()` + ajouter `_exploit_hyper_agressif()` dans `ExploitMixer`
5. Tests 10, 11, 12 → GREEN + 137 existants inchangés
6. Attendre GO pour lancer éval
