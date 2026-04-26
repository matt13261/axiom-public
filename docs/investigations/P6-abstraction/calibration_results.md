# Calibration centroides_v2 — Resultats

## Parametres

| Parametre         | Valeur                         |
|-------------------|-------------------------------|
| n_spots par street | 4000                          |
| n_clusters         | 50                            |
| n_mc_simulations   | 100                           |
| seed               | 42                            |
| Methode            | K-means sklearn, n_init=10    |
| Date               | 2026-04-26                    |

## Performance

| Config workers | Temps (n=500 x 3) |
|----------------|-------------------|
| n_workers=1    | 8.4s              |
| n_workers=4    | 6.0s (optimal)    |
| n_workers=auto | 6.5s              |

Calibration finale (n=4000 x 3 streets, n_workers=4) :
- Temps MC total : 12.4s
- Temps total (MC + K-means) : 15.9s
- Cloud NON utilise : 100% local (budget 14 euros epargne)

## Fichier genere

`data/abstraction/centroides_v2.npz` — 2 KB

| Street | Shape  | E[HS]           | E[HS2]          | Potentiel        |
|--------|--------|-----------------|-----------------|------------------|
| flop   | (50,3) | [0.016, 0.985]  | [0.014, 0.985]  | [-0.257, +0.436] |
| turn   | (50,3) | [0.019, 0.988]  | [0.016, 0.987]  | [-0.146, +0.330] |
| river  | (50,3) | [0.008, 0.996]  | [0.005, 0.995]  | [0.000,  0.000]  |

## Qualite — Discrimination par type de main

Board test : Qh 8c 3d (flop sec)

| Main              | Type      | Bucket | Commentaire                    |
|-------------------|-----------|--------|--------------------------------|
| Js Ts (OESD)      | draw      | 20     | potentiel positif              |
| Ah Kh (flush draw)| draw      | 49     | backdoor + high cards          |
| 6h 6d             | paire faible | 17  | low equity, peu de potentiel   |
| Qs Td             | top pair  | 14     | equity solide, stable          |
| As Ad             | overpair  | 3      | equity haute, potentiel negatif|
| 2c 7d             | air       | 10     | equity tres basse              |

**Draws != Pairs : OK** (bucket 20 != 17 pour OESD vs paire faible)
**Air != Overpair : OK** (bucket 10 != 3)
**Tous buckets dans [0, 49] : OK**

## Tests GREEN

- E.3 : hand types valides + air != overpair
- B.9v2 : draw JsTs != pair 6h6d (strict, avec vrais centroides)
- 171/171 tests GREEN

## Note river

Le potentiel river est exactement 0.000 pour tous les centroides : correct,
car E[HS_river] - E[HS_current] = 0 quand on est deja a la river.
