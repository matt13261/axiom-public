# Cloud Training P6 — AbstractionCartesV2

## Installation sur la VM

```bash
sudo apt-get update && sudo apt-get install -y python3 python3-pip unzip
pip3 install --break-system-packages -r scripts/cloud_train_p6/requirements.txt
```

## Vérification de l'environnement

```bash
cd ~/axiom
python3 -c "from ai.mccfr import MCCFRHoldEm; print('OK')"
python3 -c "
from abstraction.card_abstraction import AbstractionCartesV2
v2 = AbstractionCartesV2()
print('Centroides:', v2.centroides is not None)
"
```

## Lancement multi-workers (RECOMMANDÉ)

Le script parallélise via `multiprocessing.Process` : N workers tournent
en parallèle, chacun avec son propre `MCCFRHoldEm`. À la fin de chaque batch
les dicts sont fusionnés (`regrets_cumules` + `strategie_somme` + `nb_visites`
additionnés — mathématiquement valide pour CFR).

Sur VM 8 cores (n2-standard-8) :

```bash
cd ~/axiom
nohup python3 -u cloud_train_p6/train_cloud.py \
    --iterations 5000000 \
    --batch-size 1000000 \
    --output blueprint_v2_pilot.pkl \
    --workers 6 \
    > training.log 2>&1 &
echo $! > ~/pilot.pid
tail -f training.log
```

Sur VM 96 cores :

```bash
nohup python3 -u cloud_train_p6/train_cloud.py \
    --iterations 5000000 \
    --batch-size 1000000 \
    --output blueprint_v2_pilot.pkl \
    --workers 90 \
    > training.log 2>&1 &
```

> Note : laisser 6 cores pour OS + fusion entre batches.

## Stratégie batch

- `--batch-size 1000000` : 5 batches pour 5M itérations
- Checkpoint après chaque batch : `blueprint_v2_pilot_batch_N.pkl`
- Mémoire bornée par batch (pic ≈ workers × dict_taille_batch)
- Crash protection : reprise possible depuis le dernier checkpoint

## Paramètres

| Flag | Défaut | Rôle |
|---|---|---|
| `--iterations` | requis | Total d'itérations MCCFR |
| `--batch-size` | 1_000_000 | Itérations par batch (fusion + checkpoint après) |
| `--workers` | 1 | Nombre de processes parallèles (1 = inline, sans spawn) |
| `--output` | requis | Chemin final du blueprint |
| `--stacks` | 1500 | Stacks initiaux des joueurs |
| `--checkpoint-every` | 500_000 | (legacy, conservé pour compat) |

## Backward compat

`--workers 1` exécute en inline (pas de `mp.Process` spawné) — comportement
identique à l'ancienne version mono-thread.

## Suivi de progression

```bash
tail -f training.log
ls -lh ~/axiom/blueprint_v2_pilot*.pkl
```

## Kill switch

```bash
kill -SIGTERM $(cat ~/pilot.pid)
```

## Récupérer le blueprint

```bash
gcloud compute scp axiom-training:~/axiom/blueprint_v2_pilot.pkl \
    data/strategy/ --zone=europe-west4-a
```

## Arrêter la VM

```bash
gcloud compute instances stop axiom-training --zone=europe-west4-a
```
