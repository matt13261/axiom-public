# P10.A.bis cloud — variance OFT

Mesure le delta_OFT par baseline avec tracker frais (agent recree par baseline)
sur 36 runs : 6 baselines x 3 seeds x 2 conditions x 1500 mains.

## Lancement

```bash
bash scripts/cloud_train_p10/run_cloud.sh
```

Demarre la VM `axiom-training-24`, sync le code, lance dans tmux `p10abis`.

## Suivre en direct

```bash
gcloud compute ssh axiom-training-24 --zone=europe-west4-a --command='tmux attach -t p10abis'
```

(Ctrl-B puis D pour detacher sans tuer.)

## Sortie

- `p10abis.log` : sortie complete (1 ligne JSON par run, plus tableau final)
- `p10abis_runs.jsonl` : log JSONL des 36 runs
- `docs/investigations/P10/audit_oft.md` : rapport markdown auto-genere

## Contrainte memoire

RSS cap 7500 MB / worker, hard exit si depasse. 8 workers x 10 GB = 80 GB / 128 GB VM.
