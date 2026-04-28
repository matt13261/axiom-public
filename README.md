# axiom-public

Mirror lecture seule du code source AXIOM (IA poker Texas Hold'em No Limit
3-max et Heads-Up).

## Objet

Ce repo expose le code source AXIOM en lecture seule pour permettre à un
**Claude consultant** ([claude.ai](https://claude.ai)) de lire le code via
`web_fetch` lors de revues / brainstormings.

Le repo source de référence est `axiom-private` (non publié). Toute
contribution doit y être faite ; ce repo est synchronisé automatiquement.

## Contenu exposé

- **Code source** : `engine/`, `abstraction/`, `ai/`, `solver/`, `training/`,
  `scripts/`, `config/`
- **Tests** : `tests/`
- **Documentation** : `docs/` (journal, investigations, sprints)
- **Fichiers racine** : `CLAUDE.md`, `README.txt`, `TODO.txt`,
  `ARCHITECTURE.txt`, `train.py`, `train_hu.py`, `main.py`, `pytest.ini`,
  `requirements.txt`, `libratus.txt`, `PLAN_ACTION.txt`

## Contenu NON exposé

- `data/` — poids entraînés, blueprints, checkpoints, buffers, logs
- `screen/` — module de scraping plateforme (non utile à la revue algo)
- Fichiers binaires : `*.pkl`, `*.pt`, `*.npz`, `*.log`, `*.zip`
- Secrets : `.env*`, `*.secrets`, `credentials*`, `*.pem`, `*.key`
- Caches : `__pycache__/`, `.pytest_cache/`, IDE settings

## Usage Claude consultant

Pour qu'un Claude consultant lise un fichier précis :

```
https://github.com/matt13261/axiom-public/blob/main/<chemin>
```

Exemples :
- `https://github.com/matt13261/axiom-public/blob/main/abstraction/info_set.py`
- `https://github.com/matt13261/axiom-public/blob/main/ai/mccfr.py`
- `https://github.com/matt13261/axiom-public/blob/main/CLAUDE.md`

## Sync

Mise à jour automatique via hook `post-commit` du repo privé :
[`scripts/sync_public.sh`](scripts/sync_public.sh).
