# Projet AXIOM — Instructions Claude

## ⚠️ EN DÉBUT DE CHAQUE SESSION

**LIRE EN PREMIER (dans l'ordre) :**
1. `TODO.txt` — tâches techniques (priorités P1/P2/P3)
2. `docs/journal/TODO.md` — session courante
3. `docs/journal/SPRINT.md` — objectifs du sprint

Rappeler à l'utilisateur les tâches "À FAIRE" de TODO.txt **et** les tâches
ouvertes de docs/journal/TODO.md en début de session.

Format attendu du rappel :
> Rappel TODO AXIOM :
> - P1 : …
> - P2 : …
> Session : …

## Système de journal (à chaque session)

À l'ouverture :
1. Lire `docs/journal/TODO.md`
2. Vérifier `docs/journal/SPRINT.md`

Pendant :
3. Mettre à jour `SPRINT.md` si avancement
4. Mettre à jour `ROADMAP.md` si décision long terme
5. Hook git auto-génère `done/YYYY-MM-DD.md` à chaque commit

Avant fermeture :
6. Mettre à jour `TODO.md` (cocher fait, ajouter nouvelles tâches)
7. Push commits + sync axiom-public si modifs partagées
   (`bash scripts/sync_public.sh` ou automatique via hook)

## Mise à jour du TODO

Après chaque modification effectuée sur le projet :
1. Déplacer la tâche de "À FAIRE" vers "FAIT" avec la date du jour.
2. Ajouter toute nouvelle tâche identifiée dans "À FAIRE" avec sa priorité.
3. Garder le fichier `TODO.txt` propre et à jour.

## Section "NOTES / IDÉES (non planifiées)"

Maintenir activement cette section de `TODO.txt` :
- Dès que l'utilisateur ou moi-même évoquons une idée, une piste, une
  amélioration possible, un doute technique, ou toute réflexion sur le
  projet qui **n'est pas transformée immédiatement en tâche** et sur
  laquelle nous ne revenons pas dans la foulée → l'ajouter à la section
  "NOTES / IDÉES" avec la date du jour.
- Format : `- YYYY-MM-DD — <idée courte en une ligne>`.
- Ne pas demander confirmation avant d'ajouter : le faire silencieusement
  en fin de réponse.
- Si une idée des notes devient planifiée plus tard, la déplacer vers
  "À FAIRE" avec une priorité.
- Si une idée est abandonnée explicitement, la retirer.

## Contexte projet

- **AXIOM** = IA de poker Texas Hold'em No Limit (3 joueurs + Heads-Up).
- Algorithmes : MCCFR (blueprint) + Deep CFR (réseaux neuronaux).
- Langue du projet : **français** (commentaires, docs, variables).
- Voir `README.txt` pour la structure complète et les commandes.

## Fichiers de référence

- `README.txt` — structure projet + commandes
- `TODO.txt` — tâches à faire / faites (⚠️ lire en premier)
- `libratus.txt` — notes sur Libratus / Pluribus (inspiration algorithmique)
- `config/settings.py` — tous les hyperparamètres

## Invariants du projet (NE PAS ROMPRE)

Ces propriétés doivent rester vraies après chaque modification :

1. **Format des clés d'infoset (7 segments)** :
   `PHASE|pos=P|bucket=B|pot=N|stacks=(A,B,C)|hist=H|raise=R`
   - Pour HU : `HU_PHASE|pos=P|bucket=B|pot=N|stacks=(A,B)|hist=H|raise=R` (2 stacks)
   - Toute modification de `abstraction/info_set.py` doit conserver ce format.

2. **DIM_INPUT = 52** — taille du vecteur d'entrée des réseaux Deep CFR.
   Toute modification de l'encodeur (`ai/network.py::encoder_infoset`) doit
   conserver exactement 52 dimensions ou mettre à jour `DIM_INPUT` ET tous
   les réseaux sauvegardés (`data/models/*.pt`).

3. **NB_ACTIONS_MAX = 9** — FOLD+CHECK+CALL+RAISE×5+ALL_IN.
   Changer cette constante invalide tous les buffers et réseaux existants.

4. **Tests toujours verts** : `python -m pytest tests/ -q` doit passer à 100%.
   Utiliser `/poker-regression-check` avant chaque commit important.

5. **Somme des stacks constante** : la somme des stacks de tous les joueurs
   doit rester égale à `NB_JOUEURS × STACK_DEPART` tout au long d'une partie
   (validé par `test_engine.py::test_partie_complete`).

## Workflow recommandé

Pour toute modification de code substantielle :

1. Lire `TODO.txt` → identifier la tâche
2. Modifier le code
3. Lancer `/poker-regression-check` (ou `python -m pytest tests/ -q`)
4. Si vert : committer avec un message clair (`git commit -m "type: description"`)
5. Mettre à jour `TODO.txt` (tâche → FAIT avec la date)

Types de commit : `feat`, `fix`, `refactor`, `test`, `train`, `docs`

## AbstractionCartesV2 — règle d'usage (Phase 2)

- **Auto-chargement** : `V2()` charge automatiquement `DEFAULT_CENTROIDES_PATH`
  (`data/abstraction/centroides_v2.npz`) si le fichier existe. Log INFO traçable.
- **Path personnalisé** : `V2(centroides_path='autre/path.npz')`
  → `FileNotFoundError` immédiat si le fichier n'existe pas.
- **Dict direct** : `V2(centroides=mon_dict)` — utile pour tests / calibration.
- **Sans centroïdes** : si le fichier par défaut est absent et aucun arg fourni,
  `bucket_postflop()` lève `RuntimeError` — comportement intentionnel, pas un bug.
  Un log `WARNING` est émis à l'instanciation.
- **Mock pour tests** : utiliser `mock_centroides()` depuis `tests/conftest.py`
  ```python
  from tests.conftest import mock_centroides
  v2 = AbstractionCartesV2(centroides=mock_centroides())
  ```
- **Discrimination draw vs pair** : garantie uniquement avec centroïdes calibrés
  (post Étape E). Les tests avec mocks vérifient uniquement la plage [0, 49].

## Bugs connus (non bloquants)

- **Décalage stacks HU agent↔training** : l'agent lit les stacks post-blinde
  (EtatJeu), le training HU commence pré-blinde → clés différentes → 0 hits
  sur le blueprint HU en mode test. Ne pas faire échouer les tests sur ce point.
  (cf. `test_hu.py::test_coherence_cle_hu`)

## Limites théoriques connues (non-bugs)

- **Adversaires aléatoires** : pas de pattern stable → OFT ne génère pas
  de gain structurel. Acceptable et attendu théoriquement.

- **Bot raise-only mécanique** : exploit `hyper_agressif` (CHECK-trap 65%)
  ne fonctionne pas contre un agent sans modèle contextuel. Le bot raise
  systématiquement même face à un check. Nécessiterait un exploit multi-street
  (call + re-raise river) — hors scope P1. Voir Exp 05 si prioritaire.

- **Bots GTO-like (TAG/régulier)** : correctement préservés par la zone
  neutre OFT. Ne pas modifier les seuils sans re-évaluation complète.

## Module OFT (Opponent Frequency Tracker) — 2026-04-25

Résout P1 partiellement (5/6 bots) — voir `docs/investigations/P1-winrates-negatifs/SYNTHESE.md`

- `ai/opponent_tracker.py` : rolling window 30 mains, stats vpip/pfr/fold_to_cbet par seat
- `ai/exploit_mixer.py` : 4 profils (calling_station/hyper_agressif/fold_prone/neutre),
  blend `(1-c)*blueprint + c*exploit`, zone neutre stricte
- `ai/agent.py` : `enregistrer_action()`, `obtenir_distribution()`, `_detecter_game_type()`
- `training/self_play.py` : hook OFT après `_appliquer_action` dans `_jouer_tour`
- Profils : `calling_station = vpip>0.6 AND pfr<0.25` · `hyper_agressif = vpip>0.6 AND pfr>0.6`
- 141 tests · tag `p1-resolved`

## Slash commands disponibles

- `/poker-regression-check` — lance pytest et résume les résultats
- `/superpowers-tdd` — guide TDD Red-Green-Refactor (obra/superpowers)
- `/superpowers-debug` — debugging systématique (obra/superpowers)
- `/superpowers-brainstorm` — brainstorming structuré (deprecated → use skill)
- `/skill-tester` — valide un skill alirezarezvani contre les standards qualité

## Skills installés (~/.claude/skills/)

| Skill | Source | Usage |
|-------|--------|-------|
| superpowers | obra/superpowers v5.0.7 | TDD, debug, planning, subagent workflows |
| skill-tester | alirezarezvani/claude-skills | Validation qualité des skills |
| poker-regression-check | AXIOM projet | Régression pytest AXIOM |

**TDD Guard** (nizos/tdd-guard) est actif via hooks (`.claude/settings.json`) :
- Bloque `Write/Edit` sans test préalable qui échoue
- Bloque l'ajout de plusieurs tests simultanément
- Déclenché à `SessionStart`, `UserPromptSubmit`, `PreToolUse`

## Ressources skills (références)

- https://github.com/travisvn/awesome-claude-skills — liste curated de 50+ skills vérifiés
- https://github.com/karanb192/awesome-claude-skills — liste alternative
- https://github.com/obra/superpowers — skills installés (TDD, debug, planning)
- https://github.com/nizos/tdd-guard — enforcement TDD installé via hooks
- https://github.com/alirezarezvani/claude-skills — 232+ skills (skill-tester installé)
