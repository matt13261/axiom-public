================================================================================
                              AXIOM — POKER AI
     Texas Hold'em No-Limit (3 joueurs + Heads-Up)  —  MCCFR + Deep CFR
================================================================================

AXIOM est une IA de poker construite sur deux algorithmes combinés :
  - MCCFR  (Monte Carlo Counterfactual Regret Minimization) → blueprint
  - Deep CFR (Brown et al. 2019) → généralisation par réseaux de neurones

Interface graphique Pygame — 1 humain affronte 2 instances d'AXIOM.


================================================================================
  INSTALLATION
================================================================================

    pip install -r requirements.txt

Dépendances principales : treys, pygame, torch, numpy, tqdm, openpyxl


================================================================================
  STRUCTURE DU PROJET
================================================================================

axiom/
├── main.py                     Point d'entrée — partie humain vs AXIOM
├── train.py                    Entraînement MCCFR + Deep CFR (3 joueurs)
├── train_hu.py                 Entraînement MCCFR Heads-Up
├── recalibrer_3max.py          Recalibrage de l'abstraction cartes 3 joueurs
├── recalibrer_hu.py            Recalibrage de l'abstraction cartes Heads-Up
├── requirements.txt            Dépendances Python
├── README.txt                  Ce fichier
├── TODO.txt                    Tâches à faire / faites (à consulter)
├── ARCHITECTURE.txt            Documentation technique complète du code
├── CLAUDE.md                   Instructions Claude (lecture TODO en session)
├── libratus.txt                Notes de référence sur l'IA Libratus
│
├── config/
│   └── settings.py             Tous les hyperparamètres globaux
│
├── engine/                     Moteur de jeu (règles poker)
│   ├── actions.py              Actions légales (Fold/Check/Call/Raise/All-In)
│   ├── card.py                 Wrapper Treys pour le deck
│   ├── hand_evaluator.py       Évaluation mains + équités Monte Carlo
│   ├── player.py               État d'un joueur (stack, statut, cartes)
│   ├── blind_structure.py      Progression des blindes (tournoi)
│   ├── game_state.py           État complet d'une main
│   └── game.py                 Boucle de jeu complète
│
├── abstraction/                Réduction de l'espace de jeu
│   ├── card_abstraction.py     8 buckets d'équité (preflop + postflop)
│   ├── action_abstraction.py   7 actions abstraites
│   └── info_set.py             Clés d'infoset pour MCCFR
│
├── ai/                         Intelligence artificielle
│   ├── agent.py                Agent AXIOM — décisions en temps réel
│   ├── mccfr.py                CFR External Sampling
│   ├── strategy.py             Sauvegarde/chargement blueprint (pickle)
│   ├── network.py              Réseaux PyTorch (Deep CFR)
│   ├── trainer.py              Entraîneurs PyTorch (Adam + MSE)
│   ├── deep_cfr.py             Orchestration Deep CFR
│   └── reservoir.py            Reservoir Buffers (Vitter 1985)
│
├── interface/                  Interface graphique Pygame
│   ├── main_window.py          Fenêtre principale + boucle événements
│   ├── table.py                Rendu du tapis
│   ├── card_renderer.py        Rendu des cartes
│   ├── hud.py                  HUD (blindes, pot, phase, timer)
│   └── action_buttons.py       Boutons d'action
│
├── solver/                     Solveur temps réel (non finalisé)
│   ├── depth_limited.py        CFR à profondeur limitée (WIP)
│   └── subgame_solver.py       Solveur de sous-jeu (WIP)
│
├── training/                   Utilitaires d'entraînement / éval
│   ├── self_play.py            Simulation de parties (sans GUI)
│   ├── evaluator.py            Mesure du winrate vs bots adverses
│   └── trainer.py              (helpers)
│
├── tests/                      Tests unitaires
│   ├── test_abstraction.py
│   ├── test_agent.py
│   ├── test_deep_cfr.py
│   ├── test_engine.py
│   ├── test_hu.py
│   └── test_mccfr.py
│
└── data/                       Fichiers générés
    ├── strategy/               blueprint_v1.pkl, blueprint_hu.pkl
    ├── models/                 6 réseaux .pt (regret + strategy × 3 joueurs)
    ├── buffers/                Reservoir buffers (.npz)
    ├── checkpoints/            Checkpoints numérotés iteration_XXX/
    └── logs/                   CSV + Excel d'entraînement et d'éval


================================================================================
  COMMANDES UTILES
================================================================================

────────────────────────────────────────────────────────────────────────────────
  LANCER UNE PARTIE (interface graphique)
────────────────────────────────────────────────────────────────────────────────

  python main.py                                  # partie standard
  python main.py --nom "Alice"                    # changer le nom humain
  python main.py --mode deterministe              # argmax au lieu d'échantillonner
  python main.py --mode stochastique              # (défaut) tirage aléatoire
  python main.py --blueprint data/strategy/blueprint_v1.pkl

  Raccourcis clavier in-game :
    F = FOLD     X = CHECK     C = CALL     A = ALL-IN     R = RAISE


────────────────────────────────────────────────────────────────────────────────
  ENTRAÎNEMENT 3 JOUEURS (train.py)
────────────────────────────────────────────────────────────────────────────────

  # Entraînement complet (MCCFR puis Deep CFR)
  python train.py

  # MCCFR seul (blueprint)
  python train.py --mode mccfr
  python train.py --mode mccfr --iterations 100000       # test rapide
  python train.py --mode mccfr --iterations 5000000      # entraînement complet

  # Deep CFR seul
  python train.py --mode deepcfr
  python train.py --mode deepcfr --deep-iterations 500
  python train.py --mode deepcfr --traversees 3000

  # Évaluation
  python train.py --mode eval                            # éval complète
  python train.py --mode eval --nb-mains-eval 10000      # plus précis
  python train.py --mode eval-checkpoint                 # dernier checkpoint → Excel
  python train.py --mode bench                           # benchmark rapide 100 mains

  # Autres options
  python train.py --workers 8                            # forcer nb workers CPU


────────────────────────────────────────────────────────────────────────────────
  ENTRAÎNEMENT HEADS-UP (train_hu.py)
────────────────────────────────────────────────────────────────────────────────

  python train_hu.py                                 # entraînement complet HU
  python train_hu.py --iterations 75000              # itérations par niveau
  python train_hu.py --workers 15                    # nb workers CPU
  python train_hu.py --from-scratch                  # repartir de zéro
  python train_hu.py --chemin data/strategy/blueprint_hu.pkl
  python train_hu.py --verbose
  python train_hu.py --no-verbose


────────────────────────────────────────────────────────────────────────────────
  RECALIBRAGE DE L'ABSTRACTION CARTES
────────────────────────────────────────────────────────────────────────────────

  À relancer uniquement si on modifie la logique de bucket ou le nb de joueurs.

  python recalibrer_3max.py           # recalibre les buckets 3 joueurs
  python recalibrer_hu.py             # recalibre les buckets Heads-Up

  → produit recalibrage_3max_resultats.txt / recalibrage_hu_resultats.txt


────────────────────────────────────────────────────────────────────────────────
  TESTS
────────────────────────────────────────────────────────────────────────────────

  python -m pytest tests/                # tous les tests
  python -m pytest tests/test_mccfr.py   # un seul module
  python -m pytest -v                    # verbose


================================================================================
  FLUX DE TRAVAIL TYPIQUE
================================================================================

  1. (Optionnel) Recalibrer l'abstraction si modifs sur les buckets :
       python recalibrer_3max.py
       python recalibrer_hu.py

  2. Entraîner le blueprint MCCFR :
       python train.py --mode mccfr
       python train_hu.py

  3. Entraîner le Deep CFR par-dessus :
       python train.py --mode deepcfr

  4. Évaluer :
       python train.py --mode eval-checkpoint

  5. Jouer :
       python main.py


================================================================================
  RÉFÉRENCES
================================================================================

  - Brown & Sandholm (2017), "Libratus"                  — Science 2018
  - Brown, Lerer, Gross, Sandholm (2019), "Deep CFR"     — ICML 2019
  - Brown & Sandholm (2019), "Pluribus" (multi-joueurs)  — Science 2019
  - Tammelin (2014), "CFR+"                              — arXiv:1407.5042

  Voir libratus.txt pour les notes internes sur Libratus / Pluribus.

================================================================================
