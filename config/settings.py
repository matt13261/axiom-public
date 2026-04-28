# =============================================================================
# AXIOM — Configuration globale
# =============================================================================

NB_JOUEURS    = 3
STACK_DEPART  = 500

NIVEAUX_BLINDES = [
    ( 10,  20, 6),
    ( 15,  30, 6),
    ( 20,  40, 6),
    ( 30,  60, 6),
    ( 40,  80, 6),
    ( 50, 100, 6),
    ( 60, 120, 6),
    ( 80, 160, 6),
]

# -----------------------------------------------------------------------------
# TAILLES DE MISE PAR STREET
#
# Préflop  : pot (1.0×), 2.5× pot, 3× pot
#            → mises d'ouverture et 3-bets standards en tournoi
#
# Postflop : 35% pot, 65% pot, pot (1.0×)
#            → continuation bets, value bets, bluffs standards
#
# Préflop : 3 éléments → 3 tailles de raise
# Postflop : 5 éléments → NB_ACTIONS_MAX = 9 (FOLD+CHECK+CALL+RAISE×5+ALL_IN)
# -----------------------------------------------------------------------------

TAILLES_MISE_PREFLOP  = [1.0, 2.5, 3.0]    # fractions du pot — préflop
TAILLES_MISE_POSTFLOP = [0.20, 0.35, 0.65, 1.0, 1.5]  # fractions du pot — flop/turn/river

# TAILLES_MISE = liste utilisée par défaut (réseau, agent, abstraction)
# On prend le postflop car c'est là que se jouent la majorité des décisions
TAILLES_MISE = TAILLES_MISE_POSTFLOP

ALL_IN = True

# Validation : modifier TAILLES_MISE nécessite de réentraîner tous les réseaux
# (NB_ACTIONS_MAX change → incompatibilité avec les poids sauvegardés)
assert len(TAILLES_MISE) == 5, (
    f"TAILLES_MISE doit contenir 5 éléments (actuel : {len(TAILLES_MISE)}). "
    "Modifier cette liste nécessite de réentraîner les réseaux Deep CFR."
)

# -----------------------------------------------------------------------------
# PARAMÈTRES MCCFR
# -----------------------------------------------------------------------------

MCCFR_ITERATIONS = 20_000_000
MCCFR_SAVE_EVERY = 100_000

# Nombre maximum de cœurs CPU utilisés pour l'entraînement MCCFR et Deep CFR.
# Laisser quelques cœurs libres pour que le système reste fluide.
NB_WORKERS_MAX   = 13

# -----------------------------------------------------------------------------
# PARAMÈTRES DEEP CFR / RÉSEAU
# -----------------------------------------------------------------------------

DEEP_CFR_ITERATIONS  = 500
DEEP_CFR_TRAVERSALS  = 3_000
BATCH_SIZE           = 4096
# LR de base (utilisé à l'itération 1). Décroit selon 1/sqrt(t) avec
# LR_DECAY_GLOBAL=True (Point 10 — Pluribus warm-start schedule).
LEARNING_RATE        = 3e-4

# Point 10 : schedule global décroissant du LR sur les itérations Deep CFR.
# True  → LR_eff(t) = LEARNING_RATE / sqrt(t), plancher à 1% de LEARNING_RATE.
# False → LR réinitialisé à LEARNING_RATE à chaque itération (comportement actuel).
LR_DECAY_GLOBAL      = True

# Point 9 : perturbation aléatoire du sizing en partie réelle.
# Appliqué uniquement aux RAISE (pas ALL_IN, pas CALL).
# Montant final = round(montant_abstrait × (1 ± U[0, PERTURBATION_SIZING])).
PERTURBATION_SIZING  = 0.08

HIDDEN_SIZE          = 256
NB_COUCHES           = 4

# -----------------------------------------------------------------------------
# CHEMINS DE SAUVEGARDE
# -----------------------------------------------------------------------------

CHEMIN_BLUEPRINT    = "data/strategy/blueprint_v1.pkl"
CHEMIN_BLUEPRINT_HU = "data/strategy/blueprint_hu.pkl"   # Blueprint heads-up (2 joueurs)
CHEMIN_REGRET_NET   = "data/models/regret_net.pt"
CHEMIN_STRATEGY_NET = "data/models/strategy_net.pt"
CHEMIN_LOG          = "data/logs/training_log.csv"

# -----------------------------------------------------------------------------
# OFT — EXPLOIT LOGGING (Exp 04 uniquement — désactiver en production)
# -----------------------------------------------------------------------------

# Active la journalisation de chaque décision exploit dans EXPLOIT_LOG_PATH.
# Mettre False en production (overhead I/O + fichier volumineux).
EXPLOIT_LOG_ENABLED = False  # True uniquement pendant une evaluation
EXPLOIT_LOG_PATH    = (
    "docs/investigations/P1-winrates-negatifs/"
    "experiments/04-H4-exploit/exploit_decisions.jsonl"
)  # Mettre a jour le chemin avant chaque eval

# -----------------------------------------------------------------------------
# INTERFACE GRAPHIQUE
# -----------------------------------------------------------------------------

LARGEUR_FENETRE = 1200
HAUTEUR_FENETRE = 800
FPS             = 60
TITRE_FENETRE   = "AXIOM — Texas Hold'em No Limit"
