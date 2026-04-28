# =============================================================================
# AXIOM — ai/network.py
# Réseaux de neurones pour Deep CFR (Phase 4).
#
# Deep CFR (Brown et al., 2019) remplace les tables de regrets/stratégies de
# MCCFR par des réseaux de neurones. Au lieu de stocker un vecteur de regrets
# par infoset (infaisable à l'échelle du Texas Hold'em complet), on entraîne :
#
#   ReseauRegret    : f_θ(infoset) → vecteur de regrets contrefactuels
#                     Sortie non bornée — regrets peuvent être positifs ou négatifs
#
#   ReseauStrategie : g_φ(infoset) → distribution de probabilités sur les actions
#                     Sortie softmax — représente la stratégie moyenne cumulée
#
# ─────────────────────────────────────────────────────────────────────────────
# ENCODAGE DE L'INFOSET EN VECTEUR DE FEATURES (DIM_INPUT = 52)
# ─────────────────────────────────────────────────────────────────────────────
#
# L'état interne de MCCFR (dict) est converti en vecteur float de taille fixe :
#
#   [0:4]   phase          — one-hot (PREFLOP, FLOP, TURN, RIVER)
#   [4:7]   position       — one-hot (BTN=0, SB=1, BB=2)
#   [7:11]  card_texture   — Point 3 : 4 scalaires [equite_courante, nut_adv,
#                            draw_pot, equite_preflop]. Remplace l'ancien
#                            one-hot bucket 8 dims → info continue plus riche.
#   [11]    pot            — pot normalisé par la grande blinde (log-scale)
#   [12:15] stacks[0,1,2]  — stacks normalisés (log-scale)
#   [15]    oop            — N°16 : 1.0 si le joueur est OOP (SB structural)
#   [16:48] hist_phases    — Point 3 : pour chaque phase (×4), comptage des
#                            8 types d'action — 4 × 8 = 32 dims.
#                            Layout/phase : [f, x, c, a, r1, r2, r3, r4]
#                            r{bucket} = raise classé par fraction pot (1–4).
#                            Ancien encodage 5 dims (f/x/c/r/a) → 8 dims :
#                            un seul compteur 'r' remplacé par 4 compteurs
#                            de sizing, ce qui permet au réseau de distinguer
#                            un min-raise (r1) d'un pot-raise (r3).
#   [48:52] raise_frac     — one-hot partiel (4 bits) du bucket raise/pot
#                            face auquel le joueur doit agir maintenant.
#                            Niveau 0 = all zeros, niveaux 1-4 = bit actif.
#
# Total : 4 + 3 + 4 + 1 + 3 + 1 + 32 + 4 = 52 dimensions
#
# ─────────────────────────────────────────────────────────────────────────────
# CORRECTIONS APPORTÉES
# ─────────────────────────────────────────────────────────────────────────────
#
# N°8  : bucket one-hot (8 dims) au lieu du scalaire normalisé (1 dim).
#        Le scalaire bucket/7 ne permettait pas au réseau de distinguer
#        des comportements discontinus entre buckets adjacents.
#        DIM_BUCKET : 1 → 8. DIM_INPUT : 32 → 39.
#
# N°10 : dropout 0.1 → 0.05 dans les blocs résiduels.
#        En début d'entraînement Deep CFR, les buffers contiennent peu de
#        données variées. Un dropout de 0.1 aggravait la sous-représentation
#        des situations rares (ex : SB face à 3-bet).
#
# N°14 : normalisation historique /10.0 → /5.0.
#        En situations de 3-bet/4-bet face aux semi-pros LAG, le comptage
#        de 'r' peut dépasser 3–4 en une seule street. /5.0 est plus adapté
#        à l'espace de jeu à 3 joueurs avec re-raises fréquents.
#
# N°16 : ajout d'un bit OOP (1 dim).
#        1.0 si le joueur est en position OOP (doit agir en premier postflop)
#        face à au moins un joueur actif. 0.0 sinon. Aide le réseau à
#        apprendre que SB est structurellement désavantagé à 3 joueurs.
#
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE RÉSEAU
# ─────────────────────────────────────────────────────────────────────────────
#
# Entrée : DIM_INPUT = 52  (Point 3 : hist 20→32 dims + raise_frac 4 dims)
# Couches cachées : NB_COUCHES × HIDDEN_SIZE (depuis config/settings.py)
#   → Par défaut : 4 couches × 256 unités
#   → LayerNorm + ReLU + Dropout(0.05)   ← N°10
# Sortie : NB_ACTIONS_MAX (FOLD, CHECK, CALL, RAISE×N, ALL_IN)
#
# Référence : "Deep Counterfactual Regret Minimization"
#             N. Brown, A. Lerer, S. Gross, T. Sandholm — ICML 2019
# =============================================================================

import math
import torch
import torch.nn as nn
import numpy as np
from typing import List

from config.settings import HIDDEN_SIZE, NB_COUCHES, TAILLES_MISE, NB_WORKERS_MAX
from abstraction.info_set import _discretiser_raise_frac


# =============================================================================
# CONSTANTES D'ENCODAGE
# =============================================================================

# Dimensions de chaque bloc du vecteur de features
DIM_PHASE       = 4    # one-hot : PREFLOP(0), FLOP(1), TURN(2), RIVER(3)
DIM_POSITION    = 3    # one-hot : BTN(0), SB(1), BB(2)
DIM_CARD        = 4    # texture carte : [equite_courante, nut_adv, draw_pot, equite_preflop]
                        # Remplace l'ancien one-hot bucket (8 dims → 4 dims scalaires).
                        # Avantage : le réseau reçoit l'équité brute + des features
                        # orthogonales (avantage aux nuts, potentiel de draw) plutôt
                        # qu'une discrétisation en 8 cases avec perte d'information.
DIM_BUCKET      = DIM_CARD   # alias de compatibilité (ancien nom avant refactorisation)
DIM_POT         = 1    # log(1 + pot / grande_blinde)
DIM_STACKS      = 3    # log(1 + stack_i / grande_blinde) pour les 3 joueurs
DIM_OOP         = 1    # N°16 : 1.0 si joueur OOP face à tous les actifs
DIM_RAISE_FRACS = 4    # one-hot bucket raise/pot sur 5 niveaux — ajoute la taille
                        # de la mise face à laquelle on doit agir dans les features.
                        # 5 niveaux → 4 bits suffisent (niveau 0 = all zeros).

# Historique : pour chaque phase (PREFLOP, FLOP, TURN, RIVER),
# compter les occurrences de chaque type d'action dans l'historique
_NB_PHASES     = 4   # 4 streets
_NB_TYPES_HIST = 8   # f, x, c, a, r1, r2, r3, r4  (Point 3 : sizing raises)
DIM_HIST       = _NB_PHASES * _NB_TYPES_HIST   # = 32

# Dimension totale du vecteur d'entrée
DIM_INPUT = (DIM_PHASE + DIM_POSITION + DIM_CARD + DIM_POT
             + DIM_STACKS + DIM_OOP + DIM_HIST + DIM_RAISE_FRACS)
# = 4 + 3 + 4 + 1 + 3 + 1 + 32 + 4 = 52  (Point 3 : sizing raises → DIM_HIST 20→32)

# Dimension de sortie = nombre d'actions abstraites maximum
# FOLD(1) + CHECK(1) + CALL(1) + RAISE×len(TAILLES_MISE) + ALL_IN(1)
NB_ACTIONS_MAX = 3 + len(TAILLES_MISE) + 1   # = 9 avec 5 tailles de mise

# Mapping des codes d'action historique vers leur index dans le vecteur de comptage.
# Nouveau format (Point 3) : les raises sont encodés avec leur bucket de sizing.
# Chaque phase utilise 8 dims : [f, x, c, a, r1, r2, r3, r4]
# Les codes simples (f/x/c/a) occupent les indices 0-3.
# Les codes 'r{bucket}' (2 chars) occupent les indices 4-7 (bucket 1-4).
_CODE_HIST_INDEX = {'f': 0, 'x': 1, 'c': 2, 'a': 3}
# r1→4, r2→5, r3→6, r4→7 (traités spécialement dans encoder_infoset)

# Nombre de buckets raise_frac (pour le one-hot partiel — niveau 0 = all zeros)
_NB_RAISE_BUCKETS = 5   # 0=no-raise, 1=micro, 2=small, 3=pot, 4=over

# Ordre d'action postflop : SB(1) en premier → OOP
_ORDRE_POSTFLOP_PREMIER = 1   # SB agit en premier postflop → OOP

# N°14 : facteur de normalisation de l'historique
_HIST_NORM = 5.0


# =============================================================================
# DÉTECTION DU DEVICE (GPU / CPU)
# =============================================================================

def detecter_device() -> torch.device:
    """
    Détecte et retourne le device d'entraînement disponible.

    Ordre de priorité :
        1. CUDA  (NVIDIA GPU — torch.cuda.is_available())
        2. MPS   (Apple Silicon — torch.backends.mps.is_available())
        3. CPU   (fallback universel)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        nom    = torch.cuda.get_device_name(0)
        print(f"  🚀 Device : CUDA — {nom}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("  🍎 Device : MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        torch.set_num_threads(NB_WORKERS_MAX)
        nb_threads = torch.get_num_threads()
        print(f"  🖥️  Device : CPU ({nb_threads} threads)")
    return device


# Instance globale du device (initialisée une seule fois)
DEVICE: torch.device = detecter_device()


# =============================================================================
# ENCODEUR D'INFOSET
# =============================================================================

def encoder_infoset(etat: dict, joueur_idx: int) -> np.ndarray:
    """
    Convertit l'état interne de MCCFR en vecteur de features numpy float32.

    C'est la fonction la plus appelée de tout Deep CFR (une fois par nœud
    visité dans la traversée). Elle est optimisée pour la vitesse.

    Paramètres
    ----------
    etat        : dict interne de MCCFRHoldEm._dealer_aleatoire()
                  Clés utilisées : 'phase', 'buckets', 'equites', 'pot',
                                   'grande_blinde', 'stacks', 'hist_phases',
                                   'raise_fracs'
                  Clé 'equites'     : float equite_courante par joueur/phase (opt)
                  Clé 'raise_fracs' : list de 4 floats [preflop,flop,turn,river] (opt)
    joueur_idx  : int (0, 1 ou 2) — joueur dont on construit la vue subjective

    Retourne
    --------
    np.ndarray de forme (DIM_INPUT,) = (52,), dtype float32

    Encodage :
      [0:4]   phase one-hot
      [4:7]   position one-hot
      [7:11]  card texture : [equite, nut_adv, draw_pot, equite_preflop]
      [11]    pot log-normalisé
      [12:15] stacks log-normalisés
      [15]    bit OOP
      [16:48] historique par phase (32 dims = 8/phase : f,x,c,a,r1,r2,r3,r4)
      [48:52] raise_frac one-hot partiel (4 dims, niveau 0 = all zeros)
    """
    vec = np.zeros(DIM_INPUT, dtype=np.float32)
    offset = 0

    # ── 1. Phase one-hot (4 dims) ─────────────────────────────────────────
    phase = etat['phase']   # int 0..3
    if 0 <= phase < 4:
        vec[offset + phase] = 1.0
    offset += DIM_PHASE   # offset = 4

    # ── 2. Position one-hot (3 dims) ──────────────────────────────────────
    pos = joueur_idx   # 0=BTN, 1=SB, 2=BB
    if 0 <= pos < 3:
        vec[offset + pos] = 1.0
    offset += DIM_POSITION   # offset = 7

    # ── 3. Texture cartes (4 dims scalaires) ──────────────────────────────
    # [equite_courante, nut_adv, draw_pot, equite_preflop]
    # equite_courante  : équité MC à la phase courante (0–1)
    # nut_adv          : bonus si équité courante >> équité preflop (main améliorée)
    # draw_pot         : potentiel résiduel (diff equite_river - equite_courante)
    # equite_preflop   : équité preflop (contexte de la main de départ)
    equites = etat.get('equites')   # list[list[float]] ou None
    if equites is not None:
        eq_pre  = equites[joueur_idx][0]   # phase 0 = PREFLOP
        eq_now  = equites[joueur_idx][phase]
        # équité estimée river (phase 3) — borne sup du potentiel
        eq_riv  = equites[joueur_idx][3]
        nut_adv  = max(0.0, eq_now - eq_pre)          # amélioration vs preflop
        draw_pot = max(0.0, eq_riv - eq_now)           # potentiel résiduel
        vec[offset]     = eq_now
        vec[offset + 1] = nut_adv
        vec[offset + 2] = draw_pot
        vec[offset + 3] = eq_pre
    else:
        # Fallback : utiliser le bucket normalisé si equites absent
        bucket = etat['buckets'][joueur_idx][phase]   # int 0..7
        eq_approx = (bucket + 0.5) / 8.0
        vec[offset] = eq_approx
        vec[offset + 3] = eq_approx
    offset += DIM_CARD   # offset = 11

    # ── 4. Pot log-normalisé (1 dim) ──────────────────────────────────────
    gb = max(etat['grande_blinde'], 1)
    vec[offset] = math.log1p(etat['pot'] / gb)
    offset += DIM_POT   # offset = 12

    # ── 5. Stacks log-normalisés (3 dims) ─────────────────────────────────
    for i in range(3):
        vec[offset + i] = math.log1p(etat['stacks'][i] / gb)
    offset += DIM_STACKS   # offset = 15

    # ── 6. Bit OOP (1 dim) — N°16 ─────────────────────────────────────────
    if joueur_idx == _ORDRE_POSTFLOP_PREMIER:
        vec[offset] = 1.0
    offset += DIM_OOP   # offset = 16

    # ── 7. Historique par phase (32 dims) — N°14 + Point 3 ───────────────
    # Layout par phase (8 dims) : [f, x, c, a, r1, r2, r3, r4]
    # Les codes 'r{digit}' sont encodés sur 2 caractères.
    for p in range(4):
        hist_str = etat['hist_phases'][p]
        base     = offset + p * _NB_TYPES_HIST
        i = 0
        while i < len(hist_str):
            ch = hist_str[i]
            if (ch == 'r'
                    and i + 1 < len(hist_str)
                    and hist_str[i + 1].isdigit()):
                # Code 2 chars 'r{bucket}' — Point 3 : sizing raise
                bucket = int(hist_str[i + 1])   # 1–4
                if 1 <= bucket <= 4:
                    vec[base + 3 + bucket] += 1.0   # r1→4, r2→5, r3→6, r4→7
                i += 2
            else:
                idx = _CODE_HIST_INDEX.get(ch, -1)
                if idx >= 0:
                    vec[base + idx] += 1.0
                i += 1
    vec[offset : offset + DIM_HIST] /= _HIST_NORM
    offset += DIM_HIST   # offset = 48

    # ── 8. Raise fraction one-hot partiel (4 dims) ────────────────────────
    # raise_fracs[p] = fraction (mise_courante/pot) au moment de la décision
    # dans la phase p. Niveau 0 (pas de raise) → all zeros.
    # Niveaux 1–4 → bit à l'index niveau-1 (donc 4 bits suffisent).
    raise_fracs = etat.get('raise_fracs')
    if raise_fracs is not None:
        frac  = raise_fracs[phase]
        niveau = _discretiser_raise_frac(frac)
        if 1 <= niveau <= 4:
            vec[offset + niveau - 1] = 1.0
    # offset final = 52

    return vec


def encoder_infosets_batch(etats_joueurs: list) -> torch.Tensor:
    """
    Encode une liste de (etat, joueur_idx) en batch PyTorch.

    Paramètres
    ----------
    etats_joueurs : liste de tuples (etat: dict, joueur_idx: int)

    Retourne
    --------
    torch.Tensor de forme (N, DIM_INPUT), dtype float32, sur DEVICE
    """
    tableau = np.stack(
        [encoder_infoset(e, j) for e, j in etats_joueurs],
        axis=0
    )
    return torch.from_numpy(tableau).to(DEVICE)


# =============================================================================
# BLOC RÉSIDUEL (composant partagé)
# =============================================================================

class BlocResiduel(nn.Module):
    """
    Bloc résiduel léger avec LayerNorm, ReLU et Dropout.

    Connexion résiduelle : x → LayerNorm(x) → Linear → ReLU → Dropout → + x

    N°10 : dropout par défaut réduit à 0.05 (était 0.1).
    En début d'entraînement Deep CFR, les buffers contiennent peu de données
    variées. Un dropout trop fort aggravait la sous-représentation des
    situations rares (défense de blind SB, 3-bet squeeze, etc.).
    """

    def __init__(self, dim: int, dropout: float = 0.05):   # N°10 : 0.1 → 0.05
        """
        dim     : taille des couches (HIDDEN_SIZE)
        dropout : taux de dropout (N°10 : 0.05 par défaut)
        """
        super().__init__()
        self.norm    = nn.LayerNorm(dim)
        self.linear  = nn.Linear(dim, dim)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant avec connexion résiduelle (pre-norm style)."""
        return x + self.dropout(self.relu(self.linear(self.norm(x))))


# =============================================================================
# RÉSEAU DE REGRETS (RegretNet)
# =============================================================================

class ReseauRegret(nn.Module):
    """
    Réseau de neurones qui approxime les regrets contrefactuels instantanés.

    Pour un infoset I et un joueur i, prédit :
        r_i(I, a) ≈ V_i(I, do(a)) − V_i(I)    ∀ action a

    Les regrets peuvent être positifs ou négatifs.
    → Couche de sortie SANS activation (régression non bornée).

    Architecture :
        Linear(DIM_INPUT, HIDDEN_SIZE)
        → NB_COUCHES blocs résiduels (LayerNorm + Linear + ReLU + Dropout(0.05))
        → LayerNorm(HIDDEN_SIZE)
        → Linear(HIDDEN_SIZE, NB_ACTIONS_MAX)
    """

    def __init__(self,
                 dim_input:      int   = DIM_INPUT,
                 hidden_size:    int   = HIDDEN_SIZE,
                 nb_couches:     int   = NB_COUCHES,
                 nb_actions:     int   = NB_ACTIONS_MAX,
                 dropout:        float = 0.05):   # N°10 : 0.1 → 0.05
        super().__init__()

        self.projection_entree = nn.Sequential(
            nn.Linear(dim_input, hidden_size),
            nn.ReLU(),
        )

        self.blocs = nn.ModuleList([
            BlocResiduel(hidden_size, dropout=dropout)
            for _ in range(nb_couches)
        ])

        self.sortie = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, nb_actions),
        )
        # Pas d'activation finale → régression non bornée pour les regrets

        self._initialiser_poids()

    def _initialiser_poids(self):
        """Initialisation Xavier pour toutes les couches linéaires."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant.

        x       : Tensor (batch, DIM_INPUT)
        retour  : Tensor (batch, NB_ACTIONS_MAX) — regrets instantanés
        """
        h = self.projection_entree(x)
        for bloc in self.blocs:
            h = bloc(h)
        return self.sortie(h)

    def predire_strategie(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prédit la stratégie par Regret Matching sur les regrets du réseau.

        x       : Tensor (batch, DIM_INPUT)
        retour  : Tensor (batch, NB_ACTIONS_MAX) — probabilités (somme = 1)
        """
        with torch.no_grad():
            regrets   = self.forward(x)
            regrets_p = torch.clamp(regrets, min=0.0)
            somme     = regrets_p.sum(dim=-1, keepdim=True)

            uniforme = torch.full_like(regrets_p, 1.0 / regrets_p.shape[-1])
            masque   = (somme > 1e-8).expand_as(regrets_p)
            strat    = torch.where(masque, regrets_p / (somme + 1e-10), uniforme)

        return strat

    def nb_parametres(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"ReseauRegret("
                f"input={DIM_INPUT}, hidden={HIDDEN_SIZE}×{NB_COUCHES}, "
                f"output={NB_ACTIONS_MAX}, "
                f"params={self.nb_parametres():,})")


# =============================================================================
# RÉSEAU DE STRATÉGIE (StrategyNet)
# =============================================================================

class ReseauStrategie(nn.Module):
    """
    Réseau de neurones qui approxime la stratégie MOYENNE cumulée.

    Pour un infoset I et un joueur i, prédit :
        σ̄_i(a|I) ≈ stratégie moyenne sur toutes les itérations précédentes

    C'est la stratégie utilisée à la FIN de l'entraînement pour jouer.
    → Couche de sortie avec Softmax (distribution de probabilités).

    Architecture : identique à ReseauRegret, sauf la couche de sortie (Softmax).
    """

    def __init__(self,
                 dim_input:   int   = DIM_INPUT,
                 hidden_size: int   = HIDDEN_SIZE,
                 nb_couches:  int   = NB_COUCHES,
                 nb_actions:  int   = NB_ACTIONS_MAX,
                 dropout:     float = 0.05):   # N°10 : 0.1 → 0.05
        super().__init__()

        self.projection_entree = nn.Sequential(
            nn.Linear(dim_input, hidden_size),
            nn.ReLU(),
        )

        self.blocs = nn.ModuleList([
            BlocResiduel(hidden_size, dropout=dropout)
            for _ in range(nb_couches)
        ])

        self.sortie = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, nb_actions),
            nn.Softmax(dim=-1),
        )

        self._initialiser_poids()

    def _initialiser_poids(self):
        """Initialisation Xavier pour toutes les couches linéaires."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant.

        x       : Tensor (batch, DIM_INPUT)
        retour  : Tensor (batch, NB_ACTIONS_MAX) — probabilités (somme = 1)
        """
        h = self.projection_entree(x)
        for bloc in self.blocs:
            h = bloc(h)
        return self.sortie(h)

    def nb_parametres(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"ReseauStrategie("
                f"input={DIM_INPUT}, hidden={HIDDEN_SIZE}×{NB_COUCHES}, "
                f"output={NB_ACTIONS_MAX}, "
                f"params={self.nb_parametres():,})")


# =============================================================================
# RÉSEAU DE VALEUR (ValueNet) — Point 6
# =============================================================================

class ReseauValeur(nn.Module):
    """
    Réseau de neurones qui approxime la valeur (utilité espérée) d'un infoset.

    Pour un infoset I et un joueur i, prédit :
        V_i(I) ≈ utilité espérée sous la stratégie courante

    Utilisé comme oracle feuille dans le solveur depth-limited :
    au lieu de continuer la traversée jusqu'au showdown, on s'arrête
    à une profondeur donnée et on évalue avec ce réseau.

    → Couche de sortie SANS activation (régression scalaire non bornée).
    Architecture identique à ReseauRegret, sauf sortie = 1 scalaire.
    """

    def __init__(self,
                 dim_input:   int   = DIM_INPUT,
                 hidden_size: int   = HIDDEN_SIZE,
                 nb_couches:  int   = NB_COUCHES,
                 dropout:     float = 0.05):
        super().__init__()

        self.projection_entree = nn.Sequential(
            nn.Linear(dim_input, hidden_size),
            nn.ReLU(),
        )

        self.blocs = nn.ModuleList([
            BlocResiduel(hidden_size, dropout=dropout)
            for _ in range(nb_couches)
        ])

        self.sortie = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),   # scalaire — utilité espérée
        )

        self._initialiser_poids()

    def _initialiser_poids(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : Tensor (batch, DIM_INPUT)
        retour  : Tensor (batch, 1) — valeur scalaire
        """
        h = self.projection_entree(x)
        for bloc in self.blocs:
            h = bloc(h)
        return self.sortie(h)

    def predire(self, x: torch.Tensor) -> torch.Tensor:
        """Prédit la valeur scalaire (sans gradient)."""
        with torch.no_grad():
            return self.forward(x).squeeze(-1)   # (batch,)

    def nb_parametres(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"ReseauValeur("
                f"input={DIM_INPUT}, hidden={HIDDEN_SIZE}×{NB_COUCHES}, "
                f"output=1, params={self.nb_parametres():,})")


# =============================================================================
# UTILITAIRES TORCH
# =============================================================================

def sauvegarder_reseau(modele: nn.Module, chemin: str) -> None:
    """Sauvegarde les poids d'un réseau PyTorch."""
    import os
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    torch.save(modele.state_dict(), chemin)
    taille_ko = os.path.getsize(chemin) // 1024
    print(f"  💾 Réseau sauvegardé : {chemin} ({taille_ko:,} Ko)")


def charger_reseau(modele: nn.Module, chemin: str,
                   device: torch.device = None) -> nn.Module:
    """Charge les poids d'un réseau PyTorch depuis un fichier .pt."""
    if device is None:
        device = DEVICE
    state = torch.load(chemin, map_location=device, weights_only=True)
    modele.load_state_dict(state)
    modele.to(device)
    print(f"  📂 Réseau chargé : {chemin}")
    return modele


def creer_reseaux(device: torch.device = None) -> tuple:
    """
    Crée et retourne les 6 réseaux nécessaires pour Deep CFR à 3 joueurs.

    Retourne
    --------
    tuple : (
        [ReseauRegret_J0,    ReseauRegret_J1,    ReseauRegret_J2],
        [ReseauStrategie_J0, ReseauStrategie_J1, ReseauStrategie_J2]
    )
    """
    if device is None:
        device = DEVICE

    reseaux_regret    = [ReseauRegret().to(device)    for _ in range(3)]
    reseaux_strategie = [ReseauStrategie().to(device) for _ in range(3)]

    return reseaux_regret, reseaux_strategie


# =============================================================================
# AFFICHAGE / DIAGNOSTICS
# =============================================================================

def afficher_info_reseaux(
        reseaux_regret:    List['ReseauRegret'],
        reseaux_strategie: List['ReseauStrategie']) -> None:
    """Affiche un résumé des réseaux créés (paramètres, device, dimensions)."""
    print(f"\n{'═'*60}")
    print(f"  AXIOM — Deep CFR | Architecture des réseaux")
    print(f"{'═'*60}")
    print(f"  Device  : {DEVICE}")
    print(f"  Entrée  : {DIM_INPUT} dims ({DIM_PHASE} phase + {DIM_POSITION} pos + "
          f"{DIM_CARD} card_tex + {DIM_POT} pot + {DIM_STACKS} stacks + "
          f"{DIM_OOP} oop + {DIM_HIST} hist + {DIM_RAISE_FRACS} raise_frac)")
    print(f"  Couches : {NB_COUCHES} blocs résiduels × {HIDDEN_SIZE} unités "
          f"(dropout={0.05})")
    print(f"  Sortie  : {NB_ACTIONS_MAX} actions "
          f"({len(TAILLES_MISE)} raises + FOLD + CHECK + CALL + ALL_IN)")
    print()
    for i in range(3):
        rr = reseaux_regret[i]
        rs = reseaux_strategie[i]
        print(f"  Joueur {i} | RegretNet    : {rr.nb_parametres():,} paramètres")
        print(f"           | StrategyNet  : {rs.nb_parametres():,} paramètres")
    print(f"{'═'*60}\n")


# =============================================================================
# TEST RAPIDE (si exécuté directement)
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  AXIOM — Test réseau (corrections N°8, N°10, N°14, N°16)")
    print("="*60)

    # Créer les réseaux
    r_nets, s_nets = creer_reseaux()
    afficher_info_reseaux(r_nets, s_nets)

    # Vérifier DIM_INPUT = 52 (Point 3 : DIM_HIST 20→32)
    assert DIM_INPUT == 52, f"DIM_INPUT attendu=52, obtenu={DIM_INPUT}"
    print(f"  ✅ DIM_INPUT = {DIM_INPUT} (4+3+4+1+3+1+32+4)")

    # N°10 : vérifier dropout = 0.05
    bloc_test = BlocResiduel(256)
    assert bloc_test.dropout.p == 0.05, f"Dropout attendu=0.05, obtenu={bloc_test.dropout.p}"
    print(f"  ✅ N°10 : dropout = {bloc_test.dropout.p}")

    # N°14 : vérifier _HIST_NORM = 5.0
    assert _HIST_NORM == 5.0, f"_HIST_NORM attendu=5.0, obtenu={_HIST_NORM}"
    print(f"  ✅ N°14 : normalisation historique = {_HIST_NORM}")

    # N°16 : vérifier le bit OOP dans le vecteur encodé
    # oop_offset = DIM_PHASE + DIM_POSITION + DIM_CARD + DIM_POT + DIM_STACKS = 15
    oop_offset = DIM_PHASE + DIM_POSITION + DIM_CARD + DIM_POT + DIM_STACKS
    etat_sb = {
        'phase'        : 0,
        'buckets'      : [[3, 5, 2, 4], [6, 7, 1, 3], [2, 4, 6, 5]],
        'equites'      : [[0.55, 0.60, 0.58, 0.62],
                          [0.78, 0.72, 0.70, 0.75],
                          [0.42, 0.38, 0.40, 0.44]],
        'raise_fracs'  : [0.0, 0.0, 0.0, 0.0],
        'pot'          : 30,
        'grande_blinde': 20,
        'stacks'       : [480, 490, 470],
        'hist_phases'  : ['r', '', '', ''],
    }
    vec_sb  = encoder_infoset(etat_sb, joueur_idx=1)   # SB
    vec_btn = encoder_infoset(etat_sb, joueur_idx=0)   # BTN
    vec_bb  = encoder_infoset(etat_sb, joueur_idx=2)   # BB
    assert vec_sb[oop_offset]  == 1.0, f"SB devrait être OOP (=1.0), obtenu {vec_sb[oop_offset]}"
    assert vec_btn[oop_offset] == 0.0, f"BTN ne devrait pas être OOP, obtenu {vec_btn[oop_offset]}"
    assert vec_bb[oop_offset]  == 0.0, f"BB ne devrait pas être OOP, obtenu {vec_bb[oop_offset]}"
    print(f"  ✅ N°16 : bit OOP correct (SB=1.0, BTN=0.0, BB=0.0)")

    # Tester un batch fictif
    batch_size = 16
    x_test = torch.randn(batch_size, DIM_INPUT).to(DEVICE)

    with torch.no_grad():
        regrets   = r_nets[0](x_test)
        strategie = s_nets[0](x_test)

    print(f"\n  Batch : {batch_size} exemples × {DIM_INPUT} features")
    print(f"  Regrets   : shape={tuple(regrets.shape)}, "
          f"min={regrets.min():.3f}, max={regrets.max():.3f}")
    print(f"  Stratégie : shape={tuple(strategie.shape)}, "
          f"somme_moy={strategie.sum(-1).mean():.4f} (attendu ≈ 1.0)")

    # Test encodeur complet avec equites + raise_fracs
    print()
    print("  Test encodeur infoset (avec equites + raise_fracs)...")
    etat_test = {
        'phase'        : 1,   # FLOP
        'buckets'      : [[3, 5, 2, 4], [6, 7, 1, 3], [2, 4, 6, 5]],
        'equites'      : [[0.45, 0.62, 0.60, 0.65],
                          [0.72, 0.80, 0.78, 0.82],
                          [0.38, 0.30, 0.32, 0.35]],
        'raise_fracs'  : [1.0, 0.65, 0.0, 0.0],   # raise 65% pot au flop
        'pot'          : 200,
        'grande_blinde': 20,
        'stacks'       : [480, 490, 470],
        'hist_phases'  : ['xcr', 'r', '', ''],
    }
    vec = encoder_infoset(etat_test, joueur_idx=0)
    assert vec.shape == (DIM_INPUT,), f"Shape attendu ({DIM_INPUT},), obtenu {vec.shape}"
    assert vec.dtype == np.float32,   f"dtype attendu float32, obtenu {vec.dtype}"
    print(f"  Vecteur infoset : shape={vec.shape}, "
          f"min={vec.min():.3f}, max={vec.max():.3f}")

    # Vérifier one-hot phase (FLOP → index 1)
    assert vec[1] == 1.0 and vec[0] == 0.0, "One-hot phase FLOP incorrect"
    # Vérifier one-hot position (BTN=0 → index 4)
    assert vec[4] == 1.0 and all(vec[5:7] == 0.0), "One-hot position incorrect"
    # Vérifier equite courante à [7] (FLOP = phase 1, joueur 0 = 0.62)
    assert abs(vec[7] - 0.62) < 1e-5, f"Equite courante incorrecte : {vec[7]}"
    # Vérifier raise_frac bucket pour 0.65 (bucket 2 → bit à index 36+1=37)
    assert vec[36 + 1] == 1.0, f"Raise frac bucket 2 attendu à [37], vec[36:40]={vec[36:40]}"

    print("\n  ✅ Tous les tests réseau sont passés !")
    print("="*60 + "\n")
