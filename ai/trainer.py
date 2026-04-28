# =============================================================================
# AXIOM — ai/trainer.py
# Entraîneur PyTorch pour Deep CFR (Phase 4).
#
# Ce module est responsable de l'optimisation des réseaux de neurones à partir
# des données stockées dans les Reservoir Buffers.
#
# ─────────────────────────────────────────────────────────────────────────────
# DEUX ENTRAÎNEURS DISTINCTS
# ─────────────────────────────────────────────────────────────────────────────
#
# EntraineurRegret :
#   Entraîne le ReseauRegret à prédire les regrets instantanés.
#   Fonction de perte : MSE non pondérée
#       L = (1/B) Σ || f_θ(I) - r(I) ||²
#   Les actions illégales (indices ≥ nb_actions) sont masquées dans la perte.
#
# EntraineurStrategie :
#   Entraîne le ReseauStrategie à prédire la stratégie moyenne cumulée.
#   Fonction de perte : MSE pondérée linéairement par l'itération t
#       L = (1/B) Σ t × || g_φ(I) - σ_t(I) ||²
#   La pondération linéaire donne plus de poids aux stratégies récentes,
#   ce qui accélère la convergence vers l'équilibre de Nash.
#
# ─────────────────────────────────────────────────────────────────────────────
# DÉTAILS D'OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
#
# Optimiseur : Adam (lr depuis settings.py, weight_decay=1e-5)
#   Adam est robuste aux hyperparamètres et gère bien les gradients bruités,
#   ce qui est important car les cibles (regrets) changent à chaque itération.
#
# Scheduler : ReduceLROnPlateau (patience=5, factor=0.5)
#   Réduit automatiquement le taux d'apprentissage si la perte stagne.
#   Utile en fin d'entraînement quand les regrets convergent vers 0.
#
# Gradient clipping : max_norm=1.0
#   Évite les explosions de gradient sur les premiers batchs où les cibles
#   de regrets peuvent être très grandes.
#
# ─────────────────────────────────────────────────────────────────────────────
# BOUCLE D'ENTRAÎNEMENT PAR EPOCH
# ─────────────────────────────────────────────────────────────────────────────
#
# Pour chaque epoch :
#   1. Tirer nb_batchs mini-batchs aléatoires du Reservoir Buffer
#   2. Convertir en tenseurs PyTorch sur DEVICE
#   3. Passe avant → prédictions du réseau
#   4. Calculer la perte (avec masque actions illégales + pondération)
#   5. Rétropropagation → gradient clipping → mise à jour des poids
#   6. Accumuler les pertes pour le monitoring
#
# Référence : "Deep Counterfactual Regret Minimization"
#             N. Brown, A. Lerer, S. Gross, T. Sandholm — ICML 2019
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, List, Dict

from config.settings import LEARNING_RATE, BATCH_SIZE, NB_COUCHES, HIDDEN_SIZE, LR_DECAY_GLOBAL
from ai.network import (
    ReseauRegret, ReseauStrategie, ReseauValeur, NB_ACTIONS_MAX, DEVICE
)
from ai.reservoir import ReservoirBufferRegret, ReservoirBufferStrategie, ReservoirBufferValeur


# =============================================================================
# CONSTANTES D'ENTRAÎNEMENT
# =============================================================================

WEIGHT_DECAY   = 1e-5    # régularisation L2 dans Adam
MAX_GRAD_NORM  = 1.0     # seuil de gradient clipping
PATIENCE_LR    = 5       # scheduler : patience avant réduction du LR
FACTOR_LR      = 0.5     # scheduler : facteur de réduction du LR
NB_BATCHS_PAR_EPOCH = 300  # nombre de mini-batchs par epoch d'entraînement


# =============================================================================
# ENTRAÎNEUR REGRETS
# =============================================================================

class EntraineurRegret:
    """
    Entraîne un ReseauRegret à prédire les regrets contrefactuels instantanés.

    Perte MSE avec masque sur les actions illégales :
        L = (1/B) Σ_b Σ_a  masque[b,a] × (f_θ(I_b)[a] - r_b[a])²

    Le masque vaut 1 pour les actions légales (indice < nb_actions[b])
    et 0 pour les actions illégales (padding à 0 dans les buffers).

    Usage
    -----
        entraineur = EntraineurRegret(reseau, joueur_idx=0)
        stats = entraineur.entrainer_epoch(buffer, nb_batchs=100)
    """

    def __init__(self,
                 reseau      : ReseauRegret,
                 joueur_idx  : int,
                 lr          : float = LEARNING_RATE,
                 device      : torch.device = None):
        """
        reseau     : ReseauRegret (déjà sur le device)
        joueur_idx : index du joueur (0, 1 ou 2) — pour les logs
        lr         : taux d'apprentissage initial (Adam)
        device     : device PyTorch (DEVICE par défaut)
        """
        self.reseau     = reseau
        self.joueur_idx = joueur_idx
        self.device     = device or DEVICE

        self.optimiseur = optim.Adam(
            reseau.parameters(),
            lr           = lr,
            weight_decay = WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiseur,
            mode     = 'min',
            patience = PATIENCE_LR,
            factor   = FACTOR_LR,
        )
        self.historique_perte: List[float] = []   # perte moyenne par epoch

    # ------------------------------------------------------------------
    # EPOCH D'ENTRAÎNEMENT
    # ------------------------------------------------------------------

    def entrainer_epoch(self,
                        buffer    : ReservoirBufferRegret,
                        nb_batchs : int = NB_BATCHS_PAR_EPOCH,
                        batch_size: int = BATCH_SIZE) -> Dict[str, float]:
        """
        Effectue une epoch complète d'entraînement sur le buffer.

        Tire nb_batchs mini-batchs aléatoires et effectue une descente de
        gradient pour chacun. Retourne les statistiques de la perte.

        Paramètres
        ----------
        buffer     : ReservoirBufferRegret rempli avec les données de traversée
        nb_batchs  : nombre de mini-batchs par epoch
        batch_size : taille de chaque mini-batch

        Retourne
        --------
        dict avec clés :
            'perte_moy'   : float — perte MSE moyenne sur l'epoch
            'perte_min'   : float — perte minimale observée
            'perte_max'   : float — perte maximale observée
            'lr_courant'  : float — taux d'apprentissage courant
        """
        if not buffer.est_pret(batch_size):
            return {'perte_moy': float('nan'), 'perte_min': float('nan'),
                    'perte_max': float('nan'), 'lr_courant': self._lr_courant()}

        self.reseau.train()
        pertes = []

        for _ in range(nb_batchs):
            batch = buffer.echantillonner(batch_size)
            if batch is None:
                continue

            vecs, regrets_cibles, nb_actions = batch
            perte = self._pas_gradient(vecs, regrets_cibles, nb_actions)
            if perte is not None:
                pertes.append(perte)

        if not pertes:
            return {'perte_moy': float('nan'), 'perte_min': float('nan'),
                    'perte_max': float('nan'), 'lr_courant': self._lr_courant()}

        perte_moy = float(np.mean(pertes))
        self.scheduler.step(perte_moy)
        self.historique_perte.append(perte_moy)

        return {
            'perte_moy'  : perte_moy,
            'perte_min'  : float(np.min(pertes)),
            'perte_max'  : float(np.max(pertes)),
            'lr_courant' : self._lr_courant(),
        }

    # ------------------------------------------------------------------
    # PAS DE GRADIENT
    # ------------------------------------------------------------------

    def _pas_gradient(self,
                      vecs           : np.ndarray,
                      regrets_cibles : np.ndarray,
                      nb_actions     : np.ndarray) -> Optional[float]:
        """
        Effectue un pas de gradient Adam sur un mini-batch.

        Applique un masque sur les actions illégales pour éviter de pénaliser
        les prédictions sur des actions qui n'existent pas dans ce nœud.

        Retourne la valeur de la perte (float) ou None si erreur.
        """
        # Convertir en tenseurs PyTorch
        x        = torch.from_numpy(vecs).to(self.device)
        cibles   = torch.from_numpy(regrets_cibles).to(self.device)
        nb_act_t = torch.from_numpy(nb_actions.astype(np.int64)).to(self.device)

        # Construire le masque des actions légales
        # masque[b, a] = 1 si a < nb_actions[b], 0 sinon
        indices = torch.arange(NB_ACTIONS_MAX, device=self.device).unsqueeze(0)
        masque  = (indices < nb_act_t.unsqueeze(1)).float()   # (B, NB_ACTIONS_MAX)

        # Passe avant
        predictions = self.reseau(x)   # (B, NB_ACTIONS_MAX)

        # MSE masquée : ne pénaliser que les actions légales
        erreurs_carrees = (predictions - cibles) ** 2   # (B, NB_ACTIONS_MAX)
        perte = (erreurs_carrees * masque).sum() / (masque.sum() + 1e-8)

        # Rétropropagation
        self.optimiseur.zero_grad()
        perte.backward()
        nn.utils.clip_grad_norm_(self.reseau.parameters(), MAX_GRAD_NORM)
        self.optimiseur.step()

        return float(perte.detach().cpu())

    # ------------------------------------------------------------------
    # UTILITAIRES
    # ------------------------------------------------------------------

    def _lr_courant(self) -> float:
        """Retourne le taux d'apprentissage courant de l'optimiseur Adam."""
        return self.optimiseur.param_groups[0]['lr']

    def reinitialiser_scheduler(self,
                               iteration_courante: int = 1,
                               nb_iterations_total: int = 500) -> None:
        """
        Réinitialise le LR et le scheduler entre itérations Deep CFR.

        Point 10 — LR warm start (Pluribus) :
          Si LR_DECAY_GLOBAL=True, applique un schedule décroissant global :
            LR_eff(t) = LEARNING_RATE / sqrt(t)
          avec plancher à 1% de LEARNING_RATE pour éviter la paralysie.

          Les itérations tardives affinent des détails fins — un LR élevé
          en fin d'entraînement peut défaire ce que les itérations précédentes
          ont appris.
        """
        if LR_DECAY_GLOBAL and iteration_courante > 1:
            lr_plancher = LEARNING_RATE * 0.01
            lr_effectif = LEARNING_RATE / math.sqrt(float(iteration_courante))
            lr_effectif = max(lr_effectif, lr_plancher)
        else:
            lr_effectif = LEARNING_RATE

        for pg in self.optimiseur.param_groups:
            pg['lr'] = lr_effectif
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiseur, mode='min',
            patience=PATIENCE_LR, factor=FACTOR_LR,
        )

    def __repr__(self) -> str:
        return (f"EntraineurRegret(joueur={self.joueur_idx}, "
                f"lr={self._lr_courant():.2e}, "
                f"epochs={len(self.historique_perte)})")


# =============================================================================
# ENTRAÎNEUR STRATÉGIE
# =============================================================================

class EntraineurStrategie:
    """
    Entraîne un ReseauStrategie à prédire la stratégie moyenne cumulée.

    Perte MSE pondérée linéairement par l'itération t :
        L = (1/B) Σ_b  t_b × Σ_a  masque[b,a] × (g_φ(I_b)[a] - σ_b[a])²

    La pondération par t_b donne plus de poids aux stratégies des itérations
    récentes, qui sont plus proches de l'équilibre de Nash final.

    Usage
    -----
        entraineur = EntraineurStrategie(reseau, joueur_idx=0)
        stats = entraineur.entrainer_epoch(buffer, nb_batchs=100)
    """

    def __init__(self,
                 reseau      : ReseauStrategie,
                 joueur_idx  : int,
                 lr          : float = LEARNING_RATE,
                 device      : torch.device = None):
        self.reseau     = reseau
        self.joueur_idx = joueur_idx
        self.device     = device or DEVICE

        self.optimiseur = optim.Adam(
            reseau.parameters(),
            lr           = lr,
            weight_decay = WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiseur, mode='min',
            patience=PATIENCE_LR, factor=FACTOR_LR,
        )
        self.historique_perte: List[float] = []

    # ------------------------------------------------------------------
    # EPOCH D'ENTRAÎNEMENT
    # ------------------------------------------------------------------

    def entrainer_epoch(self,
                        buffer    : ReservoirBufferStrategie,
                        nb_batchs : int = NB_BATCHS_PAR_EPOCH,
                        batch_size: int = BATCH_SIZE) -> Dict[str, float]:
        """
        Effectue une epoch complète d'entraînement sur le buffer stratégie.

        Paramètres
        ----------
        buffer     : ReservoirBufferStrategie rempli avec les données de traversée
        nb_batchs  : nombre de mini-batchs par epoch
        batch_size : taille de chaque mini-batch

        Retourne
        --------
        dict avec clés :
            'perte_moy'   : perte MSE pondérée moyenne
            'perte_min'   : perte minimale
            'perte_max'   : perte maximale
            'lr_courant'  : taux d'apprentissage courant
        """
        if not buffer.est_pret(batch_size):
            return {'perte_moy': float('nan'), 'perte_min': float('nan'),
                    'perte_max': float('nan'), 'lr_courant': self._lr_courant()}

        self.reseau.train()
        pertes = []

        for _ in range(nb_batchs):
            batch = buffer.echantillonner(batch_size)
            if batch is None:
                continue

            vecs, strategies_cibles, iterations, nb_actions = batch
            perte = self._pas_gradient(vecs, strategies_cibles, iterations, nb_actions)
            if perte is not None:
                pertes.append(perte)

        if not pertes:
            return {'perte_moy': float('nan'), 'perte_min': float('nan'),
                    'perte_max': float('nan'), 'lr_courant': self._lr_courant()}

        perte_moy = float(np.mean(pertes))
        self.scheduler.step(perte_moy)
        self.historique_perte.append(perte_moy)

        return {
            'perte_moy'  : perte_moy,
            'perte_min'  : float(np.min(pertes)),
            'perte_max'  : float(np.max(pertes)),
            'lr_courant' : self._lr_courant(),
        }

    # ------------------------------------------------------------------
    # PAS DE GRADIENT
    # ------------------------------------------------------------------

    def _pas_gradient(self,
                      vecs                : np.ndarray,
                      strategies_cibles   : np.ndarray,
                      iterations          : np.ndarray,
                      nb_actions          : np.ndarray) -> Optional[float]:
        """
        Effectue un pas de gradient Adam avec pondération linéaire par itération
        et masquage des actions illégales.

        La pondération renforce l'influence des stratégies récentes
        (plus proches de l'équilibre de Nash).
        Normalisation par max(poids) pour préserver la pondération linéaire
        relative entre itérations sans distorsion par la moyenne du batch.
        """
        x        = torch.from_numpy(vecs).to(self.device)
        cibles   = torch.from_numpy(strategies_cibles).to(self.device)
        poids    = torch.from_numpy(iterations).to(self.device)   # (B,)
        nb_act_t = torch.from_numpy(nb_actions.astype(np.int64)).to(self.device)

        # Normaliser par le max pour préserver la pondération linéaire
        # (itération T → poids 1.0, itération 1 → poids 1/T)
        poids_norm = poids / (poids.max() + 1e-8)

        # Masque des actions légales : masque[b, a] = 1 si a < nb_actions[b]
        indices = torch.arange(NB_ACTIONS_MAX, device=self.device).unsqueeze(0)
        masque  = (indices < nb_act_t.unsqueeze(1)).float()   # (B, NB_ACTIONS_MAX)

        # Passe avant
        predictions = self.reseau(x)   # (B, NB_ACTIONS_MAX)

        # MSE masquée et pondérée par itération
        erreurs_carrees = (predictions - cibles) ** 2              # (B, NB_ACTIONS_MAX)
        perte_par_ex    = (erreurs_carrees * masque).sum(dim=-1) \
                          / (masque.sum(dim=-1) + 1e-8)            # (B,)
        perte           = (poids_norm * perte_par_ex).mean()

        # Rétropropagation
        self.optimiseur.zero_grad()
        perte.backward()
        nn.utils.clip_grad_norm_(self.reseau.parameters(), MAX_GRAD_NORM)
        self.optimiseur.step()

        return float(perte.detach().cpu())

    # ------------------------------------------------------------------
    # UTILITAIRES
    # ------------------------------------------------------------------

    def _lr_courant(self) -> float:
        return self.optimiseur.param_groups[0]['lr']

    def reinitialiser_scheduler(self,
                               iteration_courante: int = 1,
                               nb_iterations_total: int = 500) -> None:
        """Point 10 — même schedule décroissant que EntraineurRegret."""
        if LR_DECAY_GLOBAL and iteration_courante > 1:
            lr_plancher = LEARNING_RATE * 0.01
            lr_effectif = LEARNING_RATE / math.sqrt(float(iteration_courante))
            lr_effectif = max(lr_effectif, lr_plancher)
        else:
            lr_effectif = LEARNING_RATE

        for pg in self.optimiseur.param_groups:
            pg['lr'] = lr_effectif
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiseur, mode='min',
            patience=PATIENCE_LR, factor=FACTOR_LR,
        )

    def __repr__(self) -> str:
        return (f"EntraineurStrategie(joueur={self.joueur_idx}, "
                f"lr={self._lr_courant():.2e}, "
                f"epochs={len(self.historique_perte)})")


# =============================================================================
# ENTRAÎNEUR VALEUR — Point 6
# =============================================================================

class EntraineurValeur:
    """
    Entraîne un ReseauValeur à prédire la valeur espérée d'un infoset.

    Perte MSE scalaire :
        L = (1/B) Σ_b (V_θ(I_b) - v_b)²

    Où v_b est la valeur du nœud calculée pendant la traversée Deep CFR
    (v_b = Σ_a strategie[a] × valeur[a]).

    Utilisé comme oracle feuille dans le solveur depth-limited (Point 2).
    """

    def __init__(self,
                 reseau     : ReseauValeur,
                 joueur_idx : int,
                 lr         : float = LEARNING_RATE,
                 device     : torch.device = None):
        self.reseau     = reseau
        self.joueur_idx = joueur_idx
        self.device     = device or DEVICE

        self.optimiseur = optim.Adam(
            reseau.parameters(),
            lr           = lr,
            weight_decay = WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiseur, mode='min',
            patience=PATIENCE_LR, factor=FACTOR_LR,
        )
        self.historique_perte: List[float] = []

    def entrainer_epoch(self,
                        buffer    : ReservoirBufferValeur,
                        nb_batchs : int = NB_BATCHS_PAR_EPOCH,
                        batch_size: int = BATCH_SIZE) -> Dict[str, float]:
        """Effectue une epoch d'entraînement sur le buffer valeur."""
        if not buffer.est_pret(batch_size):
            return {'perte_moy': float('nan'), 'perte_min': float('nan'),
                    'perte_max': float('nan'), 'lr_courant': self._lr_courant()}

        self.reseau.train()
        pertes = []

        for _ in range(nb_batchs):
            batch = buffer.echantillonner(batch_size)
            if batch is None:
                continue
            vecs, valeurs_cibles = batch
            perte = self._pas_gradient(vecs, valeurs_cibles)
            if perte is not None:
                pertes.append(perte)

        if not pertes:
            return {'perte_moy': float('nan'), 'perte_min': float('nan'),
                    'perte_max': float('nan'), 'lr_courant': self._lr_courant()}

        perte_moy = float(np.mean(pertes))
        self.scheduler.step(perte_moy)
        self.historique_perte.append(perte_moy)
        return {
            'perte_moy'  : perte_moy,
            'perte_min'  : float(np.min(pertes)),
            'perte_max'  : float(np.max(pertes)),
            'lr_courant' : self._lr_courant(),
        }

    def _pas_gradient(self,
                      vecs           : np.ndarray,
                      valeurs_cibles : np.ndarray) -> Optional[float]:
        x       = torch.from_numpy(vecs).to(self.device)
        cibles  = torch.from_numpy(valeurs_cibles).to(self.device).unsqueeze(1)  # (B,1)

        predictions = self.reseau(x)   # (B, 1)
        perte = ((predictions - cibles) ** 2).mean()

        self.optimiseur.zero_grad()
        perte.backward()
        nn.utils.clip_grad_norm_(self.reseau.parameters(), MAX_GRAD_NORM)
        self.optimiseur.step()
        return float(perte.detach().cpu())

    def _lr_courant(self) -> float:
        return self.optimiseur.param_groups[0]['lr']

    def reinitialiser_scheduler(self, iteration_courante: int = 1,
                                nb_iterations_total: int = 500) -> None:
        if LR_DECAY_GLOBAL and iteration_courante > 1:
            lr_plancher = LEARNING_RATE * 0.01
            lr_effectif = max(LEARNING_RATE / math.sqrt(float(iteration_courante)), lr_plancher)
        else:
            lr_effectif = LEARNING_RATE
        for pg in self.optimiseur.param_groups:
            pg['lr'] = lr_effectif
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiseur, mode='min',
            patience=PATIENCE_LR, factor=FACTOR_LR,
        )

    def __repr__(self) -> str:
        return (f"EntraineurValeur(joueur={self.joueur_idx}, "
                f"lr={self._lr_courant():.2e}, "
                f"epochs={len(self.historique_perte)})")


# =============================================================================
# UTILITAIRE — CRÉER TOUS LES ENTRAÎNEURS
# =============================================================================

def creer_entraineurs(
        reseaux_regret    : List[ReseauRegret],
        reseaux_strategie : List[ReseauStrategie],
        lr                : float = LEARNING_RATE,
        device            : torch.device = None,
) -> tuple:
    """
    Crée les 6 entraîneurs (2 par joueur × 3 joueurs).

    Retourne
    --------
    tuple : (
        [EntraineurRegret_J0,    ..._J1,    ..._J2],
        [EntraineurStrategie_J0, ..._J1, ..._J2]
    )
    """
    entraineurs_regret = [
        EntraineurRegret(reseaux_regret[i], joueur_idx=i, lr=lr, device=device)
        for i in range(3)
    ]
    entraineurs_strategie = [
        EntraineurStrategie(reseaux_strategie[i], joueur_idx=i, lr=lr, device=device)
        for i in range(3)
    ]
    return entraineurs_regret, entraineurs_strategie


def afficher_stats_entrainement(
        stats_regret    : List[Dict],
        stats_strategie : List[Dict],
        iteration       : int) -> None:
    """
    Affiche un tableau récapitulatif des pertes après une itération Deep CFR.

    stats_regret    : liste de 3 dicts (un par joueur)
    stats_strategie : liste de 3 dicts (un par joueur)
    iteration       : numéro de l'itération Deep CFR courante
    """
    print(f"\n  Itération Deep CFR #{iteration} — Pertes d'entraînement")
    print(f"  {'─'*58}")
    print(f"  {'Joueur':>6} | {'Regrets (MSE)':>14} | {'Stratégie (MSE)':>15} | {'LR':>8}")
    print(f"  {'─'*6}-+-{'─'*14}-+-{'─'*15}-+-{'─'*8}")
    for i in range(3):
        sr = stats_regret[i]
        ss = stats_strategie[i]
        lr = sr.get('lr_courant', float('nan'))
        pr = sr.get('perte_moy', float('nan'))
        ps = ss.get('perte_moy', float('nan'))
        print(f"  {i:>6} | {pr:>14.6f} | {ps:>15.6f} | {lr:>8.2e}")
    print(f"  {'─'*58}\n")


# =============================================================================
# TEST RAPIDE (si exécuté directement)
# =============================================================================

if __name__ == '__main__':
    import time
    from ai.network import creer_reseaux, DIM_INPUT
    from ai.reservoir import creer_buffers

    print("\n" + "="*60)
    print("  AXIOM — Test Entraîneur (trainer.py)")
    print("="*60)

    # Créer réseaux, buffers, entraîneurs
    r_nets, s_nets = creer_reseaux()
    r_bufs, s_bufs = creer_buffers(taille_max=50_000)
    e_regs, e_strats = creer_entraineurs(r_nets, s_nets)

    print(f"\n  Réseaux et entraîneurs créés pour 3 joueurs.")
    print(f"  {e_regs[0]}")
    print(f"  {e_strats[0]}")

    # Remplir les buffers avec des données synthétiques
    print("\n  Remplissage des buffers (données synthétiques)...")
    N = 10_000
    for i in range(N):
        vec      = np.random.randn(DIM_INPUT).astype(np.float32)
        regrets  = np.random.randn(NB_ACTIONS_MAX).astype(np.float32)
        strat    = np.random.dirichlet(np.ones(NB_ACTIONS_MAX)).astype(np.float32)
        for j in range(3):
            r_bufs[j].ajouter(vec, regrets, NB_ACTIONS_MAX)
            s_bufs[j].ajouter(vec, strat, iteration=i+1)

    print(f"  Buffer regrets J0 : {r_bufs[0]}")
    print(f"  Buffer stratégie J0 : {s_bufs[0]}")

    # Test d'une epoch pour chaque joueur
    print("\n  Test d'une epoch d'entraînement...")
    t0 = time.time()
    for j in range(3):
        stats_r = e_regs[j].entrainer_epoch(r_bufs[j], nb_batchs=10, batch_size=256)
        stats_s = e_strats[j].entrainer_epoch(s_bufs[j], nb_batchs=10, batch_size=256)
        print(f"  Joueur {j} | Regrets loss = {stats_r['perte_moy']:.6f} | "
              f"Stratégie loss = {stats_s['perte_moy']:.6f}")
        assert not np.isnan(stats_r['perte_moy']), "Perte regrets NaN !"
        assert not np.isnan(stats_s['perte_moy']), "Perte stratégie NaN !"
    print(f"  Temps : {time.time() - t0:.2f}s")

    # Vérifier que les pertes sont des floats valides
    stats_list_r = [e_regs[j].entrainer_epoch(r_bufs[j], nb_batchs=5) for j in range(3)]
    stats_list_s = [e_strats[j].entrainer_epoch(s_bufs[j], nb_batchs=5) for j in range(3)]
    afficher_stats_entrainement(stats_list_r, stats_list_s, iteration=1)

    # Vérifier que les poids du réseau ont bien changé après entraînement
    params_avant = [p.clone() for p in r_nets[0].parameters()]
    e_regs[0].entrainer_epoch(r_bufs[0], nb_batchs=5, batch_size=256)
    params_apres = list(r_nets[0].parameters())
    changements = sum(
        not torch.equal(a, b) for a, b in zip(params_avant, params_apres)
    )
    assert changements > 0, "Les poids du réseau n'ont pas changé après entraînement !"
    print(f"  ✅ Poids mis à jour : {changements} tenseurs modifiés sur "
          f"{len(params_avant)} couches")

    print("\n  ✅ Tous les tests entraîneur sont passés !")
    print("="*60 + "\n")
