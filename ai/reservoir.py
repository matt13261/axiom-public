# =============================================================================
# AXIOM — ai/reservoir.py
# Reservoir Buffer pour Deep CFR (Phase 4).
#
# ─────────────────────────────────────────────────────────────────────────────
# POURQUOI UN RESERVOIR BUFFER ?
# ─────────────────────────────────────────────────────────────────────────────
#
# Deep CFR ne peut pas entraîner le réseau directement sur les données générées
# à chaque traversée : l'ordre d'arrivée des échantillons introduirait un biais
# catastrophique (le réseau oublierait les premières itérations).
#
# Solution : stocker tous les échantillons dans un buffer, puis entraîner le
# réseau sur des mini-batchs tirés ALÉATOIREMENT du buffer.
#
# Problème : la mémoire disponible est limitée. Si on stocke tout, on explose.
#
# Reservoir Sampling (Vitter, 1985) :
#   → Maintenir un buffer de taille N fixe.
#   → Chaque nouvel échantillon (le k-ième) est inséré avec probabilité N/k.
#   → Si inséré, il remplace un échantillon existant tiré aléatoirement.
#   → Propriété : à tout moment, le buffer contient un échantillon UNIFORME
#                 de tous les données vues jusqu'ici, sans biais temporel.
#
# C'est exactement ce que préconise le papier Deep CFR (Brown et al., 2019).
#
# ─────────────────────────────────────────────────────────────────────────────
# CONTENU DES ÉCHANTILLONS
# ─────────────────────────────────────────────────────────────────────────────
#
# Chaque échantillon du ReservoirBufferRegret contient :
#   infoset_vec : np.ndarray (DIM_INPUT,)  — vecteur de features de l'infoset
#   regrets     : np.ndarray (NB_ACTIONS,) — regrets instantanés de la traversée
#   nb_actions  : int                      — nombre d'actions légales au nœud
#
# Chaque échantillon du ReservoirBufferStrategie contient :
#   infoset_vec : np.ndarray (DIM_INPUT,)  — vecteur de features de l'infoset
#   strategie   : np.ndarray (NB_ACTIONS,) — stratégie courante (somme = 1)
#   iteration   : int                      — numéro d'itération Deep CFR
#                                            (pour la pondération linéaire)
#
# ─────────────────────────────────────────────────────────────────────────────
# PONDÉRATION LINÉAIRE DES STRATÉGIES
# ─────────────────────────────────────────────────────────────────────────────
#
# Dans Deep CFR, les échantillons de stratégie plus récents doivent avoir un
# poids plus grand (car la stratégie s'améliore au fil des itérations).
# Le poids de l'itération t est proportionnel à t (pondération linéaire).
#
# Cela est implémenté via la pondération dans la fonction de perte MSE :
#   loss = Σ_t  t × || g_φ(I) - σ_t(I) ||²
#
# Référence : Brown et al. 2019, Section 3.3
#
# ─────────────────────────────────────────────────────────────────────────────
# TAILLE DES BUFFERS
# ─────────────────────────────────────────────────────────────────────────────
#
# La taille est configurable via TAILLE_BUFFER (défaut : 2_000_000).
# En pratique, Deep CFR avec 1000 traversées × ~50 nœuds/traversée
# génère ~50 000 échantillons par itération.
# Avec 2M de slots, le buffer couvre ~40 itérations complètes en mémoire.
#
# Mémoire approximative :
#   1 échantillon regret   ≈ 32 + 10 = 42 floats × 4 octets = 168 octets
#   2 000 000 échantillons ≈ 336 Mo (raisonnable sur toute machine moderne)
# =============================================================================
 
import random
import numpy as np
from typing import Optional, Tuple, List
 
from ai.network import DIM_INPUT, NB_ACTIONS_MAX
 
 
# =============================================================================
# TAILLE PAR DÉFAUT DU BUFFER
# =============================================================================
 
TAILLE_BUFFER_DEFAUT = 3_000_000   # slots dans le reservoir
 
 
# =============================================================================
# RESERVOIR BUFFER — REGRETS
# =============================================================================
 
class ReservoirBufferRegret:
    """
    Reservoir buffer pour les échantillons de regrets de Deep CFR.
 
    Maintient un échantillon uniforme de tous les triplets
    (infoset_vec, regrets, nb_actions) générés pendant les traversées.
 
    Attributs
    ---------
    taille_max   : capacité maximale du buffer (slots)
    nb_total     : nombre total d'échantillons ajoutés depuis la création
                   (peut dépasser taille_max — sert au calcul de probabilité)
    _vecs        : np.ndarray (taille_max, DIM_INPUT)  — vecteurs infoset
    _regrets     : np.ndarray (taille_max, NB_ACTIONS_MAX) — regrets
    _nb_actions  : np.ndarray (taille_max,) int32 — nb d'actions légales
    _rng         : random.Random — générateur local (thread-safe)
    """
 
    def __init__(self, taille_max: int = TAILLE_BUFFER_DEFAUT):
        """
        taille_max : nombre maximum d'échantillons stockés simultanément.
        """
        self.taille_max  : int = taille_max
        self.nb_total    : int = 0          # total ajoutés (pour proba reservoir)
        self._nb_rempli  : int = 0          # slots effectivement remplis (≤ taille_max)
        self._rng        = random.Random()
 
        # Pré-allouer les arrays numpy (évite les réallocations coûteuses)
        self._vecs      = np.zeros((taille_max, DIM_INPUT),       dtype=np.float32)
        self._regrets   = np.zeros((taille_max, NB_ACTIONS_MAX),  dtype=np.float32)
        self._nb_actions = np.zeros(taille_max,                   dtype=np.int32)
 
    # ------------------------------------------------------------------
    # AJOUT D'UN ÉCHANTILLON (Reservoir Sampling)
    # ------------------------------------------------------------------
 
    def ajouter(self,
                infoset_vec : np.ndarray,
                regrets     : np.ndarray,
                nb_actions  : int) -> None:
        """
        Ajoute un échantillon au buffer via l'algorithme de Reservoir Sampling.
 
        Si le buffer n'est pas plein : ajout direct.
        Sinon : insertion avec probabilité taille_max / (nb_total + 1),
                en remplaçant un slot tiré uniformément au hasard.
 
        Cette procédure garantit que le buffer contient en permanence
        un échantillon uniforme de TOUS les échantillons vus.
 
        Paramètres
        ----------
        infoset_vec : np.ndarray (DIM_INPUT,)  — vecteur de features
        regrets     : np.ndarray (nb_actions,) — regrets instantanés
        nb_actions  : int — nombre d'actions légales au nœud (≤ NB_ACTIONS_MAX)
        """
        self.nb_total += 1
        k = self.nb_total
 
        if self._nb_rempli < self.taille_max:
            # Buffer pas encore plein : insertion directe
            idx = self._nb_rempli
            self._nb_rempli += 1
        else:
            # Reservoir sampling : insérer avec proba taille_max / k
            j = self._rng.randint(1, k)   # j uniformément dans [1, k]
            if j > self.taille_max:
                return   # pas d'insertion ce coup-ci
            idx = j - 1   # index à remplacer (0-based)
 
        # Écriture dans les arrays pré-alloués
        self._vecs[idx]      = infoset_vec[:DIM_INPUT]    # tronquer si besoin
        self._regrets[idx]   = 0.0                         # reset d'abord
        n = min(len(regrets), NB_ACTIONS_MAX)
        self._regrets[idx, :n] = regrets[:n]
        self._nb_actions[idx]  = min(nb_actions, NB_ACTIONS_MAX)
 
    # ------------------------------------------------------------------
    # TIRAGE D'UN MINI-BATCH
    # ------------------------------------------------------------------
 
    def echantillonner(
            self, taille_batch: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Tire un mini-batch aléatoire depuis le buffer (sans remise).
 
        Si le buffer contient moins d'échantillons que taille_batch,
        retourne None (pas encore assez de données pour entraîner).
 
        Retourne
        --------
        (vecs, regrets, nb_actions) ou None
          vecs       : np.ndarray (taille_batch, DIM_INPUT)
          regrets    : np.ndarray (taille_batch, NB_ACTIONS_MAX)
          nb_actions : np.ndarray (taille_batch,) int32
        """
        n = self._nb_rempli
        if n < taille_batch:
            return None
 
        indices = np.random.choice(n, size=taille_batch, replace=False)
        return (
            self._vecs[indices],
            self._regrets[indices],
            self._nb_actions[indices],
        )
 
    # ------------------------------------------------------------------
    # PROPRIÉTÉS ET AFFICHAGE
    # ------------------------------------------------------------------
 
    def __len__(self) -> int:
        """Nombre d'échantillons effectivement stockés dans le buffer."""
        return self._nb_rempli
 
    def est_pret(self, taille_batch: int) -> bool:
        """Retourne True si le buffer contient assez d'échantillons."""
        return self._nb_rempli >= taille_batch
 
    def taux_remplissage(self) -> float:
        """Fraction du buffer utilisée (0.0 → 1.0)."""
        return self._nb_rempli / self.taille_max
 
    def reinitialiser(self) -> None:
        """Vide le buffer (remet tous les compteurs à zéro)."""
        self.nb_total   = 0
        self._nb_rempli = 0
        # Les arrays ne sont pas réinitialisés — ils seront écrasés
 
    def sauvegarder(self, chemin: str) -> None:
        """Sauvegarde le buffer sur disque (numpy .npz) — écriture atomique.
        Écrit dans un fichier temporaire puis renomme, pour éviter toute
        corruption si le processus est interrompu pendant la sauvegarde.
        Note : np.savez_compressed ajoute automatiquement .npz au nom passé,
        donc on passe chemin_tmp_base (sans .npz) et on récupère chemin_tmp_base.npz."""
        import os
        chemin_npz    = chemin if chemin.endswith('.npz') else chemin + '.npz'
        chemin_tmp_base = chemin_npz[:-4] + '.tmp'   # ex: buffer_regret_j0.tmp
        chemin_tmp    = chemin_tmp_base + '.npz'      # numpy va créer ce fichier
        os.makedirs(os.path.dirname(chemin_npz) if os.path.dirname(chemin_npz) else '.', exist_ok=True)
        np.savez_compressed(
            chemin_tmp_base,                           # numpy ajoute .npz → chemin_tmp
            vecs       = self._vecs[:self._nb_rempli],
            regrets    = self._regrets[:self._nb_rempli],
            nb_actions = self._nb_actions[:self._nb_rempli],
            nb_total   = np.array([self.nb_total]),
        )
        # Remplacement atomique : le .npz final n'est jamais partiellement écrit
        os.replace(chemin_tmp, chemin_npz)
 
    def charger(self, chemin: str) -> bool:
        """
        Charge le buffer depuis disque. Retourne True si succès, False si fichier absent.
        """
        chemin_npz = chemin if chemin.endswith('.npz') else chemin + '.npz'
        if not __import__('os').path.exists(chemin_npz):
            return False
        data           = np.load(chemin_npz)
        n              = len(data['vecs'])
        n              = min(n, self.taille_max)
        self._vecs[:n]       = data['vecs'][:n]
        self._regrets[:n]    = data['regrets'][:n]
        self._nb_actions[:n] = data['nb_actions'][:n]
        self._nb_rempli      = n
        self.nb_total        = int(data['nb_total'][0])
        return True
 
    def __repr__(self) -> str:
        return (f"ReservoirBufferRegret("
                f"{self._nb_rempli:,}/{self.taille_max:,} slots, "
                f"{self.nb_total:,} ajoutés au total)")
 
 
# =============================================================================
# RESERVOIR BUFFER — STRATÉGIE
# =============================================================================
 
class ReservoirBufferStrategie:
    """
    Reservoir buffer pour les échantillons de stratégie de Deep CFR.
 
    Maintient un échantillon uniforme de triplets
    (infoset_vec, strategie, iteration).
 
    La pondération linéaire par `iteration` est utilisée lors du calcul de
    la perte MSE dans l'entraînement du ReseauStrategie :
        loss = Σ iteration × || g_φ(I) - σ(I) ||²
 
    Attributs
    ---------
    taille_max   : capacité maximale
    nb_total     : total d'échantillons ajoutés
    _vecs        : vecteurs infoset (taille_max, DIM_INPUT)
    _strategies  : distributions (taille_max, NB_ACTIONS_MAX)
    _iterations  : numéros d'itération float32 (taille_max,)
    """
 
    def __init__(self, taille_max: int = TAILLE_BUFFER_DEFAUT):
        self.taille_max  : int = taille_max
        self.nb_total    : int = 0
        self._nb_rempli  : int = 0
        self._rng        = random.Random()
 
        self._vecs       = np.zeros((taille_max, DIM_INPUT),      dtype=np.float32)
        self._strategies = np.zeros((taille_max, NB_ACTIONS_MAX), dtype=np.float32)
        self._iterations = np.zeros(taille_max,                   dtype=np.float32)
        self._nb_actions = np.zeros(taille_max,                   dtype=np.int32)

    # ------------------------------------------------------------------
    # AJOUT D'UN ÉCHANTILLON
    # ------------------------------------------------------------------

    def ajouter(self,
                infoset_vec : np.ndarray,
                strategie   : np.ndarray,
                iteration   : int,
                nb_actions  : int = NB_ACTIONS_MAX) -> None:
        """
        Ajoute un échantillon de stratégie via Reservoir Sampling.

        infoset_vec : np.ndarray (DIM_INPUT,)  — vecteur de features
        strategie   : np.ndarray (nb_actions,) — stratégie courante (somme ≈ 1)
        iteration   : int — numéro d'itération Deep CFR (pour pondération)
        nb_actions  : int — nombre d'actions légales au nœud (pour masquage)
        """
        self.nb_total += 1
        k = self.nb_total

        if self._nb_rempli < self.taille_max:
            idx = self._nb_rempli
            self._nb_rempli += 1
        else:
            j = self._rng.randint(1, k)
            if j > self.taille_max:
                return
            idx = j - 1

        self._vecs[idx]       = infoset_vec[:DIM_INPUT]
        self._strategies[idx] = 0.0
        n = min(len(strategie), NB_ACTIONS_MAX)
        self._strategies[idx, :n] = strategie[:n]
        self._iterations[idx] = float(iteration)
        self._nb_actions[idx] = min(nb_actions, NB_ACTIONS_MAX)

    # ------------------------------------------------------------------
    # TIRAGE D'UN MINI-BATCH
    # ------------------------------------------------------------------

    def echantillonner(
            self, taille_batch: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Tire un mini-batch aléatoire depuis le buffer.

        Retourne
        --------
        (vecs, strategies, iterations, nb_actions) ou None
          vecs        : np.ndarray (taille_batch, DIM_INPUT)
          strategies  : np.ndarray (taille_batch, NB_ACTIONS_MAX)
          iterations  : np.ndarray (taille_batch,) float32 — poids de pondération
          nb_actions  : np.ndarray (taille_batch,) int32   — actions légales
        """
        n = self._nb_rempli
        if n < taille_batch:
            return None

        indices = np.random.choice(n, size=taille_batch, replace=False)
        return (
            self._vecs[indices],
            self._strategies[indices],
            self._iterations[indices],
            self._nb_actions[indices],
        )
 
    # ------------------------------------------------------------------
    # PROPRIÉTÉS ET AFFICHAGE
    # ------------------------------------------------------------------
 
    def __len__(self) -> int:
        return self._nb_rempli
 
    def est_pret(self, taille_batch: int) -> bool:
        return self._nb_rempli >= taille_batch
 
    def taux_remplissage(self) -> float:
        return self._nb_rempli / self.taille_max
 
    def reinitialiser(self) -> None:
        self.nb_total   = 0
        self._nb_rempli = 0
 
    def sauvegarder(self, chemin: str) -> None:
        """Sauvegarde le buffer sur disque (numpy .npz) — écriture atomique.
        Écrit dans un fichier temporaire puis renomme, pour éviter toute
        corruption si le processus est interrompu pendant la sauvegarde.
        Note : np.savez_compressed ajoute automatiquement .npz au nom passé,
        donc on passe chemin_tmp_base (sans .npz) et on récupère chemin_tmp_base.npz."""
        import os
        chemin_npz      = chemin if chemin.endswith('.npz') else chemin + '.npz'
        chemin_tmp_base = chemin_npz[:-4] + '.tmp'   # ex: buffer_strategie_j0.tmp
        chemin_tmp      = chemin_tmp_base + '.npz'    # numpy va créer ce fichier
        os.makedirs(os.path.dirname(chemin_npz) if os.path.dirname(chemin_npz) else '.', exist_ok=True)
        np.savez_compressed(
            chemin_tmp_base,                           # numpy ajoute .npz → chemin_tmp
            vecs       = self._vecs[:self._nb_rempli],
            strategies = self._strategies[:self._nb_rempli],
            iterations = self._iterations[:self._nb_rempli],
            nb_actions = self._nb_actions[:self._nb_rempli],
            nb_total   = np.array([self.nb_total]),
        )
        # Remplacement atomique : le .npz final n'est jamais partiellement écrit
        os.replace(chemin_tmp, chemin_npz)

    def charger(self, chemin: str) -> bool:
        """
        Charge le buffer depuis disque. Retourne True si succès, False si fichier absent.
        """
        chemin_npz = chemin if chemin.endswith('.npz') else chemin + '.npz'
        if not __import__('os').path.exists(chemin_npz):
            return False
        data             = np.load(chemin_npz)
        n                = len(data['vecs'])
        n                = min(n, self.taille_max)
        self._vecs[:n]       = data['vecs'][:n]
        self._strategies[:n] = data['strategies'][:n]
        self._iterations[:n] = data['iterations'][:n]
        # Rétrocompatibilité : anciens fichiers sans nb_actions → NB_ACTIONS_MAX
        if 'nb_actions' in data:
            self._nb_actions[:n] = data['nb_actions'][:n]
        else:
            self._nb_actions[:n] = NB_ACTIONS_MAX
        self._nb_rempli      = n
        self.nb_total        = int(data['nb_total'][0])
        return True
 
    def __repr__(self) -> str:
        return (f"ReservoirBufferStrategie("
                f"{self._nb_rempli:,}/{self.taille_max:,} slots, "
                f"{self.nb_total:,} ajoutés au total)")
 
 
# =============================================================================
# RESERVOIR BUFFER — VALEUR (Point 6)
# =============================================================================

class ReservoirBufferValeur:
    """
    Reservoir buffer pour les échantillons de valeur de Deep CFR.

    Maintient un échantillon uniforme de paires
    (infoset_vec, valeur_cible) pour entraîner le ReseauValeur.

    La valeur cible = Σ_a strategie[a] × valeur[a] (valeur du nœud
    au moment de la traversée Deep CFR — calculée avant le stockage).

    Attributs
    ---------
    taille_max : capacité maximale
    nb_total   : total d'échantillons ajoutés
    _vecs      : vecteurs infoset (taille_max, DIM_INPUT)
    _valeurs   : valeurs scalaires (taille_max,) float32
    """

    def __init__(self, taille_max: int = TAILLE_BUFFER_DEFAUT):
        self.taille_max  : int = taille_max
        self.nb_total    : int = 0
        self._nb_rempli  : int = 0
        self._rng        = random.Random()

        self._vecs    = np.zeros((taille_max, DIM_INPUT), dtype=np.float32)
        self._valeurs = np.zeros(taille_max,              dtype=np.float32)

    def ajouter(self,
                infoset_vec : np.ndarray,
                valeur      : float) -> None:
        """
        Ajoute un échantillon (infoset_vec, valeur_cible) via Reservoir Sampling.

        infoset_vec : np.ndarray (DIM_INPUT,)
        valeur      : float — valeur du nœud (utilité espérée du joueur traversant)
        """
        self.nb_total += 1
        k = self.nb_total

        if self._nb_rempli < self.taille_max:
            idx = self._nb_rempli
            self._nb_rempli += 1
        else:
            j = self._rng.randint(1, k)
            if j > self.taille_max:
                return
            idx = j - 1

        self._vecs[idx]    = infoset_vec[:DIM_INPUT]
        self._valeurs[idx] = float(valeur)

    def echantillonner(
            self, taille_batch: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Tire un mini-batch aléatoire.

        Retourne (vecs, valeurs) ou None si pas assez de données.
          vecs    : np.ndarray (taille_batch, DIM_INPUT)
          valeurs : np.ndarray (taille_batch,) float32
        """
        n = self._nb_rempli
        if n < taille_batch:
            return None
        indices = np.random.choice(n, size=taille_batch, replace=False)
        return self._vecs[indices], self._valeurs[indices]

    def __len__(self)           : return self._nb_rempli
    def est_pret(self, b: int)  : return self._nb_rempli >= b
    def taux_remplissage(self)  : return self._nb_rempli / self.taille_max
    def reinitialiser(self)     : self.nb_total = 0; self._nb_rempli = 0

    def __repr__(self) -> str:
        return (f"ReservoirBufferValeur("
                f"{self._nb_rempli:,}/{self.taille_max:,} slots, "
                f"{self.nb_total:,} ajoutés au total)")


# =============================================================================
# UTILITAIRE — AFFICHAGE ÉTAT DES BUFFERS
# =============================================================================
 
def afficher_etat_buffers(
        buffers_regret    : List[ReservoirBufferRegret],
        buffers_strategie : List[ReservoirBufferStrategie]) -> None:
    """
    Affiche l'état de tous les buffers (un par joueur).
 
    buffers_regret    : liste de 3 ReservoirBufferRegret
    buffers_strategie : liste de 3 ReservoirBufferStrategie
    """
    print(f"\n  {'─'*55}")
    print(f"  {'Joueur':>7} | {'Buffer Regrets':>20} | {'Buffer Stratégie':>20}")
    print(f"  {'─'*7}-+-{'─'*20}-+-{'─'*20}")
    for i in range(3):
        br = buffers_regret[i]
        bs = buffers_strategie[i]
        print(f"  {i:>7} | "
              f"{len(br):>10,} / {br.taille_max:>8,} | "
              f"{len(bs):>10,} / {bs.taille_max:>8,}")
    print(f"  {'─'*55}\n")
 
 
def creer_buffers(
        taille_max: int = TAILLE_BUFFER_DEFAUT
) -> tuple:
    """
    Crée les 6 buffers nécessaires pour Deep CFR à 3 joueurs.
 
    Retourne
    --------
    tuple : (
        [ReservoirBufferRegret_J0,    ..._J1,    ..._J2],
        [ReservoirBufferStrategie_J0, ..._J1, ..._J2]
    )
    """
    buffers_regret    = [ReservoirBufferRegret(taille_max)    for _ in range(3)]
    buffers_strategie = [ReservoirBufferStrategie(taille_max) for _ in range(3)]
    return buffers_regret, buffers_strategie
 
 
# =============================================================================
# TEST RAPIDE (si exécuté directement)
# =============================================================================
 
if __name__ == '__main__':
    import math
 
    print("\n" + "="*60)
    print("  AXIOM — Test Reservoir Buffer (reservoir.py)")
    print("="*60)
 
    # ── Test Reservoir Buffer Regrets ──────────────────────────────────
    print("\n  Test ReservoirBufferRegret...")
    buf = ReservoirBufferRegret(taille_max=1000)
    assert len(buf) == 0
 
    # Ajouter 500 échantillons → buffer pas encore plein
    for i in range(500):
        vec     = np.random.randn(DIM_INPUT).astype(np.float32)
        regrets = np.random.randn(NB_ACTIONS_MAX).astype(np.float32)
        buf.ajouter(vec, regrets, NB_ACTIONS_MAX)
 
    assert len(buf) == 500, f"Attendu 500, obtenu {len(buf)}"
    assert buf.nb_total == 500
 
    # Ajouter 2000 de plus → reservoir sampling doit maintenir ~1000 slots
    for i in range(2000):
        vec     = np.random.randn(DIM_INPUT).astype(np.float32)
        regrets = np.random.randn(NB_ACTIONS_MAX).astype(np.float32)
        buf.ajouter(vec, regrets, NB_ACTIONS_MAX)
 
    assert len(buf) == 1000, f"Buffer plein attendu 1000, obtenu {len(buf)}"
    assert buf.nb_total == 2500
 
    # Test échantillonnage
    batch = buf.echantillonner(64)
    assert batch is not None
    vecs_b, regrets_b, nb_b = batch
    assert vecs_b.shape    == (64, DIM_INPUT),       f"Shape vecs: {vecs_b.shape}"
    assert regrets_b.shape == (64, NB_ACTIONS_MAX),  f"Shape regrets: {regrets_b.shape}"
    assert len(nb_b)       == 64
    print(f"  ✅ Buffer regrets : {buf}")
 
    # ── Test Reservoir Buffer Stratégie ───────────────────────────────
    print("\n  Test ReservoirBufferStrategie...")
    buf_s = ReservoirBufferStrategie(taille_max=500)
 
    for it in range(1, 1501):
        vec  = np.random.randn(DIM_INPUT).astype(np.float32)
        strat = np.random.dirichlet(np.ones(NB_ACTIONS_MAX)).astype(np.float32)
        buf_s.ajouter(vec, strat, iteration=it)
 
    assert len(buf_s) == 500
 
    batch_s = buf_s.echantillonner(32)
    assert batch_s is not None
    vecs_s, strats_s, iters_s = batch_s
    assert vecs_s.shape  == (32, DIM_INPUT)
    assert strats_s.shape == (32, NB_ACTIONS_MAX)
 
    # Vérifier que les stratégies somment à ~1 sur les actions légales
    # (les colonnes hors nb_actions seront à 0)
    sommes = strats_s.sum(axis=1)
    assert all(abs(s - 1.0) < 1e-5 for s in sommes), f"Stratégies non normalisées: {sommes[:5]}"
    print(f"  ✅ Buffer stratégie : {buf_s}")
 
    # ── Test creer_buffers ─────────────────────────────────────────────
    print("\n  Test creer_buffers (3 joueurs)...")
    br_list, bs_list = creer_buffers(taille_max=10_000)
    assert len(br_list) == 3 and len(bs_list) == 3
    afficher_etat_buffers(br_list, bs_list)
 
    # ── Propriété statistique du reservoir sampling ────────────────────
    # Vérifier que les échantillons récents ne sont pas sur-représentés.
    # Méthode : ajouter N échantillons étiquetés avec leur index,
    # puis vérifier que la distribution dans le buffer est ~uniforme.
    print("  Test propriété uniforme du reservoir sampling...")
    N = 10_000
    TAILLE = 1_000
    buf_stat = ReservoirBufferRegret(taille_max=TAILLE)
    for i in range(N):
        vec     = np.full(DIM_INPUT, float(i), dtype=np.float32)
        regrets = np.zeros(NB_ACTIONS_MAX, dtype=np.float32)
        buf_stat.ajouter(vec, regrets, NB_ACTIONS_MAX)
 
    # Récupérer tous les indices stockés (première feature = index original)
    indices_stockes = buf_stat._vecs[:TAILLE, 0]
    # Diviser en 10 déciles et vérifier que chaque décile est représenté
    decile = N // 10
    comptes = [
        ((indices_stockes >= i * decile) & (indices_stockes < (i + 1) * decile)).sum()
        for i in range(10)
    ]
    # Chaque décile devrait avoir ~100 échantillons (1000/10)
    # On tolère un écart de ±50% par rapport à l'espérance
    esperance = TAILLE / 10
    for i, c in enumerate(comptes):
        assert abs(c - esperance) < esperance * 0.6, \
            f"Décile {i}: {c} échantillons, attendu ~{esperance:.0f}"
    print(f"  ✅ Distribution uniforme vérifiée (comptes déciles : "
          f"{[int(c) for c in comptes]})")
 
    print("\n  ✅ Tous les tests reservoir sont passés !")
    print("="*60 + "\n")
 