# =============================================================================
# AXIOM — ai/strategy.py
# Sauvegarde et chargement de la stratégie blueprint.
#
# Le blueprint est le résultat de l'entraînement MCCFR : un dictionnaire
# de NoeudCFR indexé par clé d'infoset.
# Il est sérialisé en pickle pour être chargé rapidement en production.
#
# Usage :
#   sauvegarder_blueprint(mccfr.noeuds, 'data/strategy/blueprint_v1.pkl')
#   noeuds = charger_blueprint('data/strategy/blueprint_v1.pkl')
#   strat  = obtenir_strategie(noeuds, cle)
# =============================================================================

import os
import pickle
from ai.mccfr import NoeudCFR


# -----------------------------------------------------------------------------
# SAUVEGARDE
# -----------------------------------------------------------------------------

def sauvegarder_blueprint(noeuds: dict, chemin: str) -> None:
    """
    Sauvegarde le dictionnaire de NoeudCFR en fichier pickle.

    Crée automatiquement les dossiers parents si nécessaire.
    Le fichier peut être rechargé avec charger_blueprint().

    noeuds : dict {cle_infoset (str) → NoeudCFR}
    chemin : chemin vers le fichier .pkl (ex: 'data/strategy/blueprint_v1.pkl')
    """
    # Créer les dossiers parents si nécessaire
    dossier = os.path.dirname(chemin)
    if dossier:
        os.makedirs(dossier, exist_ok=True)

    with open(chemin, 'wb') as f:
        pickle.dump(noeuds, f, protocol=pickle.HIGHEST_PROTOCOL)

    nb_infosets = len(noeuds)
    taille_ko   = os.path.getsize(chemin) // 1024
    print(f"  💾 Blueprint sauvegardé : {chemin}")
    print(f"     {nb_infosets:,} infosets | {taille_ko:,} Ko")


# -----------------------------------------------------------------------------
# CHARGEMENT
# -----------------------------------------------------------------------------

def charger_blueprint(chemin: str) -> dict:
    """
    Charge un blueprint depuis un fichier pickle.

    Retourne un dictionnaire {cle_infoset (str) → NoeudCFR}.
    Lève FileNotFoundError si le fichier est introuvable.

    chemin : chemin vers le fichier .pkl
    """
    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"Blueprint introuvable : '{chemin}'\n"
            f"Lancez d'abord l'entraînement MCCFR pour générer ce fichier."
        )

    with open(chemin, 'rb') as f:
        noeuds = pickle.load(f)

    taille_ko = os.path.getsize(chemin) // 1024
    print(f"  📂 Blueprint chargé : {chemin}")
    print(f"     {len(noeuds):,} infosets | {taille_ko:,} Ko")
    return noeuds


# -----------------------------------------------------------------------------
# STATISTIQUES
# -----------------------------------------------------------------------------

def afficher_stats_blueprint(noeuds: dict) -> None:
    """
    Affiche des statistiques détaillées sur le blueprint.

    Pour chaque phase (PREFLOP, FLOP, TURN, RIVER) :
      - Nombre d'infosets
      - Nombre moyen de visites
      - Nombre moyen d'actions disponibles

    Affiche aussi les 5 infosets les plus visités (les plus importants).

    noeuds : dict {cle_infoset (str) → NoeudCFR}
    """
    if not noeuds:
        print("  Blueprint vide — aucun infoset.")
        return

    phases = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
    stats_phases = {
        p: {'nb': 0, 'visites': 0, 'actions': 0}
        for p in phases
    }

    for cle, noeud in noeuds.items():
        for p in phases:
            if cle.startswith(p + '|'):
                stats_phases[p]['nb']      += 1
                stats_phases[p]['visites'] += noeud.nb_visites
                stats_phases[p]['actions'] += noeud.nb_actions
                break

    total_noeuds  = len(noeuds)
    total_visites = sum(n.nb_visites for n in noeuds.values())

    print(f"\n{'═'*55}")
    print(f"  AXIOM — Statistiques Blueprint")
    print(f"{'═'*55}")
    print(f"  Infosets total   : {total_noeuds:,}")
    print(f"  Visites totales  : {total_visites:,}")
    print(f"\n  {'Phase':8} | {'Infosets':>10} | {'Vis. moy':>9} | {'Act. moy':>8}")
    print(f"  {'─'*8}-+-{'─'*10}-+-{'─'*9}-+-{'─'*8}")

    for p in phases:
        s  = stats_phases[p]
        nb = s['nb']
        if nb > 0:
            vis_moy = s['visites'] / nb
            act_moy = s['actions'] / nb
        else:
            vis_moy = act_moy = 0.0
        print(f"  {p:8} | {nb:>10,} | {vis_moy:>9.1f} | {act_moy:>8.1f}")

    # Top 5 infosets les plus visités
    if noeuds:
        top5 = sorted(noeuds.items(),
                      key=lambda x: x[1].nb_visites,
                      reverse=True)[:5]
        print(f"\n  Top 5 infosets les plus visités :")
        for i, (cle, noeud) in enumerate(top5, 1):
            strat = noeud.strategie_moyenne()
            strat_str = ' '.join(f"{p:.2f}" for p in strat[:4])
            cle_courte = cle[:50] + ('...' if len(cle) > 50 else '')
            print(f"  {i}. [{noeud.nb_visites:,} vis] {cle_courte}")
            print(f"     Strat. moy : [{strat_str}]")

    print(f"{'═'*55}\n")


# -----------------------------------------------------------------------------
# CONSULTATION DE LA STRATÉGIE
# -----------------------------------------------------------------------------

def obtenir_strategie(noeuds: dict, cle: str) -> list:
    """
    Retourne la stratégie moyenne pour une clé d'infoset donnée.

    La stratégie moyenne est la distribution de probabilités sur les actions
    qui converge vers l'Équilibre de Nash après l'entraînement MCCFR.

    noeuds : dict {cle_infoset (str) → NoeudCFR}
    cle    : clé d'infoset (format "PHASE|pos=X|bucket=Y|pot=Z|stacks=...|hist=")
    retour : liste de probabilités (somme = 1.0), ou None si clé inconnue
    """
    if cle not in noeuds:
        return None
    return noeuds[cle].strategie_moyenne()