# AXIOM — main.py
# Point d'entrée principal : lance une partie complète 1 humain vs 2 AXIOM
# avec interface graphique Pygame.
# =============================================================================
#
# Usage :
#     python main.py
#     python main.py --nom "Alice"
#     python main.py --nom "Alice" --mode deterministe
#     python main.py --blueprint data/strategy/blueprint_v1.pkl
#
# =============================================================================

import os
import sys
import argparse

# ---------------------------------------------------------------------------
# Vérification des dépendances avant tout import lourd
# ---------------------------------------------------------------------------

def _verifier_dependances() -> bool:
    """Vérifie que les dépendances critiques sont installées."""
    manquantes = []
    for module in ("pygame", "torch", "numpy", "treys"):
        try:
            __import__(module)
        except ImportError:
            manquantes.append(module)
    if manquantes:
        print("❌ Dépendances manquantes :", ", ".join(manquantes))
        print("   Installe-les avec : pip install " + " ".join(manquantes))
        return False
    return True


# ---------------------------------------------------------------------------
# Imports projet
# ---------------------------------------------------------------------------

def _importer_modules():
    """
    Importe les modules AXIOM.
    Retourne un dict des objets nécessaires, ou None en cas d'erreur.
    """
    try:
        from engine.player import Joueur, TypeJoueur
        from engine.game_state import EtatJeu
        from engine.blind_structure import StructureBlinde
        from ai.agent import creer_agent
        from interface.main_window import FenetrePrincipale
        from config.settings import STACK_DEPART, NB_JOUEURS

        return {
            "Joueur"           : Joueur,
            "TypeJoueur"       : TypeJoueur,
            "EtatJeu"          : EtatJeu,
            "StructureBlinde"  : StructureBlinde,
            "creer_agent"      : creer_agent,
            "FenetrePrincipale": FenetrePrincipale,
            "STACK_DEPART"     : STACK_DEPART,
            "NB_JOUEURS"       : NB_JOUEURS,
        }
    except ImportError as e:
        print(f"❌ Erreur d'import : {e}")
        print("   Vérifie que tu es bien dans le dossier axiom/ et que"
              " toutes les phases sont en place.")
        return None


# ---------------------------------------------------------------------------
# Création de la partie
# ---------------------------------------------------------------------------

def creer_joueurs(nom_humain: str, modules: dict) -> list:
    """
    Crée la liste des 3 joueurs :
      - Position 0 (BTN, bas-centre)  : joueur humain
      - Position 1 (SB, haut-gauche)  : AXIOM-1
      - Position 2 (BB, haut-droite)  : AXIOM-2

    Retourne la liste [humain, axiom1, axiom2].
    """
    Joueur     = modules["Joueur"]
    TypeJoueur = modules["TypeJoueur"]
    stack      = modules["STACK_DEPART"]

    humain = Joueur(nom_humain, TypeJoueur.HUMAIN, stack, position=0)
    axiom1 = Joueur("AXIOM-1",  TypeJoueur.AXIOM,  stack, position=1)
    axiom2 = Joueur("AXIOM-2",  TypeJoueur.AXIOM,  stack, position=2)

    return [humain, axiom1, axiom2]


def initialiser_partie(nom_humain: str,
                       chemin_blueprint: str,
                       mode_deterministe: bool,
                       modules: dict) -> dict:
    """
    Initialise tous les objets nécessaires à une partie :
      - Joueurs
      - EtatJeu
      - StructureBlinde
      - AgentAXIOM

    Retourne un dict avec les objets prêts.
    """
    EtatJeu         = modules["EtatJeu"]
    StructureBlinde = modules["StructureBlinde"]
    creer_agent     = modules["creer_agent"]

    # ── Joueurs ────────────────────────────────────────────────────────────
    joueurs = creer_joueurs(nom_humain, modules)

    # ── Structure des blindes ──────────────────────────────────────────────
    structure = StructureBlinde()

    # ── État de jeu ────────────────────────────────────────────────────────
    etat = EtatJeu(
        joueurs,
        petite_blinde = structure.petite_blinde,
        grande_blinde = structure.grande_blinde,
    )

    # ── Agent AXIOM ────────────────────────────────────────────────────────
    # creer_agent() tente de charger le blueprint et Deep CFR s'ils existent.
    # Si les fichiers sont absents (première utilisation avant entraînement),
    # il bascule en mode heuristique sans lever d'exception.
    agent = creer_agent(
        chemin_blueprint  = chemin_blueprint,
        mode_deterministe = mode_deterministe,
        verbose           = True,
    )

    return {
        "joueurs"  : joueurs,
        "etat"     : etat,
        "structure": structure,
        "agent"    : agent,
    }


# ---------------------------------------------------------------------------
# Écran d'accueil console (avant Pygame)
# ---------------------------------------------------------------------------

def afficher_accueil(nom_humain: str, agent) -> None:
    """Affiche un résumé de la configuration dans le terminal."""
    largeur = 56
    print("\n" + "=" * largeur)
    print("  ██████╗ ██╗  ██╗██╗ ██████╗ ███╗   ███╗")
    print("  ██╔══██╗╚██╗██╔╝██║██╔═══██╗████╗ ████║")
    print("  ███████║ ╚███╔╝ ██║██║   ██║██╔████╔██║")
    print("  ██╔══██║ ██╔██╗ ██║██║   ██║██║╚██╔╝██║")
    print("  ██║  ██║██╔╝ ██╗██║╚██████╔╝██║ ╚═╝ ██║")
    print("  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝     ╚═╝")
    print("=" * largeur)
    print(f"  Texas Hold'em No Limit — 1 vs 2 AXIOM Bots")
    print("=" * largeur)
    print(f"  Joueur humain  : {nom_humain}")
    print(f"  Adversaires    : AXIOM-1, AXIOM-2")
    print(f"  Agent          : {agent}")
    print("─" * largeur)
    print("  Raccourcis clavier en jeu :")
    print("    [F] Fold   [C] Check/Call   [A] All-in")
    print("    [Échap] Quitter")
    print("─" * largeur)
    print("  Lancement de l'interface graphique…")
    print("=" * largeur + "\n")


# ---------------------------------------------------------------------------
# POINT D'ENTRÉE PRINCIPAL
# ---------------------------------------------------------------------------

def main() -> None:
    """Lance une partie AXIOM complète."""

    # ── Parsing des arguments ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="AXIOM — IA de poker Texas Hold'em niveau professionnel",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--nom",
        type=str,
        default="Joueur",
        help="Ton nom (affiché dans l'interface)",
    )
    parser.add_argument(
        "--blueprint",
        type=str,
        default=None,
        help="Chemin vers le fichier blueprint .pkl\n"
             "(par défaut : data/strategy/blueprint_v1.pkl)",
    )
    parser.add_argument(
        "--mode",
        choices=["stochastique", "deterministe"],
        default="stochastique",
        help="Mode de décision de l'IA :\n"
             "  stochastique  : l'IA échantillonne sa distribution (défaut)\n"
             "  deterministe  : l'IA joue toujours son meilleur coup",
    )
    args = parser.parse_args()

    nom_humain        = args.nom
    chemin_blueprint  = args.blueprint
    mode_deterministe = (args.mode == "deterministe")

    # ── Vérification des dépendances ───────────────────────────────────────
    if not _verifier_dependances():
        sys.exit(1)

    # ── Import des modules ─────────────────────────────────────────────────
    modules = _importer_modules()
    if modules is None:
        sys.exit(1)

    # ── Initialisation de la partie ────────────────────────────────────────
    print("\n🃏 Initialisation de la partie AXIOM…")
    partie = initialiser_partie(
        nom_humain        = nom_humain,
        chemin_blueprint  = chemin_blueprint,
        mode_deterministe = mode_deterministe,
        modules           = modules,
    )

    # ── Écran d'accueil console ────────────────────────────────────────────
    afficher_accueil(nom_humain, partie["agent"])

    # ── Lancement de l'interface graphique ────────────────────────────────
    FenetrePrincipale = modules["FenetrePrincipale"]

    fenetre = FenetrePrincipale()
    fenetre.lancer(
        etat_jeu         = partie["etat"],
        agent            = partie["agent"],
        structure_blinde = partie["structure"],
        index_humain     = 0,   # le joueur humain est toujours à l'index 0
    )

    # ── Statistiques de fin de partie ─────────────────────────────────────
    # (atteint uniquement si l'interface se ferme proprement sans sys.exit)
    partie["agent"].afficher_stats()


# ---------------------------------------------------------------------------
# ENTRÉE DU SCRIPT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ajouter le dossier parent au sys.path si nécessaire
    # (permet de lancer depuis n'importe quel répertoire)
    dossier_script = os.path.dirname(os.path.abspath(__file__))
    if dossier_script not in sys.path:
        sys.path.insert(0, dossier_script)

    main()
