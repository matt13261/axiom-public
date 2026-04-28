# =============================================================================
# AXIOM — engine/game.py
# Boucle de jeu principale.
# Enchaîne les mains, gère les tours de parole, distribue les pots,
# élimine les joueurs et détermine le vainqueur du tournoi.
# =============================================================================

from engine.game_state import EtatJeu, Phase
from engine.player import Joueur, StatutJoueur, TypeJoueur
from engine.actions import Action, TypeAction, actions_legales
from engine.hand_evaluator import determiner_gagnants, classe_main, score_main
from engine.blind_structure import StructureBlinde
from config.settings import NB_JOUEURS, STACK_DEPART


class Jeu:
    """
    Orchestre une partie complète de Texas Hold'em No Limit à 3 joueurs.

    Usage :
        jeu = Jeu(joueurs, agent_ia)
        jeu.lancer()
    """

    def __init__(self, joueurs: list, agent_ia=None):
        """
        joueurs  : liste d'objets Joueur
        agent_ia : instance de l'IA (agent.py), ou None pour mode console
        """
        self.joueurs        = joueurs
        self.agent_ia       = agent_ia
        self.blindes        = StructureBlinde()
        self.etat           = EtatJeu(joueurs,
                                      self.blindes.petite_blinde,
                                      self.blindes.grande_blinde)
        self.main_numero    = 0
        self.en_cours       = True

    # ------------------------------------------------------------------
    # BOUCLE PRINCIPALE
    # ------------------------------------------------------------------

    def lancer(self):
        """Lance la partie complète jusqu'à ce qu'il reste 1 joueur."""
        print(f"\n{'#'*50}")
        print(f"  AXIOM — Tournoi Texas Hold'em No Limit 3 joueurs")
        print(f"{'#'*50}\n")

        while self.en_cours:
            actifs = self.etat.joueurs_non_elimines()
            if len(actifs) == 1:
                print(f"\n🏆 VAINQUEUR : {actifs[0].nom} avec {actifs[0].stack} jetons !")
                self.en_cours = False
                break

            self.jouer_une_main()

    def jouer_une_main(self):
        """Joue une main complète de A à Z."""
        self.main_numero += 1
        print(f"\n--- Main #{self.main_numero} | Niveau {self.blindes.niveau_actuel} "
              f"| Blindes {self.blindes.petite_blinde}/{self.blindes.grande_blinde} ---")

        # Mettre à jour les blindes dans l'état
        self.etat.petite_blinde = self.blindes.petite_blinde
        self.etat.grande_blinde = self.blindes.grande_blinde
        self.etat.mise_min_raise = self.blindes.grande_blinde

        # Lancer la main
        self.etat.nouvelle_main()
        self.etat.afficher()

        # Jouer chaque street
        for phase in [Phase.PREFLOP, Phase.FLOP, Phase.TURN, Phase.RIVER]:
            if self.etat.phase == Phase.TERMINEE:
                break

            # Vérifier s'il reste au moins 2 joueurs actifs
            if len(self.etat.joueurs_actifs_dans_main()) < 2:
                break

            self._jouer_tour()

            # Passer à la street suivante si la main continue
            if (self.etat.phase not in (Phase.SHOWDOWN, Phase.TERMINEE)
                    and len(self.etat.joueurs_actifs_dans_main()) >= 2):
                self.etat.passer_phase_suivante()
                self.etat.afficher()

        # Showdown ou gain par défaut
        self._resoudre_main()

        # Avancer les blindes
        self.blindes.avancer_main()

        # Faire tourner le bouton (changer la position du dealer)
        self._faire_tourner_bouton()

    # ------------------------------------------------------------------
    # TOUR DE PAROLE
    # ------------------------------------------------------------------

    def _jouer_tour(self):
        """
        Gère un tour de parole complet (une street).
        Continue jusqu'à ce que tous les joueurs aient égalisé la mise
        ou soient fold/all-in.
        """
        nb_joueurs_pouvant_agir = len(self.etat.joueurs_pouvant_agir())
        if nb_joueurs_pouvant_agir == 0:
            return

        # Compteur pour éviter les boucles infinies
        actions_ce_tour = 0
        max_actions     = nb_joueurs_pouvant_agir * 4   # sécurité

        while actions_ce_tour < max_actions:
            joueur = self.etat.joueur_actif()

            # Vérifier si le tour est terminé (tout le monde a suivi ou fold)
            if self._tour_termine():
                break

            if not joueur.peut_agir:
                self.etat.passer_au_suivant()
                continue

            # Calculer les actions légales
            legales = actions_legales(
                joueur,
                mise_a_suivre  = self.etat.mise_courante,
                pot            = self.etat.pot,
                mise_min_raise = self.etat.mise_min_raise
            )

            # Obtenir l'action (humain ou IA)
            action = self._obtenir_action(joueur, legales)

            # Appliquer l'action
            self._appliquer_action(joueur, action)
            self.etat.enregistrer_action(joueur, action)

            print(f"  {joueur.nom} → {action}")

            # Passer au joueur suivant
            self.etat.passer_au_suivant()
            actions_ce_tour += 1

            # Vérifier si la main est terminée (1 seul joueur restant)
            if len(self.etat.joueurs_actifs_dans_main()) <= 1:
                break

    def _tour_termine(self) -> bool:
        """
        Le tour est terminé quand tous les joueurs actifs ont mis le même montant
        ET que chacun a eu la possibilité d'agir au moins une fois.
        """
        pouvant_agir = self.etat.joueurs_pouvant_agir()
        if not pouvant_agir:
            return True

        # Tous les joueurs actifs ont-ils la même mise que la mise courante ?
        return all(j.mise_tour == self.etat.mise_courante for j in pouvant_agir)

    # ------------------------------------------------------------------
    # OBTENIR UNE ACTION
    # ------------------------------------------------------------------

    def _obtenir_action(self, joueur: Joueur, legales: list) -> Action:
        """
        Demande une action au joueur.
        Si c'est un humain → saisie console (sera remplacé par l'interface graphique).
        Si c'est l'IA → délègue à l'agent.
        """
        if joueur.type == TypeJoueur.HUMAIN:
            return self._saisie_console(joueur, legales)
        else:
            # IA : on passe l'état complet à l'agent
            if self.agent_ia:
                return self.agent_ia.choisir_action(self.etat, joueur, legales)
            else:
                # Pas d'agent IA connecté → call par défaut
                for a in legales:
                    if a.type == TypeAction.CALL:
                        return a
                return legales[0]

    def _saisie_console(self, joueur: Joueur, legales: list) -> Action:
        """Interface console temporaire pour le joueur humain."""
        print(f"\nActions disponibles pour {joueur.nom} (stack: {joueur.stack}) :")
        for i, a in enumerate(legales):
            print(f"  [{i}] {a}")

        while True:
            try:
                choix = int(input("Ton choix : "))
                if 0 <= choix < len(legales):
                    return legales[choix]
            except ValueError:
                pass
            print("Choix invalide, réessaie.")

    # ------------------------------------------------------------------
    # APPLIQUER UNE ACTION
    # ------------------------------------------------------------------

    def _appliquer_action(self, joueur: Joueur, action: Action):
        """Applique une action sur l'état du jeu."""
        if action.type == TypeAction.FOLD:
            joueur.statut = StatutJoueur.FOLD

        elif action.type == TypeAction.CHECK:
            pass   # rien à faire

        elif action.type == TypeAction.CALL:
            a_payer = self.etat.mise_courante - joueur.mise_tour
            mise_reelle = joueur.miser(a_payer)
            self.etat.pot += mise_reelle

        elif action.type in (TypeAction.RAISE, TypeAction.ALL_IN):
            # montant = mise totale visée par ce joueur
            a_payer = action.montant - joueur.mise_tour
            mise_reelle = joueur.miser(a_payer)
            self.etat.pot += mise_reelle

            if action.montant > self.etat.mise_courante:
                # Mettre à jour le minimum de raise
                raise_increment = action.montant - self.etat.mise_courante
                self.etat.mise_min_raise = max(self.etat.mise_min_raise, raise_increment)
                self.etat.mise_courante  = action.montant

    # ------------------------------------------------------------------
    # RÉSOLUTION DE LA MAIN
    # ------------------------------------------------------------------

    def _resoudre_main(self):
        """
        Détermine le(s) gagnant(s) et distribue le pot.
        Cas 1 : un seul joueur restant (tous les autres ont fold) → il gagne tout.
        Cas 2 : showdown → évaluation des mains.
        """
        actifs = self.etat.joueurs_actifs_dans_main()

        if len(actifs) == 1:
            # Gain par défaut
            gagnant = actifs[0]
            gagnant.recevoir(self.etat.pot)
            print(f"\n  → {gagnant.nom} remporte {self.etat.pot} jetons (tous fold)")
        else:
            # Showdown
            print(f"\n  *** SHOWDOWN ***")
            for j in actifs:
                from engine.card import cartes_en_texte
                score = score_main(j.cartes, self.etat.board)
                combi = classe_main(score)
                print(f"  {j.nom} : {cartes_en_texte(j.cartes)} → {combi}")

            gagnants = determiner_gagnants(actifs, self.etat.board)
            gain_par_gagnant = self.etat.pot // len(gagnants)

            for g in gagnants:
                g.recevoir(gain_par_gagnant)
                print(f"  → {g.nom} remporte {gain_par_gagnant} jetons")

        # Éliminer les joueurs sans jetons
        for j in self.joueurs:
            if j.stack == 0:
                j.statut = StatutJoueur.ELIMINE
                print(f"  ❌ {j.nom} est éliminé !")

        print()

    # ------------------------------------------------------------------
    # ROTATION DU BOUTON
    # ------------------------------------------------------------------

    def _faire_tourner_bouton(self):
        """
        Fait tourner le bouton dealer dans le sens des aiguilles d'une montre.
        Les joueurs sont réordonnés pour la prochaine main.
        """
        actifs = self.etat.joueurs_non_elimines()
        if len(actifs) >= 2:
            # Faire pivoter la liste des joueurs d'une position
            self.joueurs.append(self.joueurs.pop(0))
            self.etat.joueurs = self.joueurs


# ------------------------------------------------------------------
# FONCTION DE CRÉATION RAPIDE D'UNE PARTIE
# ------------------------------------------------------------------

def creer_partie(noms_humains: list = None, agent_ia=None) -> Jeu:
    """
    Crée une partie avec NB_JOUEURS joueurs.

    noms_humains : liste de noms pour les joueurs humains (max NB_JOUEURS)
    agent_ia     : instance de l'IA AXIOM

    Les joueurs sans nom humain seront des bots AXIOM.
    """
    from config.settings import NB_JOUEURS, STACK_DEPART

    noms_humains = noms_humains or []
    joueurs = []

    for i in range(NB_JOUEURS):
        if i < len(noms_humains):
            j = Joueur(noms_humains[i], TypeJoueur.HUMAIN, STACK_DEPART, i)
        else:
            j = Joueur(f"AXIOM-{i+1}", TypeJoueur.AXIOM, STACK_DEPART, i)
        joueurs.append(j)

    return Jeu(joueurs, agent_ia)
