# =============================================================================
# AXIOM — engine/game_state.py
# État complet d'une main en cours.
# =============================================================================

from enum import Enum, auto
from engine.card import DeckAXIOM, cartes_en_texte
from engine.player import Joueur, StatutJoueur


class Phase(Enum):
    PREFLOP  = auto()
    FLOP     = auto()
    TURN     = auto()
    RIVER    = auto()
    SHOWDOWN = auto()
    TERMINEE = auto()


# Mapping Phase → index 0-3 (compatible avec mccfr.py)
_PHASE_IDX = {
    Phase.PREFLOP  : 0,
    Phase.FLOP     : 1,
    Phase.TURN     : 2,
    Phase.RIVER    : 3,
    Phase.SHOWDOWN : 3,
    Phase.TERMINEE : 3,
}


class EtatJeu:
    """
    Représente l'état complet d'une main en cours.

    Nouveauté v2 : historique_phases
    ---------------------------------
    En plus de self.historique (toutes les actions de la main),
    on maintient self.historique_phases : liste de 4 strings (une par street),
    qui n'accumule QUE les actions de décision (pas les blindes) de la phase
    courante, dans le même format que mccfr.py.

    C'est ce que info_set.py utilise pour construire les clés MCCFR —
    garantissant la compatibilité parfaite entre entraînement et jeu.
    """

    def __init__(self, joueurs: list, petite_blinde: int, grande_blinde: int):
        self.joueurs          = joueurs
        self.petite_blinde    = petite_blinde
        self.grande_blinde    = grande_blinde
        self.pot              = 0
        self.pots_secondaires = []
        self.board            = []
        self.phase            = Phase.PREFLOP
        self.index_actif      = 0
        self.mise_courante    = 0
        self.mise_min_raise   = grande_blinde
        self.historique       = []               # [(joueur, action)] — main entière
        self.historique_phases = ['', '', '', ''] # actions par phase (compatible MCCFR)
        self.deck             = DeckAXIOM()
        self.position_dealer  = 0

    # ------------------------------------------------------------------
    # INITIALISATION D'UNE NOUVELLE MAIN
    # ------------------------------------------------------------------

    def nouvelle_main(self):
        self.pot               = 0
        self.pots_secondaires  = []
        self.board             = []
        self.phase             = Phase.PREFLOP
        self.mise_courante     = self.grande_blinde
        self.mise_min_raise    = self.grande_blinde
        self.historique        = []
        self.historique_phases = ['', '', '', '']   # reset à chaque main
        self.deck.melanger()

        for j in self.joueurs:
            j.reinitialiser_main()

        actifs = self.joueurs_non_elimines()
        if len(actifs) < 2:
            self.phase = Phase.TERMINEE
            return

        for j in actifs:
            j.recevoir_cartes(self.deck.distribuer(2))

        self._poster_blindes(actifs)
        self.index_actif = self._index_apres_bb(actifs)

    def _poster_blindes(self, actifs: list):
        """
        Poste les blindes.
        Les blindes sont enregistrées dans self.historique (pour l'affichage)
        mais PAS dans self.historique_phases (les blindes sont automatiques,
        pas des décisions — cohérent avec mccfr.py).
        """
        nb      = len(actifs)
        dealer  = self.position_dealer
        # Trier les actifs dans l'ordre de jeu à partir du dealer
        actifs_ordonnes = sorted(
            actifs, key=lambda j: (j.position - dealer) % nb)
        # actifs_ordonnes[0] = BTN, [1] = SB, [2] = BB
        sb_joueur = actifs_ordonnes[1 % nb]
        bb_joueur = actifs_ordonnes[2 % nb]

        mise_sb = sb_joueur.miser(self.petite_blinde)
        self.pot += mise_sb
        mise_bb = bb_joueur.miser(self.grande_blinde)
        self.pot += mise_bb

        self.mise_courante  = self.grande_blinde
        self.mise_min_raise = self.grande_blinde

        # Historique général uniquement (pas dans historique_phases)
        self.historique.append((sb_joueur, f"SB({mise_sb})"))
        self.historique.append((bb_joueur, f"BB({mise_bb})"))

    def _index_apres_bb(self, actifs: list) -> int:
        nb = len(actifs)
        dealer = self.position_dealer
        actifs_ordonnes = sorted(
            actifs, key=lambda j: (j.position - dealer) % nb)
        return self.joueurs.index(actifs_ordonnes[0])

    # ------------------------------------------------------------------
    # ENREGISTREMENT D'UNE ACTION (à appeler à chaque décision)
    # ------------------------------------------------------------------

    @staticmethod
    def _bucket_raise(frac: float) -> int:
        """
        Discrétise fraction raise/pot en bucket 1-4.
        Inline de _discretiser_raise_frac (évite import circulaire avec abstraction/).
        """
        if frac <= 0.33: return 1
        if frac <= 0.75: return 2
        if frac <= 1.25: return 3
        return 4

    def enregistrer_action(self, joueur, action):
        """
        Enregistre une action dans :
          - self.historique       (historique complet de la main)
          - self.historique_phases[phase_courante]  (compatible MCCFR)

        Point 3 : les RAISE sont encodés 'r{bucket}' (2 chars) plutôt que 'r',
        afin que le blueprint et le réseau Deep CFR distinguent les sizings.
        Le bucket est calculé sur mise_courante_avant_raise / pot courant.
        """
        from engine.actions import TypeAction

        # Historique général
        self.historique.append((joueur, action))

        # Historique par phase (code lettre, compatible mccfr.py)
        idx = _PHASE_IDX.get(self.phase, 0)
        if action.type == TypeAction.RAISE and action.montant and self.pot > 0:
            # Sizing bucket AVANT que le pot soit mis à jour par game.py
            frac   = action.montant / max(self.pot, 1)
            bucket = self._bucket_raise(frac)
            code   = f'r{bucket}'
        else:
            _codes = {
                TypeAction.FOLD  : 'f',
                TypeAction.CHECK : 'x',
                TypeAction.CALL  : 'c',
                TypeAction.RAISE : 'r',   # fallback si montant absent
                TypeAction.ALL_IN: 'a',
            }
            code = _codes.get(action.type, '?')
        self.historique_phases[idx] += code

    # ------------------------------------------------------------------
    # NAVIGATION ENTRE LES JOUEURS
    # ------------------------------------------------------------------

    def joueurs_non_elimines(self) -> list:
        return [j for j in self.joueurs if not j.est_elimine]

    def joueurs_actifs_dans_main(self) -> list:
        return [j for j in self.joueurs
                if j.statut in (StatutJoueur.ACTIF, StatutJoueur.ALL_IN)]

    def joueurs_pouvant_agir(self) -> list:
        return [j for j in self.joueurs if j.peut_agir]

    def joueur_actif(self) -> Joueur:
        return self.joueurs[self.index_actif]

    def passer_au_suivant(self):
        nb = len(self.joueurs)
        for i in range(1, nb + 1):
            suivant = (self.index_actif + i) % nb
            if self.joueurs[suivant].peut_agir:
                self.index_actif = suivant
                return

    # ------------------------------------------------------------------
    # GESTION DU POT
    # ------------------------------------------------------------------

    def ajouter_au_pot(self, montant: int):
        self.pot += montant

    # ------------------------------------------------------------------
    # PROGRESSION DES PHASES
    # ------------------------------------------------------------------

    def passer_phase_suivante(self):
        for j in self.joueurs:
            j.reinitialiser_tour()
        self.mise_courante  = 0
        self.mise_min_raise = self.grande_blinde

        if self.phase == Phase.PREFLOP:
            self.board  = self.deck.distribuer(3)
            self.phase  = Phase.FLOP
        elif self.phase == Phase.FLOP:
            self.board += self.deck.distribuer(1)
            self.phase  = Phase.TURN
        elif self.phase == Phase.TURN:
            self.board += self.deck.distribuer(1)
            self.phase  = Phase.RIVER
        elif self.phase == Phase.RIVER:
            self.phase = Phase.SHOWDOWN

        actifs = [j for j in self.joueurs if j.peut_agir]
        if actifs:
            self.index_actif = self.joueurs.index(actifs[0])

    # ------------------------------------------------------------------
    # AFFICHAGE
    # ------------------------------------------------------------------

    def afficher(self):
        print(f"\n{'='*50}")
        print(f"Phase : {self.phase.name} | Pot : {self.pot}")
        print(f"Board : {cartes_en_texte(self.board) if self.board else '(vide)'}")
        print(f"Blindes : {self.petite_blinde}/{self.grande_blinde}")
        print(f"Mise courante : {self.mise_courante}")
        print()
        for j in self.joueurs:
            cartes_str = cartes_en_texte(j.cartes) if j.cartes else '??'
            actif_str  = " ← À TOI" if self.joueurs.index(j) == self.index_actif else ""
            print(f"  {j.nom:10} | stack={j.stack:5} | mise_tour={j.mise_tour:4} | "
                  f"{j.statut.name:8} | {cartes_str}{actif_str}")
        print(f"{'='*50}\n")

    def __repr__(self):
        return f"EtatJeu(phase={self.phase.name}, pot={self.pot})"
