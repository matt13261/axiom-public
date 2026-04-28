# =============================================================================
# AXIOM — engine/blind_structure.py
# Gestion des blindes croissantes (format tournoi).
# Suit le nombre de mains jouées et monte automatiquement de niveau.
# =============================================================================

from config.settings import NIVEAUX_BLINDES


class StructureBlinde:
    """
    Gère la progression des blindes au fil des mains.

    À chaque main terminée, on appelle avancer_main().
    Quand la durée d'un niveau est atteinte, on passe au niveau suivant.
    """

    def __init__(self):
        self.niveau_index  = 0     # index dans NIVEAUX_BLINDES
        self.mains_jouees  = 0     # mains jouées dans le niveau actuel
        self._charger_niveau()

    def _charger_niveau(self):
        """Charge les paramètres du niveau actuel."""
        niveau = NIVEAUX_BLINDES[self.niveau_index]
        self.petite_blinde = niveau[0]
        self.grande_blinde = niveau[1]
        self.duree         = niveau[2]   # nombre de mains avant passage au niveau suivant

    def avancer_main(self):
        """
        Appelé après chaque main.
        Incrémente le compteur et monte de niveau si nécessaire.
        """
        self.mains_jouees += 1
        if self.mains_jouees >= self.duree:
            self.mains_jouees = 0
            self._monter_niveau()

    def _monter_niveau(self):
        """Passe au niveau de blindes suivant (s'il existe)."""
        if self.niveau_index < len(NIVEAUX_BLINDES) - 1:
            self.niveau_index += 1
            self._charger_niveau()
        # Si on est au dernier niveau, on reste dessus

    @property
    def est_dernier_niveau(self) -> bool:
        return self.niveau_index >= len(NIVEAUX_BLINDES) - 1

    @property
    def mains_restantes_niveau(self) -> int:
        """Mains restantes avant le prochain changement de niveau."""
        return self.duree - self.mains_jouees

    @property
    def niveau_actuel(self) -> int:
        """Numéro du niveau actuel (commence à 1 pour l'affichage)."""
        return self.niveau_index + 1

    def __repr__(self):
        return (f"StructureBlinde(niveau={self.niveau_actuel} | "
                f"PB={self.petite_blinde} / GB={self.grande_blinde} | "
                f"mains={self.mains_jouees}/{self.duree})")
