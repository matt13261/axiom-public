"""
OpponentTracker — collecte de stats comportementales par seat adversaire.

Tracking par seat (index de position 0-2), pas par identite persistante.
Limitation MVP : si un joueur change de seat entre les mains, les stats
ne suivent pas l'individu. Suffisant pour les bots de test (style fixe).

Ref spec : docs/investigations/P1-winrates-negatifs/experiments/04-H4-exploit/spec.md
"""
from collections import defaultdict, deque

WINDOW_DEFAULT      = 30
CONFIANCE_MIN_MAINS = 5
CONFIANCE_MAX_MAINS = 30


class OpponentTracker:
    """Fenetre glissante de stats par seat pour le module ExploitMixer.

    Stocke les WINDOW_DEFAULT dernieres observations par seat.
    Chaque observation : {'action': int, 'contexte': dict}

    Actions : 0=FOLD, 1=CHECK, 2=CALL, 3+=RAISE
    Contexte : cles optionnelles 'phase' ('preflop'/'postflop'),
               'est_cbet_opp' (bool)
    """

    def __init__(self, window=WINDOW_DEFAULT):
        # deque(maxlen=window) : les vieilles observations expirent automatiquement
        self._obs = defaultdict(lambda: deque(maxlen=window))

    def observer_action(self, seat_index, action, contexte):
        """Enregistre une observation de l'adversaire au seat donne."""
        self._obs[seat_index].append({"action": action, "contexte": contexte})

    def mains_observees(self, seat_index):
        """Nombre d'observations dans la fenetre courante pour ce seat."""
        return len(self._obs[seat_index])

    def confiance(self, seat_index):
        """Lerp lineaire 0 → 1 entre CONFIANCE_MIN_MAINS et CONFIANCE_MAX_MAINS.

        n < 5        : 0.0   (blueprint pur garanti)
        5 <= n < 30  : (n - 5) / 25
        n >= 30      : 1.0
        """
        n = self.mains_observees(seat_index)
        if n < CONFIANCE_MIN_MAINS:
            return 0.0
        if n >= CONFIANCE_MAX_MAINS:
            return 1.0
        return (n - CONFIANCE_MIN_MAINS) / (CONFIANCE_MAX_MAINS - CONFIANCE_MIN_MAINS)

    def fold_to_cbet(self, seat_index):
        """Taux de fold face a une continuation bet opportunity.

        fold_to_cbet = nb_folds_vs_cbet / nb_cbet_opportunities
        Retourne 0.0 si aucune cbet opportunity observee.
        """
        obs_cbet = [
            o for o in self._obs[seat_index]
            if o["contexte"].get("est_cbet_opp")
        ]
        if not obs_cbet:
            return 0.0
        folds = sum(1 for o in obs_cbet if o["action"] == 0)
        return folds / len(obs_cbet)

    def pfr(self, seat_index):
        """Taux de raise preflop (Preflop Raise Rate).

        pfr = nb_raises_preflop / nb_actions_preflop
        Raise preflop : action >= 3 (RAISE ou ALL_IN — exclu CALL).
        Retourne 0.0 si aucune action preflop observee.
        """
        obs_preflop = [
            o for o in self._obs[seat_index]
            if o["contexte"].get("phase") == "preflop"
        ]
        if not obs_preflop:
            return 0.0
        raises = sum(1 for o in obs_preflop if o["action"] >= 3)
        return raises / len(obs_preflop)

    def vpip(self, seat_index):
        """Taux d'entree volontaire dans le pot preflop (CALL ou RAISE).

        vpip = nb_entrees_volontaires / nb_actions_preflop
        Entree volontaire : action >= 2 (CALL ou RAISE, hors blind force).
        Retourne 0.0 si aucune action preflop observee.
        """
        obs_preflop = [
            o for o in self._obs[seat_index]
            if o["contexte"].get("phase") == "preflop"
        ]
        if not obs_preflop:
            return 0.0
        entrees = sum(1 for o in obs_preflop if o["action"] >= 2)
        return entrees / len(obs_preflop)
