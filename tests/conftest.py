"""
conftest.py — Fixtures et helpers partagés pour les tests AXIOM.
"""

import numpy as np
import pytest


def mock_centroides(seed=42):
    """
    Centroïdes pseudo-réalistes pour tests de AbstractionCartesV2.

    Distribués dans l'espace features (E[HS], E[HS²], Potentiel) :
    - flop  : potentiel dans [-0.3, 0.3]
    - turn  : potentiel dans [-0.2, 0.2]
    - river : potentiel ~0 (board complet)

    Usage :
        v2 = AbstractionCartesV2(centroides=mock_centroides())
    """
    rng = np.random.RandomState(seed)
    return {
        'flop':  (rng.rand(50, 3) * [1.0, 1.0, 0.6] - [0.0, 0.0, 0.3]).astype(np.float32),
        'turn':  (rng.rand(50, 3) * [1.0, 1.0, 0.4] - [0.0, 0.0, 0.2]).astype(np.float32),
        'river': (rng.rand(50, 3) * [1.0, 1.0, 0.05] - [0.0, 0.0, 0.025]).astype(np.float32),
    }


@pytest.fixture
def centroides_mock():
    """Fixture pytest exposant les centroïdes mock pour les tests V2."""
    return mock_centroides()
