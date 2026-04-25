"""
TDD — Déterminisme du bucketing MC postflop (H3-final).

Seed=42 fixée avant chaque appel MC + Deck.GetFullDeck() (ordre fixe).
NB_SIMULATIONS reste 100 pour compatibilité blueprint.
"""
from treys import Card


def test_bucket_postflop_deterministe():
    """
    10 appels identiques sur la même main+board doivent retourner le même bucket.
    """
    from abstraction.card_abstraction import AbstractionCartes

    ab     = AbstractionCartes(nb_simulations=100, mode='3max')
    cartes = [Card.new('8c'), Card.new('7c')]
    board  = [Card.new('As'), Card.new('Kh'), Card.new('Qs')]

    resultats = [ab.bucket_postflop(cartes, board) for _ in range(10)]
    assert len(set(resultats)) == 1, (
        f"Bucket postflop non deterministe sur 10 appels : {resultats}"
    )


def test_bucket_postflop_deterministe_main_limite():
    """
    20 appels sur une main en frontière de bucket doivent tous retourner le même résultat.
    """
    from abstraction.card_abstraction import AbstractionCartes

    ab     = AbstractionCartes(nb_simulations=100, mode='3max')
    cartes = [Card.new('Js'), Card.new('Tc')]
    board  = [Card.new('Kd'), Card.new('5h'), Card.new('2c')]

    resultats = [ab.bucket_postflop(cartes, board) for _ in range(20)]
    assert len(set(resultats)) == 1, (
        f"Bucket non deterministe (20 appels, main limite) : {resultats}"
    )


def test_run_exp01_final_script_existe():
    import os
    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'docs', 'investigations', 'P1-winrates-negatifs',
        'experiments', '01-H3-final', 'run_exp01_final.py'
    )
    assert os.path.isfile(script), f"Script exp01-final non trouve : {script}"


def test_run_exp01_final_importable():
    """Le script doit exposer une fonction run() appelable."""
    import importlib.util, os
    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'docs', 'investigations', 'P1-winrates-negatifs',
        'experiments', '01-H3-final', 'run_exp01_final.py'
    )
    spec   = importlib.util.spec_from_file_location("run_exp01_final", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, 'run'), "run_exp01_final.py doit exposer une fonction run()"
