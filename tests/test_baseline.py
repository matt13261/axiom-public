"""Tests TDD pour le script baseline P1."""
import os


def test_baseline_script_exists():
    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'docs', 'investigations', 'P1-winrates-negatifs',
        'experiments', '00-baseline', 'run_baseline.py'
    )
    assert os.path.isfile(script), f"Baseline script not found: {script}"
