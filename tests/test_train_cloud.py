"""
Tests TDD pour scripts/cloud_train_p6/train_cloud.py (Étape G.1).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts' / 'cloud_train_p6'))


# =============================================================================
# TEST G.1a — train_cloud est importable et expose main()
# =============================================================================

def test_train_cloud_importable():
    """train_cloud.py doit être importable et exposer main()."""
    import train_cloud
    assert hasattr(train_cloud, 'main'), "train_cloud doit exposer main()"
    assert callable(train_cloud.main)


# =============================================================================
# TEST G.1b — main() avec args valides produit un fichier .pkl
# =============================================================================

def test_train_cloud_produces_output(tmp_path, monkeypatch):
    """main() avec --iterations 10 doit créer le fichier de sortie."""
    import pytest
    import train_cloud

    output = str(tmp_path / 'blueprint_test.pkl')
    monkeypatch.setattr(sys, 'argv', [
        'train_cloud.py',
        '--iterations', '10',
        '--output', output,
        '--checkpoint-every', '0',
    ])

    train_cloud.main()

    assert Path(output).exists(), f"Le fichier {output} doit exister après training"
    assert Path(output).stat().st_size > 0, "Le fichier ne doit pas être vide"


# =============================================================================
# TEST G.1c — main() quitte avec code 1 si centroïdes absents
# =============================================================================

def test_train_cloud_exits_if_no_centroides(tmp_path, monkeypatch):
    """main() doit sys.exit(1) si les centroïdes V2 sont absents."""
    import os
    import pytest
    import train_cloud

    output = str(tmp_path / 'blueprint_test.pkl')
    monkeypatch.setattr(sys, 'argv', [
        'train_cloud.py',
        '--iterations', '5',
        '--output', output,
        '--checkpoint-every', '0',
    ])
    monkeypatch.setattr(os.path, 'exists', lambda p: False)

    with pytest.raises(SystemExit) as exc:
        train_cloud.main()

    assert exc.value.code == 1, f"Code de sortie attendu 1, obtenu {exc.value.code}"
