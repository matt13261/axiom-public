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


# =============================================================================
# TEST G.fix.1 — _fusionner_noeuds additionne regrets + strategies + visites
# =============================================================================

def test_fusionner_noeuds_additionne_regrets_et_strategies():
    """_fusionner_noeuds doit additionner regrets_cumules, strategie_somme, nb_visites."""
    import train_cloud
    from ai.mccfr import NoeudCFR

    n1 = NoeudCFR(nb_actions=2)
    n1.regrets_cumules = [1.0, 2.0]
    n1.strategie_somme = [0.3, 0.7]
    n1.nb_visites = 10

    n2 = NoeudCFR(nb_actions=2)
    n2.regrets_cumules = [3.0, 4.0]
    n2.strategie_somme = [0.5, 0.5]
    n2.nb_visites = 20

    fusionne = train_cloud._fusionner_noeuds([{'cle1': n1}, {'cle1': n2}])

    assert fusionne['cle1'].regrets_cumules == [4.0, 6.0]
    assert fusionne['cle1'].strategie_somme == [0.8, 1.2]
    assert fusionne['cle1'].nb_visites == 30


# =============================================================================
# TEST G.fix.2 — main() avec --batch-size crée checkpoints intermédiaires
# =============================================================================

def test_train_cloud_creates_batch_checkpoints(tmp_path, monkeypatch):
    """main() avec iterations=20, batch-size=10 doit créer 2 fichiers _batch_N.pkl."""
    import train_cloud

    output = tmp_path / 'blueprint_test.pkl'
    monkeypatch.setattr(sys, 'argv', [
        'train_cloud.py',
        '--iterations', '20',
        '--batch-size', '10',
        '--output', str(output),
        '--workers', '1',
    ])

    train_cloud.main()

    assert (tmp_path / 'blueprint_test_batch_1.pkl').exists()
    assert (tmp_path / 'blueprint_test_batch_2.pkl').exists()
    assert output.exists()


# =============================================================================
# TEST G.fix.3 — main() avec --workers 4 spawn 4 processes par batch
# =============================================================================

def test_train_cloud_spawns_workers_processes(tmp_path, monkeypatch):
    """main() avec --workers 4 doit spawner exactement 4 mp.Process par batch."""
    import multiprocessing as mp
    import train_cloud

    spawn_count = {'n': 0}
    real_process = mp.Process

    class FakeProcess:
        def __init__(self, target=None, args=(), daemon=None, **kw):
            spawn_count['n'] += 1
            self._target = target
            self._args = args
        def start(self):
            self._target(*self._args)
        def join(self):
            pass
        def is_alive(self):
            return False

    monkeypatch.setattr(train_cloud.mp, 'Process', FakeProcess)

    output = tmp_path / 'blueprint_mp.pkl'
    monkeypatch.setattr(sys, 'argv', [
        'train_cloud.py',
        '--iterations', '8',
        '--batch-size', '8',
        '--output', str(output),
        '--workers', '4',
    ])

    train_cloud.main()

    assert spawn_count['n'] == 4, f"Attendu 4 processes, vu {spawn_count['n']}"


# =============================================================================
# TEST G.fix.4 — workers doivent fournir le training (pas inline dans main)
# =============================================================================

def test_train_cloud_workers_provide_training(tmp_path, monkeypatch):
    """
    Si les workers ne font rien (FakeProcess no-op), le blueprint produit
    doit être VIDE. Cela prouve que le training se fait dans les workers,
    pas inline dans main().
    """
    import multiprocessing as mp
    import pickle
    import train_cloud

    class NoopProcess:
        def __init__(self, target=None, args=(), daemon=None, **kw):
            pass
        def start(self):
            pass
        def join(self):
            pass
        def is_alive(self):
            return False

    monkeypatch.setattr(train_cloud.mp, 'Process', NoopProcess)

    output = tmp_path / 'blueprint_workers.pkl'
    monkeypatch.setattr(sys, 'argv', [
        'train_cloud.py',
        '--iterations', '20',
        '--batch-size', '20',
        '--output', str(output),
        '--workers', '2',
    ])

    train_cloud.main()

    with open(output, 'rb') as f:
        data = pickle.load(f)
    assert len(data) == 0, (
        f"Blueprint contient {len(data)} infosets — main() entraîne inline "
        f"au lieu de déléguer aux workers"
    )
