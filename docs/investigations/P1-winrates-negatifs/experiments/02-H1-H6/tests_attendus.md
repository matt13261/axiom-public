# Exp 02 — Tests TDD à écrire (dans l'ordre)

_Fichier cible : `tests/test_heuristique_agressive.py`_
_Un seul test à la fois. Chaque test doit être RED avant l'implémentation._

---

## Test 1 — RED avant fix : BTN face raise + main faible → fold majoritaire

```python
def test_heuristique_btn_face_raise_main_faible():
    """
    BTN (pos=0) face à un raise avec main faible (bucket 0-2) :
    FOLD doit être > 50% de la stratégie normalisée.
    Doit FAIL avant le fix (actuellement FOLD=10% normalisé).
    """
    from ai.agent import AgentAXIOM
    from engine.game_state import EtatJeu
    from engine.player import Joueur

    agent = AgentAXIOM.__new__(AgentAXIOM)
    agent._abs_cartes = <mock retournant bucket=1>
    
    etat = <EtatJeu avec mise_courante > 0>
    joueur = <Joueur position=0, mise_tour=0>  # BTN, face à raise
    
    vec = agent._heuristique(etat, joueur)
    assert vec[0] > 0.50, f"BTN face raise main faible : FOLD={vec[0]:.2f} doit être >0.50"
```

**Pourquoi RED ?** Actuellement branche `else` → FOLD normalisé = 10/160 = 6.3%.

---

## Test 2 — BTN face raise + main forte → raise majoritaire

```python
def test_heuristique_btn_face_raise_main_forte():
    """
    BTN face à raise avec main forte (bucket >=5) :
    RAISE total (vec[3:]) doit être > 50%.
    """
    # bucket=6, face_a_raise=True, position=0
    # Post-fix : RAISE_FORT=65% → RAISE total ≈ 65%
    ...
    raise_total = vec[3:].sum()
    assert raise_total > 0.50, f"BTN face raise main forte : RAISE={raise_total:.2f} doit être >0.50"
```

**Pourquoi RED ?** Actuellement RAISE normalisé = 30/160 = 18.8%.

---

## Test 3 — BTN sans raise + main forte → raise/bet majoritaire (pas check)

```python
def test_heuristique_btn_sans_raise_main_forte():
    """
    BTN sans raise adverse avec main forte (bucket >=5) :
    RAISE total > CHECK (opportunité de value bet).
    """
    # face_a_raise=False, bucket=6, position=0
    # Post-fix : RAISE_FORT=50% > CHECK=20%
    ...
    assert vec[3:].sum() > vec[1], "BTN sans raise main forte : RAISE doit dépasser CHECK"
```

**Pourquoi RED ?** Actuellement CHECK normalisé = 37.5% > RAISE 18.8%.

---

## Test 4 — Invariants SB/BB non dégradés

```python
def test_heuristique_sb_bb_inchanges():
    """
    SB et BB face à raise doivent conserver leur comportement actuel.
    H1 ne touche que la branche BTN/else — SB/BB ne doivent pas changer.
    """
    # SB bucket >=5 face raise : FOLD=20%, CALL=40%, RAISE_FORT=40%
    # BB bucket 0-2 face raise : FOLD=60%, CALL=35%, RAISE=5%
    ...
```

**Pourquoi RED ?** Si l'implémentation casse SB/BB par erreur.

---

## Ordre d'exécution TDD Guard

1. Écrire `test_heuristique_btn_face_raise_main_faible` → RED
2. Implémenter le cas `face_a_raise + bucket 0-2` dans la branche else → GREEN
3. Vérifier que pytest complet reste vert
4. Écrire `test_heuristique_btn_face_raise_main_forte` → RED
5. Implémenter le cas `face_a_raise + bucket >=5` → GREEN
6. Etc.

---

## Difficultés anticipées

### Mocking du bucket

`_heuristique()` appelle `self._abs_cartes.bucket(joueur.cartes, etat.board)`.
Pour tester un bucket spécifique sans lancer le MC, il faut mocker `_abs_cartes`.

Approche : créer un `MockAbstraction` minimal :
```python
class MockAbstraction:
    def __init__(self, bucket_fixe):
        self._bucket = bucket_fixe
    def bucket(self, cartes, board):
        return self._bucket
```

### Constructing minimal EtatJeu and Joueur

`_heuristique()` accède à :
- `joueur.position` (int : 0=BTN, 1=SB, 2=BB)
- `etat.mise_courante` (float)
- `joueur.mise_tour` (float)
- `joueur.cartes` (list — pour bucket, mocké)
- `etat.board` (list — pour bucket, mocké)

Pas besoin d'un jeu complet — des objets simples avec ces attributs suffisent.
