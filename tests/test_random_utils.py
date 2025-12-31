import numpy as np

from pa_core.random import spawn_agent_rngs, spawn_rngs


def test_spawn_rngs_reproducible():
    r1, r2 = spawn_rngs(123, 2)
    a1 = r1.normal(size=5)
    a2 = r2.normal(size=5)
    r1b, r2b = spawn_rngs(123, 2)
    assert np.allclose(a1, r1b.normal(size=5))
    assert np.allclose(a2, r2b.normal(size=5))


def test_spawn_rngs_independent():
    r1, r2 = spawn_rngs(42, 2)
    assert not np.allclose(r1.normal(size=3), r2.normal(size=3))


def test_spawn_rngs_none_seed():
    rngs = spawn_rngs(None, 3)
    assert len(rngs) == 3


def test_spawn_agent_rngs_reproducible():
    names = ["A", "B"]
    rngs = spawn_agent_rngs(42, names)
    vals_a = rngs["A"].normal(size=3)
    vals_b = rngs["B"].normal(size=3)
    rngs2 = spawn_agent_rngs(42, names)
    assert np.allclose(vals_a, rngs2["A"].normal(size=3))
    assert np.allclose(vals_b, rngs2["B"].normal(size=3))


def test_spawn_agent_rngs_error():
    try:
        spawn_agent_rngs(0, [])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty list")


def test_spawn_agent_rngs_duplicate_names_error():
    try:
        spawn_agent_rngs(0, ["A", "A"])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for duplicate names")
