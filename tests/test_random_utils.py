import numpy as np

from pa_core.random import derive_agent_substream_ids, spawn_agent_rngs, spawn_rngs


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


def test_spawn_agent_rngs_order_independent():
    names_a = ["A", "B", "C"]
    names_b = ["C", "A", "B"]
    rngs_a = spawn_agent_rngs(123, names_a)
    rngs_b = spawn_agent_rngs(123, names_b)
    for name in names_a:
        assert np.allclose(rngs_a[name].normal(size=5), rngs_b[name].normal(size=5))


def test_spawn_agent_rngs_adding_sleeve_stable():
    base_names = ["A", "B"]
    extended_names = ["B", "C", "A"]
    rngs_base = spawn_agent_rngs(321, base_names)
    rngs_ext = spawn_agent_rngs(321, extended_names)
    for name in base_names:
        assert np.allclose(rngs_base[name].normal(size=5), rngs_ext[name].normal(size=5))


def test_spawn_agent_rngs_legacy_order_dependent():
    names_a = ["A", "B"]
    names_b = ["B", "A"]
    rngs_a = spawn_agent_rngs(99, names_a, legacy_order=True)
    rngs_b = spawn_agent_rngs(99, names_b, legacy_order=True)
    assert not np.allclose(rngs_a["A"].normal(size=8), rngs_b["A"].normal(size=8))


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


def test_spawn_rngs_invalid_n_error():
    try:
        spawn_rngs(0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-positive n")


def test_seed_none_substream_ids_do_not_collide(monkeypatch):
    def fake_base_entropy(_seed):
        return 0

    monkeypatch.setattr("pa_core.random._base_entropy", fake_base_entropy)
    names = ["A"]
    ids_none = derive_agent_substream_ids(None, names)
    ids_seed = derive_agent_substream_ids(0, names)
    assert ids_none["A"] != ids_seed["A"]
