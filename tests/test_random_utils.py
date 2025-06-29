import numpy as np
from pa_core.random import spawn_rngs

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

