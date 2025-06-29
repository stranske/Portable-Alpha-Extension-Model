from hypothesis import given, strategies as st
import numpy as np
from pa_core.simulations import simulate_financing

@given(
    T=st.integers(min_value=1, max_value=24),
    n_scenarios=st.integers(min_value=1, max_value=10),
)
def test_simulate_financing_shapes(T, n_scenarios):
    out = simulate_financing(T, 0.0, 0.01, 0.0, 1.0, n_scenarios=n_scenarios)
    expected_shape = (T,) if n_scenarios == 1 else (n_scenarios, T)
    assert out.shape == expected_shape
    assert np.all(np.isfinite(out))
