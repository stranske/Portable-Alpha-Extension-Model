import pandas as pd
import pytest
from pa_core import sensitivity
def test_one_factor_deltas():
    base = pd.DataFrame({"Sharpe": [1.0]})
    scenarios = {
        "mu": pd.DataFrame({"Sharpe": [1.2]}),
        "sigma": pd.DataFrame({"Sharpe": [0.9]}),
    }
    deltas = sensitivity.one_factor_deltas(base, scenarios)
    assert pytest.approx(deltas["mu"], rel=1e-6) == 0.2
    assert pytest.approx(deltas["sigma"], rel=1e-6) == -0.1
