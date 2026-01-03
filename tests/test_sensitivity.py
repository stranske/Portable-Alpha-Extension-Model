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


def test_one_factor_deltas_with_ann_return():
    """Test sensitivity analysis with terminal_AnnReturn column."""
    base = pd.DataFrame({"terminal_AnnReturn": [8.5]})
    scenarios = {
        "mu_H_+5%": pd.DataFrame({"terminal_AnnReturn": [9.0]}),
        "mu_H_-5%": pd.DataFrame({"terminal_AnnReturn": [8.0]}),
        "sigma_H_+5%": pd.DataFrame({"terminal_AnnReturn": [8.3]}),
        "sigma_H_-5%": pd.DataFrame({"terminal_AnnReturn": [8.7]}),
    }
    deltas = sensitivity.one_factor_deltas(base, scenarios, value="terminal_AnnReturn")

    # Check that deltas are computed correctly
    assert pytest.approx(deltas["mu_H_+5%"], rel=1e-6) == 0.5
    assert pytest.approx(deltas["mu_H_-5%"], rel=1e-6) == -0.5
    assert pytest.approx(deltas["sigma_H_+5%"], rel=1e-6) == -0.2
    assert pytest.approx(deltas["sigma_H_-5%"], rel=1e-6) == 0.2

    # Check that results are sorted by absolute magnitude
    abs_deltas = deltas.abs().values
    assert all(abs_deltas[i] >= abs_deltas[i + 1] for i in range(len(abs_deltas) - 1))


def test_one_factor_deltas_orders_ties():
    base = pd.DataFrame({"terminal_AnnReturn": [1.0]})
    scenarios = {
        "beta": pd.DataFrame({"terminal_AnnReturn": [1.1]}),
        "alpha": pd.DataFrame({"terminal_AnnReturn": [1.1]}),
        "gamma": pd.DataFrame({"terminal_AnnReturn": [1.05]}),
    }
    deltas = sensitivity.one_factor_deltas(base, scenarios, value="terminal_AnnReturn")
    assert list(deltas.index) == ["alpha", "beta", "gamma"]


def test_one_factor_deltas_missing_column():
    """Test error handling for missing columns."""
    base = pd.DataFrame({"terminal_AnnReturn": [8.5]})
    scenarios = {
        "mu_H_+5%": pd.DataFrame({"WrongColumn": [9.0]}),
    }

    with pytest.raises(KeyError, match="WrongColumn column missing"):
        sensitivity.one_factor_deltas(base, scenarios, value="WrongColumn")

    with pytest.raises(KeyError, match="terminal_AnnReturn column missing from scenario"):
        sensitivity.one_factor_deltas(base, scenarios, value="terminal_AnnReturn")
