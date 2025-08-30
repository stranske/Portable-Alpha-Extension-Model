import pandas as pd

from pa_core.config import load_config
from pa_core.sweep import run_parameter_sweep_cached, sweep_results_to_dataframe


def test_cached_sweep_deterministic():
    cfg = load_config("config/alpha_shares_mode_template.yml")
    idx = pd.Series([0.01, 0.02, -0.01, 0.0])
    res1 = run_parameter_sweep_cached(cfg, idx, seed=123)
    res2 = run_parameter_sweep_cached(cfg, idx, seed=123)
    # cache should return same object
    assert res1 is res2

    df = sweep_results_to_dataframe(res1)
    # expect multiple scenarios and standard metrics
    assert df.shape[0] > 1
    for col in ["AnnReturn", "AnnVol", "TE"]:
        assert col in df.columns
