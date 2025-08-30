import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pa_core.config import load_config
from pa_core.sleeve_suggestor import suggest_sleeve_sizes


def test_suggest_sleeve_sizes_returns_feasible():
    cfg = load_config('test_params.yml')
    cfg = cfg.model_copy(update={"N_SIMULATIONS": 50})
    idx_series = pd.Series([0.0] * cfg.N_MONTHS)
    df = suggest_sleeve_sizes(
        cfg,
        idx_series,
        max_te=0.02,
        max_breach=0.5,
        max_cvar=0.05,
        step=0.5,
        seed=1,
    )
    assert not df.empty
    assert {"external_pa_capital", "active_ext_capital", "internal_pa_capital"}.issubset(df.columns)
