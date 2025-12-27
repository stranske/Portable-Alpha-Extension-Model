from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pa_core.cli import main
from pa_core.config import load_config
from pa_core.sleeve_suggestor import suggest_sleeve_sizes

yaml: Any = pytest.importorskip("yaml")


def test_suggest_sleeve_sizes_returns_feasible():
    cfg = load_config("examples/scenarios/test_params.yml")
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
    assert {
        "external_pa_capital",
        "active_ext_capital",
        "internal_pa_capital",
    }.issubset(df.columns)


def test_suggest_sleeve_sizes_respects_bounds():
    cfg = load_config("examples/scenarios/test_params.yml")
    cfg = cfg.model_copy(update={"N_SIMULATIONS": 50})
    idx_series = pd.Series([0.0] * cfg.N_MONTHS)
    max_te = 0.02
    max_breach = 0.5
    max_cvar = 0.05
    df = suggest_sleeve_sizes(
        cfg,
        idx_series,
        max_te=max_te,
        max_breach=max_breach,
        max_cvar=max_cvar,
        step=0.5,
        seed=1,
    )
    assert (df.filter(regex="_TE").fillna(0) <= max_te).all().all()
    assert (df.filter(regex="_BreachProb").fillna(0) <= max_breach).all().all()
    assert (df.filter(regex="_CVaR").fillna(0).abs() <= max_cvar).all().all()


def test_cli_sleeve_suggestion(tmp_path, monkeypatch):
    cfg = {"N_SIMULATIONS": 10, "N_MONTHS": 1}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    idx_csv = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"
    out_file = tmp_path / "out.xlsx"
    monkeypatch.setattr("builtins.input", lambda _: "0")
    main(
        [
            "--config",
            str(cfg_path),
            "--index",
            str(idx_csv),
            "--output",
            str(out_file),
            "--suggest-sleeves",
            "--max-te",
            "0.02",
            "--max-breach",
            "0.5",
            "--max-cvar",
            "0.05",
            "--sleeve-step",
            "0.5",
        ]
    )
    assert out_file.exists()
