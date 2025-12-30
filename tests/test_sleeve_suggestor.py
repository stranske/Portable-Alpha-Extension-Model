from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pa_core.cli import main
from pa_core.config import ModelConfig, load_config
from pa_core.sim.metrics import summary_table
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


def test_cli_sleeve_suggestion_auto_apply(tmp_path, monkeypatch):
    cfg = {"N_SIMULATIONS": 10, "N_MONTHS": 1}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    idx_csv = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"
    out_file = tmp_path / "out.xlsx"

    def _fail_input(*_args):
        raise AssertionError("input should not be called")

    monkeypatch.setattr("builtins.input", _fail_input)
    main(
        [
            "--config",
            str(cfg_path),
            "--index",
            str(idx_csv),
            "--output",
            str(out_file),
            "--suggest-sleeves",
            "--suggest-apply-index",
            "0",
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


def test_suggest_sleeve_sizes_total_constraints(monkeypatch):
    cfg = load_config("examples/scenarios/test_params.yml")
    cfg = cfg.model_copy(update={"N_SIMULATIONS": 2, "N_MONTHS": 2})
    idx_series = pd.Series([0.0] * cfg.N_MONTHS)

    base = np.array([[0.0, 0.0], [0.0, 0.0]])
    ext = np.array([[0.1, -0.1], [0.1, -0.1]])
    act = np.array([[0.05, -0.05], [0.05, -0.05]])
    intr = np.array([[0.2, -0.2], [0.2, -0.2]])
    returns = {"Base": base, "ExternalPA": ext, "ActiveExt": act, "InternalPA": intr}
    summary = summary_table(returns, benchmark="Base")

    class DummyOrchestrator:
        def __init__(self, cfg, idx_series):
            self.cfg = cfg
            self.idx_series = idx_series

        def run(self, seed=None):
            return returns, summary

    monkeypatch.setattr(
        "pa_core.sleeve_suggestor.SimulatorOrchestrator", DummyOrchestrator
    )

    df = suggest_sleeve_sizes(
        cfg,
        idx_series,
        max_te=0.0,
        max_breach=1.0,
        max_cvar=1.0,
        step=1.0,
        constraint_scope="total",
    )
    assert df.empty

    df = suggest_sleeve_sizes(
        cfg,
        idx_series,
        max_te=1.0,
        max_breach=1.0,
        max_cvar=1.0,
        step=1.0,
        constraint_scope="total",
    )
    assert not df.empty
    assert {"Total_TE", "Total_BreachProb", "Total_CVaR"}.issubset(df.columns)
    assert (df["Total_TE"] <= 1.0).all()
    assert (df["Total_BreachProb"] <= 1.0).all()
    assert (df["Total_CVaR"].abs() <= 1.0).all()


def test_suggest_sleeve_sizes_caps_max_evals(monkeypatch):
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1)
    idx_series = pd.Series([0.0])

    returns = {
        "Base": np.zeros((1, 1)),
        "ExternalPA": np.zeros((1, 1)),
        "ActiveExt": np.zeros((1, 1)),
        "InternalPA": np.zeros((1, 1)),
    }
    summary = summary_table(returns, benchmark="Base")

    class DummyOrchestrator:
        def __init__(self, cfg, idx_series):
            self.cfg = cfg
            self.idx_series = idx_series

        def run(self, seed=None):
            return returns, summary

    monkeypatch.setattr(
        "pa_core.sleeve_suggestor.SimulatorOrchestrator", DummyOrchestrator
    )

    df = suggest_sleeve_sizes(
        cfg,
        idx_series,
        max_te=1.0,
        max_breach=1.0,
        max_cvar=1.0,
        step=0.5,
        max_evals=2,
    )

    assert len(df) == 2


def test_suggest_sleeve_sizes_skips_invalid_metrics(monkeypatch):
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1)
    idx_series = pd.Series([0.0])

    returns = {
        "Base": np.zeros((1, 1)),
        "ExternalPA": np.zeros((1, 1)),
        "ActiveExt": np.zeros((1, 1)),
        "InternalPA": np.zeros((1, 1)),
    }
    summary = summary_table(returns, benchmark="Base")
    summary.loc[summary["Agent"] == "ExternalPA", "TE"] = np.nan

    class DummyOrchestrator:
        def __init__(self, cfg, idx_series):
            self.cfg = cfg
            self.idx_series = idx_series

        def run(self, seed=None):
            return returns, summary

    monkeypatch.setattr(
        "pa_core.sleeve_suggestor.SimulatorOrchestrator", DummyOrchestrator
    )

    df = suggest_sleeve_sizes(
        cfg,
        idx_series,
        max_te=1.0,
        max_breach=1.0,
        max_cvar=1.0,
        step=1.0,
    )

    assert df.empty
