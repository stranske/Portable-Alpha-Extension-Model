from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import pa_core.cli as cli_module
from pa_core.cli import Dependencies, main
from pa_core.config import ModelConfig, load_config
from pa_core.data import load_index_returns
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
            "--seed",
            "0",
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
            "--seed",
            "0",
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


def test_cli_sleeve_suggestion_applies_to_inputs(tmp_path, monkeypatch):
    cfg = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "analysis_mode": "single_with_sensitivity",
        "total_fund_capital": 300.0,
        "external_pa_capital": 100.0,
        "active_ext_capital": 50.0,
        "internal_pa_capital": 150.0,
        "risk_metrics": ["Return", "Risk", "ShortfallProb"],
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    idx_csv = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"
    out_file = tmp_path / "out.xlsx"

    suggestions = pd.DataFrame(
        [
            {
                "external_pa_capital": 200.0,
                "active_ext_capital": 50.0,
                "internal_pa_capital": 50.0,
            }
        ]
    )
    monkeypatch.setattr(
        "pa_core.sleeve_suggestor.suggest_sleeve_sizes",
        lambda *_args, **_kwargs: suggestions,
    )
    monkeypatch.setattr(
        "pa_core.sim.sensitivity.one_factor_deltas",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )

    captured: dict[str, object] = {}

    def _export_to_excel(inputs_dict, _summary, _raw_returns_dict, **_kwargs):
        captured["inputs_dict"] = inputs_dict

    deps = Dependencies(
        build_from_config=lambda _cfg: object(),
        export_to_excel=_export_to_excel,
        draw_financing_series=lambda *_args, **_kwargs: (
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
        ),
        draw_joint_returns=lambda *_args, **_kwargs: (
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
        ),
        build_cov_matrix=lambda *_args, **_kwargs: np.zeros((4, 4)),
        simulate_agents=lambda *_args, **_kwargs: {"Base": np.zeros((1, 1))},
    )

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
            "--sensitivity",
        ],
        deps=deps,
    )

    inputs = captured.get("inputs_dict")
    assert isinstance(inputs, dict)
    assert inputs["external_pa_capital"] == 200.0
    assert inputs["active_ext_capital"] == 50.0
    assert inputs["internal_pa_capital"] == 50.0


def test_suggest_sleeve_sizes_total_constraints(monkeypatch):
    cfg = load_config("examples/scenarios/test_params.yml")
    cfg = cfg.model_copy(update={"N_SIMULATIONS": 2, "N_MONTHS": 2})
    idx_series = pd.Series([0.0] * cfg.N_MONTHS)

    base = np.array([[0.0, 0.0], [0.0, 0.0]])
    ext = np.array([[0.1, -0.1], [0.1, -0.1]])
    act = np.array([[0.05, -0.05], [0.05, -0.05]])
    intr = np.array([[0.2, -0.2], [0.2, -0.2]])
    total = ext + act + intr
    returns = {
        "Base": base,
        "ExternalPA": ext,
        "ActiveExt": act,
        "InternalPA": intr,
        "Total": total,
    }
    summary = summary_table(returns, benchmark="Base")

    class DummyOrchestrator:
        def __init__(self, cfg, idx_series):
            self.cfg = cfg
            self.idx_series = idx_series

        def run(self, seed=None):
            return returns, summary

    monkeypatch.setattr("pa_core.sleeve_suggestor.SimulatorOrchestrator", DummyOrchestrator)

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

    monkeypatch.setattr("pa_core.sleeve_suggestor.SimulatorOrchestrator", DummyOrchestrator)

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

    monkeypatch.setattr("pa_core.sleeve_suggestor.SimulatorOrchestrator", DummyOrchestrator)

    df = suggest_sleeve_sizes(
        cfg,
        idx_series,
        max_te=1.0,
        max_breach=1.0,
        max_cvar=1.0,
        step=1.0,
    )

    assert df.empty


def test_sleeve_suggestor_matches_cli_summary(tmp_path, monkeypatch):
    cfg_data = {
        "N_SIMULATIONS": 4,
        "N_MONTHS": 3,
        "analysis_mode": "single_with_sensitivity",
        "total_fund_capital": 1000.0,
        "external_pa_capital": 500.0,
        "active_ext_capital": 250.0,
        "internal_pa_capital": 250.0,
        "risk_metrics": ["ShortfallProb"],
    }
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg_data))
    idx_path = tmp_path / "index.csv"
    idx_path.write_text("Return\n0.01\n0.02\n-0.01\n0.03\n")

    cfg = load_config(cfg_path)
    idx_series = load_index_returns(idx_path)

    suggestions = suggest_sleeve_sizes(
        cfg,
        idx_series,
        max_te=1.0,
        max_breach=1.0,
        max_cvar=1.0,
        step=0.25,
        min_external=500.0,
        max_external=500.0,
        min_active=250.0,
        max_active=250.0,
        min_internal=250.0,
        max_internal=250.0,
        seed=123,
    )
    assert len(suggestions) == 1

    captured: dict[str, pd.DataFrame] = {}
    original_summary = cli_module.create_enhanced_summary

    def _capture_summary(returns_map, *, benchmark=None):
        summary = original_summary(returns_map, benchmark=benchmark)
        captured.setdefault("summary", summary)
        return summary

    monkeypatch.setattr(cli_module, "create_enhanced_summary", _capture_summary)

    deps = Dependencies(export_to_excel=lambda *_args, **_kwargs: None)
    main(
        [
            "--config",
            str(cfg_path),
            "--index",
            str(idx_path),
            "--output",
            str(tmp_path / "out.xlsx"),
            "--seed",
            "123",
            "--sensitivity",
        ],
        deps=deps,
    )

    summary = captured.get("summary")
    assert isinstance(summary, pd.DataFrame)
    row = suggestions.iloc[0]
    for agent in ["ExternalPA", "ActiveExt", "InternalPA"]:
        sub = summary[summary["Agent"] == agent]
        assert not sub.empty
        summary_row = sub.iloc[0]
        np.testing.assert_allclose(
            float(summary_row["TE"]),
            float(row[f"{agent}_TE"]),
        )
        np.testing.assert_allclose(
            float(summary_row["BreachProb"]),
            float(row[f"{agent}_BreachProb"]),
        )
        np.testing.assert_allclose(
            float(summary_row["CVaR"]),
            float(row[f"{agent}_CVaR"]),
        )
