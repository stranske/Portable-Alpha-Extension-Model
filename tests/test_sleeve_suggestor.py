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
        "risk_metrics": ["Return", "Risk", "terminal_ShortfallProb"],
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
        max_te=1.5,
        max_breach=1.0,
        max_cvar=1.0,
        step=1.0,
        constraint_scope="total",
    )
    assert not df.empty
    assert {"Total_monthly_TE", "Total_monthly_BreachProb", "Total_monthly_CVaR"}.issubset(
        df.columns
    )
    assert (df["Total_monthly_TE"] <= 1.5).all()
    assert (df["Total_monthly_BreachProb"] <= 1.0).all()
    assert (df["Total_monthly_CVaR"].abs() <= 1.0).all()


def test_suggest_sleeve_sizes_caps_max_evals(monkeypatch):
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1, financing_mode="broadcast")
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


def test_suggest_sleeve_sizes_reuses_cached_streams(monkeypatch):
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        total_fund_capital=100.0,
        external_pa_capital=50.0,
        active_ext_capital=25.0,
        internal_pa_capital=25.0,
    )
    idx_series = pd.Series([0.0])

    r = np.zeros((1, 1))
    streams = (r, r, r, r, r, r, r)
    calls = {"draw": 0}

    class DummyOrchestrator:
        def __init__(self, cfg, idx_series):
            self.cfg = cfg
            self.idx_series = idx_series

        def draw_streams(self, seed=None):
            calls["draw"] += 1
            return streams

        def run(self, seed=None):
            raise AssertionError("run should not be called when streams are cached")

    monkeypatch.setattr("pa_core.sleeve_suggestor.SimulatorOrchestrator", DummyOrchestrator)

    df = suggest_sleeve_sizes(
        cfg,
        idx_series,
        max_te=1.0,
        max_breach=1.0,
        max_cvar=1.0,
        step=0.5,
        max_evals=3,
    )

    assert calls["draw"] == 1
    assert not df.empty


def test_suggest_sleeve_sizes_skips_invalid_metrics(monkeypatch):
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1, financing_mode="broadcast")
    idx_series = pd.Series([0.0])

    returns = {
        "Base": np.zeros((1, 1)),
        "ExternalPA": np.zeros((1, 1)),
        "ActiveExt": np.zeros((1, 1)),
        "InternalPA": np.zeros((1, 1)),
    }
    summary = summary_table(returns, benchmark="Base")
    summary.loc[summary["Agent"] == "ExternalPA", "monthly_TE"] = np.nan

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
        "financing_mode": "broadcast",
        "analysis_mode": "single_with_sensitivity",
        "total_fund_capital": 1000.0,
        "external_pa_capital": 500.0,
        "active_ext_capital": 250.0,
        "internal_pa_capital": 250.0,
        "risk_metrics": ["terminal_ShortfallProb"],
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
            float(summary_row["monthly_TE"]),
            float(row[f"{agent}_monthly_TE"]),
        )
        np.testing.assert_allclose(
            float(summary_row["monthly_BreachProb"]),
            float(row[f"{agent}_monthly_BreachProb"]),
        )
        np.testing.assert_allclose(
            float(summary_row["monthly_CVaR"]),
            float(row[f"{agent}_monthly_CVaR"]),
        )


def _make_linear_summary(cfg: ModelConfig) -> pd.DataFrame:
    per_cap = {
        "ExternalPA": {
            "terminal_AnnReturn": 0.03,
            "terminal_ExcessReturn": 0.025,
            "monthly_TE": 0.0001,
            "monthly_BreachProb": 0.0002,
            "monthly_CVaR": -0.0003,
        },
        "ActiveExt": {
            "terminal_AnnReturn": 0.01,
            "terminal_ExcessReturn": 0.008,
            "monthly_TE": 0.0001,
            "monthly_BreachProb": 0.0002,
            "monthly_CVaR": -0.0003,
        },
        "InternalPA": {
            "terminal_AnnReturn": 0.02,
            "terminal_ExcessReturn": 0.015,
            "monthly_TE": 0.0001,
            "monthly_BreachProb": 0.0002,
            "monthly_CVaR": -0.0003,
        },
    }
    rows = []
    totals = {
        "terminal_AnnReturn": 0.0,
        "terminal_ExcessReturn": 0.0,
        "monthly_TE": 0.0,
        "monthly_BreachProb": 0.0,
        "monthly_CVaR": 0.0,
    }
    for agent, capital in (
        ("ExternalPA", cfg.external_pa_capital),
        ("ActiveExt", cfg.active_ext_capital),
        ("InternalPA", cfg.internal_pa_capital),
    ):
        metrics = {k: v * capital for k, v in per_cap[agent].items()}
        totals = {k: totals[k] + metrics[k] for k in totals}
        rows.append({"Agent": agent, **metrics})
    rows.append({"Agent": "Total", **totals})
    return pd.DataFrame(rows)


def test_suggest_sleeve_sizes_optimize_prefers_best(monkeypatch):
    _ = pytest.importorskip("scipy")
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        total_fund_capital=100.0,
        external_pa_capital=40.0,
        active_ext_capital=30.0,
        internal_pa_capital=30.0,
    )
    idx_series = pd.Series([0.0])

    class DummyOrchestrator:
        def __init__(self, cfg, idx_series):
            self.cfg = cfg
            self.idx_series = idx_series

        def run(self, seed=None):
            return {}, _make_linear_summary(self.cfg)

    monkeypatch.setattr("pa_core.sleeve_suggestor.SimulatorOrchestrator", DummyOrchestrator)

    df = suggest_sleeve_sizes(
        cfg,
        idx_series,
        max_te=0.05,
        max_breach=0.05,
        max_cvar=0.05,
        step=0.5,
        min_external=0.0,
        max_external=80.0,
        min_active=0.0,
        max_active=80.0,
        min_internal=0.0,
        max_internal=100.0,
        optimize=True,
        objective="total_return",
    )
    assert not df.empty
    row = df.iloc[0]
    assert row["optimizer_success"] is True
    assert row["constraints_satisfied"] is True
    assert row["external_pa_capital"] == pytest.approx(80.0)
    assert row["active_ext_capital"] == pytest.approx(0.0)
    assert row["internal_pa_capital"] == pytest.approx(20.0)

    rng = np.random.default_rng(0)
    random_scores = []
    for _ in range(25):
        ext = float(rng.uniform(0.0, 80.0))
        act = float(rng.uniform(0.0, 80.0))
        remaining = 100.0 - ext - act
        if remaining < 0:
            continue
        random_scores.append(0.03 * ext + 0.02 * remaining + 0.01 * act)
    assert row["objective_value"] >= max(random_scores)


def test_suggest_sleeve_sizes_optimize_missing_scipy_falls_back(monkeypatch):
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        total_fund_capital=100.0,
        external_pa_capital=40.0,
        active_ext_capital=30.0,
        internal_pa_capital=30.0,
    )
    idx_series = pd.Series([0.0])

    class DummyOrchestrator:
        def __init__(self, cfg, idx_series):
            self.cfg = cfg
            self.idx_series = idx_series

        def run(self, seed=None):
            return {}, _make_linear_summary(self.cfg)

    monkeypatch.setattr("pa_core.sleeve_suggestor.SimulatorOrchestrator", DummyOrchestrator)
    monkeypatch.setattr("pa_core.sleeve_suggestor._load_minimize", lambda: None)

    df = suggest_sleeve_sizes(
        cfg,
        idx_series,
        max_te=0.05,
        max_breach=0.05,
        max_cvar=0.05,
        step=0.5,
        optimize=True,
        objective="total_return",
    )
    assert not df.empty
    assert str(df.loc[0, "optimizer_status"]).startswith("grid_fallback")


def test_suggest_sleeve_sizes_infeasible_constraints_returns_status(monkeypatch):
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        total_fund_capital=100.0,
        external_pa_capital=40.0,
        active_ext_capital=30.0,
        internal_pa_capital=30.0,
    )
    idx_series = pd.Series([0.0])

    class DummyOrchestrator:
        def __init__(self, cfg, idx_series):
            self.cfg = cfg
            self.idx_series = idx_series

        def run(self, seed=None):
            return {}, _make_linear_summary(self.cfg)

    class DummyResult:
        def __init__(self, x):
            self.x = np.array(x, dtype=float)
            self.success = False
            self.message = "failed"

    def _fake_minimize(fun, x0, method=None, bounds=None, constraints=None, options=None):
        return DummyResult(x0)

    monkeypatch.setattr("pa_core.sleeve_suggestor.SimulatorOrchestrator", DummyOrchestrator)
    monkeypatch.setattr("pa_core.sleeve_suggestor._load_minimize", lambda: _fake_minimize)

    df = suggest_sleeve_sizes(
        cfg,
        idx_series,
        max_te=1e-6,
        max_breach=1e-6,
        max_cvar=1e-6,
        step=0.5,
        optimize=True,
        objective="total_return",
    )
    assert not df.empty
    assert not bool(df.loc[0, "constraints_satisfied"])
    assert str(df.loc[0, "optimizer_status"]).startswith("fallback_failed")
