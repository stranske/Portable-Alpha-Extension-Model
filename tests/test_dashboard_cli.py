from __future__ import annotations

from pathlib import Path

import dashboard.cli as dashboard_cli
import dashboard.utils as dashboard_utils
import pandas as pd


def test_dashboard_cli_runs_streamlit(monkeypatch):
    called = {}

    def _fake_run(cmd, check):
        called["cmd"] = cmd
        called["check"] = check

    monkeypatch.setattr(dashboard_cli.subprocess, "run", _fake_run)

    dashboard_cli.main()

    expected_app = Path(dashboard_cli.__file__).with_name("app.py")
    assert called["cmd"] == ["streamlit", "run", str(expected_app)]
    assert called["check"] is True


def test_build_pa_core_args_includes_seed():
    args = dashboard_cli.build_pa_core_args(
        "config.yaml",
        "index.csv",
        "output.xlsx",
        use_seed=True,
        seed_value=123,
    )

    assert args[:6] == [
        "--config",
        "config.yaml",
        "--index",
        "index.csv",
        "--output",
        "output.xlsx",
    ]
    assert "--seed" in args
    seed_idx = args.index("--seed") + 1
    assert args[seed_idx] == "123"


def test_build_pa_core_args_omits_seed_when_disabled():
    args = dashboard_cli.build_pa_core_args(
        "config.yaml",
        "index.csv",
        "output.xlsx",
        use_seed=False,
        seed_value=None,
    )

    assert "--seed" not in args


def test_run_sleeve_suggestions_forwards_seed(monkeypatch):
    captured = {}

    def _fake_suggest(cfg, idx_series, **kwargs):
        captured["seed"] = kwargs.get("seed")
        return pd.DataFrame()

    monkeypatch.setattr(dashboard_utils, "suggest_sleeve_sizes", _fake_suggest)

    dashboard_utils.run_sleeve_suggestions(
        object(),
        pd.Series([0.1, 0.2, 0.3]),
        max_te=0.01,
        max_breach=0.5,
        max_cvar=0.05,
        max_shortfall=0.1,
        step=0.25,
        max_evals=10,
        constraint_scope="portfolio",
        seed=77,
    )

    assert captured["seed"] == 77
