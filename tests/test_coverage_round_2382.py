"""Focused coverage for ranked #2382 targets not finished in PR #2385.

Evidence: docs/evidence/issue-2382-closer-repair/
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pa_core.calibration import (
    CalibrationResult,
    SeriesEstimate,
    calibrate_returns,
)
from pa_core.config import ModelConfig, SweepConfig, SweepParameter
from pa_core.data.loaders import (
    infer_index_frequency,
    load_index_returns,
    load_parameters,
    resample_to_monthly,
    validate_frequency,
)
from pa_core.random import spawn_agent_rngs, spawn_rngs
from pa_core.sweep import (
    SweepRunner,
    _estimate_total_combinations,
    _override_keys_for_config,
    clear_sweep_cache,
    generate_parameter_combinations,
    progress_bar,
    run_parameter_sweep,
    run_parameter_sweep_cached,
)


def _minimal_broadcast_config(**overrides: object) -> ModelConfig:
    base = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "financing_mode": "broadcast",
    }
    base.update(overrides)
    return ModelConfig(**base)


def test_config_explicit_return_unit_and_regime_validation_branches() -> None:
    with pytest.raises(ValueError, match="provide only annual or monthly"):
        ModelConfig(
            N_SIMULATIONS=1,
            N_MONTHS=1,
            financing_mode="broadcast",
            mu_H_annual=0.05,
            mu_H_monthly=0.01,
        )
    with pytest.raises(ValueError, match="multiple keys provided"):
        ModelConfig(
            N_SIMULATIONS=1,
            N_MONTHS=1,
            financing_mode="broadcast",
            mu_H_annual=0.05,
            **{"In-House annual return (%)": 0.06},
        )
    with pytest.raises(ValueError, match="do not mix mu_H"):
        ModelConfig(
            N_SIMULATIONS=1,
            N_MONTHS=1,
            financing_mode="broadcast",
            mu_H=0.05,
            mu_H_annual=0.05,
        )
    assert ModelConfig.compile_agent_config([]) == []  # type: ignore[arg-type]
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        regimes=[{"name": "stress", "sigma_H": 0.24}],
        regime_transition=[[1.0]],
    )
    assert cfg.regimes[0].sigma_H == pytest.approx(0.24 / (12**0.5))
    with pytest.raises(ValueError, match="return_distribution must be one of"):
        ModelConfig(
            N_SIMULATIONS=1,
            N_MONTHS=1,
            financing_mode="broadcast",
            return_distribution_H="bad",
        )
    with pytest.raises(ValueError, match="return_copula must be one of"):
        ModelConfig(
            N_SIMULATIONS=1,
            N_MONTHS=1,
            financing_mode="broadcast",
            return_copula="bad",
        )
    with pytest.raises(ValueError, match="return_copula must be 'gaussian'"):
        ModelConfig(
            N_SIMULATIONS=1,
            N_MONTHS=1,
            financing_mode="broadcast",
            return_distribution="normal",
            return_copula="t",
        )
    with pytest.raises(ValueError, match="regime_transition requires regimes"):
        ModelConfig(
            N_SIMULATIONS=1,
            N_MONTHS=1,
            financing_mode="broadcast",
            regime_transition=[[1.0]],
        )
    with pytest.raises(ValueError, match="regime_start requires regimes"):
        ModelConfig(
            N_SIMULATIONS=1,
            N_MONTHS=1,
            financing_mode="broadcast",
            regime_start="calm",
        )


def test_loaders_cover_frequency_inference_and_csv_paths(tmp_path: Path) -> None:
    assert infer_index_frequency(42) == "unknown"  # type: ignore[arg-type]
    dup = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-02-01"])
    assert infer_index_frequency(dup) == "irregular"
    weekly = pd.DatetimeIndex(["2020-01-01", "2020-01-08", "2020-01-15"])
    assert infer_index_frequency(weekly) == "weekly"
    daily = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
    assert infer_index_frequency(daily) == "daily"
    quarterly = pd.DatetimeIndex(["2020-01-31", "2020-04-30", "2020-07-31"])
    assert infer_index_frequency(quarterly) == "quarterly"

    monthly = pd.Series(
        [0.01, 0.02],
        index=pd.DatetimeIndex(["2020-01-31", "2020-02-29"]),
    )
    monthly.attrs["frequency"] = "monthly"
    assert resample_to_monthly(monthly).equals(monthly)

    wide = tmp_path / "wide.csv"
    wide.write_text("N_SIMULATIONS,In-House annual return (%)\n10,12\n")
    with pytest.warns(DeprecationWarning):
        loaded = load_parameters(
            wide,
            {"N_SIMULATIONS": "N_SIMULATIONS", "In-House annual return (%)": "mu_H"},
        )
    assert loaded["N_SIMULATIONS"] == 10
    assert loaded["mu_H"] == pytest.approx(0.12)

    dated = tmp_path / "dated_index.csv"
    dated.write_text("Date,Monthly_TR\n2020-01-31,0.01\n2020-02-29,0.02\n2020-03-31,not-a-number\n")
    series = load_index_returns(dated)
    assert len(series) == 2
    assert series.attrs["data_quality"]["coerced_non_numeric"] == 1

    bad_date = tmp_path / "bad_date.csv"
    bad_date.write_text("Date,Monthly_TR\n2020-01-31,0.01\nnot-a-date,0.02\n")
    with pytest.raises(ValueError, match="Failed to parse dates"):
        load_index_returns(bad_date)

    fmt_only = tmp_path / "fmt_only.csv"
    fmt_only.write_text("Monthly_TR\n0.01\n")
    with pytest.raises(ValueError, match="Date format provided but no Date column"):
        load_index_returns(fmt_only, date_format="%Y-%m-%d")

    with pytest.warns(UserWarning, match="Index frequency mismatch"):
        validate_frequency(
            pd.Series([0.01, 0.02], index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"])),
            expected="monthly",
            strict=False,
        )


def test_sweep_analysis_modes_and_runner_index_normalization(monkeypatch) -> None:
    import pa_core.sweep as sweep_mod

    monkeypatch.setattr(sweep_mod, "_HAS_TQDM", True)
    monkeypatch.setattr(
        sweep_mod,
        "_tqdm",
        lambda iterable, **_: list(iterable),
        raising=False,
    )
    assert progress_bar([1, 2], total=2) == [1, 2]

    grid_cfg = _minimal_broadcast_config(
        sweep=SweepConfig(
            method="grid",
            parameters={
                "mu_H": SweepParameter(min=0.0, max=0.01, step=0.01),
            },
        )
    )
    assert _estimate_total_combinations(grid_cfg) == 2

    returns_cfg = _minimal_broadcast_config(
        analysis_mode="returns",
        in_house_return_min_pct=0.0,
        in_house_return_max_pct=0.0,
        in_house_return_step_pct=1.0,
        in_house_vol_min_pct=0.0,
        in_house_vol_max_pct=0.0,
        in_house_vol_step_pct=1.0,
        alpha_ext_return_min_pct=0.0,
        alpha_ext_return_max_pct=0.0,
        alpha_ext_return_step_pct=1.0,
        alpha_ext_vol_min_pct=0.0,
        alpha_ext_vol_max_pct=0.0,
        alpha_ext_vol_step_pct=1.0,
    )
    combos = list(generate_parameter_combinations(returns_cfg))
    assert combos == [
        {
            "mu_H": pytest.approx(0.0),
            "sigma_H": pytest.approx(0.0),
            "mu_E": pytest.approx(0.0),
            "sigma_E": pytest.approx(0.0),
        }
    ]
    assert _estimate_total_combinations(returns_cfg) == 1
    assert _override_keys_for_config(returns_cfg) == {"mu_H", "sigma_H", "mu_E", "sigma_E"}

    alpha_cfg = _minimal_broadcast_config(
        analysis_mode="alpha_shares",
        external_pa_alpha_min_pct=0.0,
        external_pa_alpha_max_pct=0.0,
        external_pa_alpha_step_pct=1.0,
        active_share_min_pct=0.0,
        active_share_max_pct=0.0,
        active_share_step_pct=1.0,
    )
    alpha_combo = next(generate_parameter_combinations(alpha_cfg))
    assert set(alpha_combo) == {"theta_extpa", "active_share"}
    assert _estimate_total_combinations(alpha_cfg) == 1

    vol_cfg = _minimal_broadcast_config(
        analysis_mode="vol_mult",
        sd_multiple_min=1.0,
        sd_multiple_max=1.0,
        sd_multiple_step=0.5,
    )
    vol_combo = next(generate_parameter_combinations(vol_cfg))
    assert set(vol_combo) == {"sigma_H", "sigma_E", "sigma_M"}
    assert _estimate_total_combinations(vol_cfg) == 1

    capital_cfg = _minimal_broadcast_config(
        analysis_mode="capital",
        max_external_combined_pct=0.0,
        external_step_size_pct=0.1,
        total_fund_capital=1000.0,
    )
    assert list(generate_parameter_combinations(capital_cfg)) == [
        {
            "external_pa_capital": 0.0,
            "active_ext_capital": 0.0,
            "internal_pa_capital": 1000.0,
        }
    ]

    bad_mode_cfg = _minimal_broadcast_config(analysis_mode="capital")
    bad_mode_cfg = bad_mode_cfg.model_copy(update={"analysis_mode": "unsupported"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported analysis mode"):
        list(generate_parameter_combinations(bad_mode_cfg))

    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["idx"] = args[1]
        return []

    monkeypatch.setattr("pa_core.sweep.run_parameter_sweep", fake_run)
    runner = SweepRunner(_minimal_broadcast_config(), pd.DataFrame([[0.01], [0.02]]))
    runner.run()
    assert isinstance(captured["idx"], pd.Series)


def test_calibration_validation_and_constant_target() -> None:
    df = pd.DataFrame(
        {"IDX": [0.01, 0.02], "H": [0.01, 0.02], "E": [0.01, 0.02], "M": [0.01, 0.02]}
    )
    with pytest.raises(ValueError, match="returns data is empty"):
        calibrate_returns(df.iloc[0:0], index_id="IDX")
    with pytest.raises(ValueError, match="index_id not found"):
        calibrate_returns(df, index_id="MISSING")
    with pytest.raises(ValueError, match="mean_shrinkage must be"):
        calibrate_returns(df, index_id="IDX", mean_shrinkage=1.5)
    with pytest.raises(ValueError, match="corr_shrinkage must be"):
        calibrate_returns(df, index_id="IDX", corr_shrinkage=-0.1)
    with pytest.raises(ValueError, match="corr_target must be"):
        calibrate_returns(df, index_id="IDX", corr_target="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="confidence_level must be"):
        calibrate_returns(df, index_id="IDX", confidence_level=1.0)

    varied = pd.DataFrame(
        {
            "IDX": [0.01, 0.02, 0.03, 0.04],
            "H": [0.02, 0.01, 0.03, 0.02],
            "E": [0.01, 0.02, 0.01, 0.02],
            "M": [0.03, 0.02, 0.01, 0.02],
        }
    )
    constant = calibrate_returns(varied, index_id="IDX", corr_target="constant", annualize=False)
    assert constant.corr_target == "constant"

    stats = SeriesEstimate(0.01, 0.02, None, None, 2)
    broken = CalibrationResult(
        index_id="IDX",
        series={"H": stats},
        correlations=[],
        corr_target="identity",
        mean_shrinkage=0.0,
        corr_shrinkage=0.0,
        confidence_level=0.95,
        annualize=False,
    )
    with pytest.raises(ValueError, match="index_id not found in series estimates"):
        broken.to_scenario()
    with pytest.raises(ValueError, match="alpha_map must include"):
        broken.to_model_config(alpha_map={"H": "H"})

    full = calibrate_returns(df, index_id="IDX", annualize=False)
    params = full.to_model_config(alpha_map={"H": "H", "E": "E", "M": "M"})
    assert "rho_H_E" in params

    with pytest.raises(ValueError, match="series 'MISSING' not found"):
        full.to_model_config(alpha_map={"H": "H", "E": "MISSING", "M": "M"})
    with pytest.raises(ValueError, match="missing correlation for pair"):
        CalibrationResult(
            index_id="IDX",
            series=full.series,
            correlations=[],
            corr_target="identity",
            mean_shrinkage=0.0,
            corr_shrinkage=0.0,
            confidence_level=0.95,
            annualize=False,
        ).to_model_config(alpha_map={"H": "H", "E": "E", "M": "M"})


def test_config_regime_and_correlation_repair_edges() -> None:
    with pytest.raises(ValueError, match="regimes must include at least one regime"):
        ModelConfig(
            N_SIMULATIONS=1,
            N_MONTHS=1,
            financing_mode="broadcast",
            regimes=[],
            regime_transition=[],
        )
    with pytest.raises(ValueError, match="regime names must be unique"):
        ModelConfig(
            N_SIMULATIONS=1,
            N_MONTHS=1,
            financing_mode="broadcast",
            regimes=[{"name": "a"}, {"name": "a"}],
            regime_transition=[[0.5, 0.5], [0.5, 0.5]],
        )
    with pytest.raises(ValueError, match="must sum to 1"):
        ModelConfig(
            N_SIMULATIONS=1,
            N_MONTHS=1,
            financing_mode="broadcast",
            regimes=[{"name": "a"}, {"name": "b"}],
            regime_transition=[[0.9, 0.1], [0.2, 0.5]],
        )
    with pytest.raises(ValueError, match="correlation_repair_max_abs_delta"):
        ModelConfig(
            N_SIMULATIONS=1,
            N_MONTHS=1,
            financing_mode="broadcast",
            correlation_repair_max_abs_delta=-0.1,
        )
    with pytest.raises(ValueError, match="return_t_df must be finite"):
        ModelConfig(
            N_SIMULATIONS=1,
            N_MONTHS=1,
            financing_mode="broadcast",
            return_distribution="student_t",
            return_t_df=1.0,
        )


def test_loaders_csv_variants_and_infer_freq_branches(tmp_path: Path, monkeypatch) -> None:
    params = tmp_path / "params.csv"
    params.write_text("Parameter,Value\nNumber of simulations,10\nRisk metrics,Return;Risk\n")
    with pytest.warns(DeprecationWarning):
        loaded = load_parameters(
            params,
            {"Number of simulations": "N_SIMULATIONS", "Risk metrics": "risk_metrics"},
        )
    assert loaded["risk_metrics"] == ["Return", "Risk"]

    empty_index = tmp_path / "empty.csv"
    empty_index.write_text("")
    with pytest.raises(ValueError, match="Failed to read index returns CSV"):
        load_index_returns(empty_index)

    monkeypatch.setattr(
        "pa_core.data.loaders.pd.infer_freq",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("infer failed")),
    )
    monthly_alias = pd.DatetimeIndex(["2020-01-31", "2020-02-29", "2020-03-31"])
    assert infer_index_frequency(monthly_alias) == "monthly"


def test_sweep_run_paths_cover_regimes_cache_and_vol_mult() -> None:
    idx = pd.Series([0.01, -0.02, 0.03, 0.0, 0.02, -0.01])
    cfg = ModelConfig(
        N_SIMULATIONS=2,
        N_MONTHS=3,
        financing_mode="broadcast",
        analysis_mode="vol_mult",
        sd_multiple_min=1.0,
        sd_multiple_max=1.0,
        sd_multiple_step=0.5,
    )
    rng_returns = spawn_rngs(7, 1)[0]
    fin_rngs = spawn_agent_rngs(7, ["internal", "external_pa", "active_ext"])
    clear_sweep_cache()
    progress_calls: list[tuple[int, int]] = []

    def progress(current: int, total: int) -> None:
        progress_calls.append((current, total))

    results = run_parameter_sweep_cached(cfg, idx, seed=7, progress=progress)
    assert len(results) == 1
    assert progress_calls

    regime_cfg = cfg.model_copy(
        update={
            "analysis_mode": "capital",
            "regimes": [{"name": "calm"}],
            "regime_transition": None,
        }
    )
    with pytest.raises(ValueError, match="regime_transition is required"):
        run_parameter_sweep(regime_cfg, idx, rng_returns, fin_rngs, seed=7)

    stepped_cfg = _minimal_broadcast_config(
        sweep=SweepConfig(
            method="random",
            samples=1,
            seed=1,
            parameters={"mu_H": SweepParameter(min=0.0, max=0.01, step=0.005)},
        )
    )
    combo = next(generate_parameter_combinations(stepped_cfg))
    assert "mu_H" in combo
