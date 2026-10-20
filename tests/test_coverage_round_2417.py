"""Focused regression coverage for the ranked #2417 coverage targets."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pa_core import facade, pa, sweep, units
from pa_core.calibration import (
    CalibrationResult,
    CorrelationEstimate,
    EstimateCI,
    SeriesEstimate,
    build_calibration_report,
    infer_series_ids,
)
from pa_core.config import ModelConfig, SweepConfig, SweepParameter, get_field_mappings
from pa_core.data.loaders import (
    FrequencyValidationError,
    infer_index_frequency,
    load_index_returns,
    load_parameters,
    resample_to_monthly,
    validate_frequency,
)
from pa_core.llm import result_explain
from pa_core.manifest import ManifestWriter
from pa_core.sim.internal_pa_financing import resolve_internal_pa_financing_series


def test_result_explain_tolerates_telemetry_failure_and_missing_metric_delta(monkeypatch) -> None:
    monkeypatch.setattr(
        result_explain, "record_fleet_event", lambda *_: (_ for _ in ()).throw(RuntimeError())
    )

    text, trace_url, _payload = result_explain.explain_results_details(pd.DataFrame({"value": [1]}))

    assert "LLM configuration is required" in text
    assert trace_url is None
    assert result_explain._extract_metric_delta({}) is None
    assert (
        result_explain._extract_metric_delta({"stress_delta_summary": {"summary": {"x": {}}}})
        is None
    )
    assert result_explain._extract_metric_delta(
        {"stress_delta_summary": {"summary": {"a": {"mean": 0.1}, "b": {"mean": -0.4}}}}
    ) == pytest.approx(-0.4)
    assert result_explain._manifest_cli_output_ref({"cli_args": {"output": ""}}) is None
    assert (
        result_explain._manifest_cli_output_ref({"cli_args": {"output": "output.xlsx"}})
        == "output.xlsx"
    )
    assert result_explain._with_disclaimer("") == result_explain.EXPLAIN_RESULTS_DISCLAIMER
    already = result_explain.EXPLAIN_RESULTS_DISCLAIMER
    assert result_explain._with_disclaimer(already) == already
    assert result_explain._to_json_safe((pd.Timestamp("2020-01-01"), pd.NA)) == [
        "2020-01-01T00:00:00",
        None,
    ]
    # float NaN is returned as-is because the float branch precedes pd.isna.
    out_nan = result_explain._to_json_safe(float("nan"))
    assert isinstance(out_nan, float) and np.isnan(out_nan)
    with pytest.raises(TypeError, match="DataFrame"):
        result_explain.explain_results_details([{"value": 1}])  # type: ignore[arg-type]


def test_pa_mapping_helpers_distinguish_mapping_and_top_level_defaults(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.yml"
    mapping.write_text("calibration:\n  corr_target: constant\n")
    top_level = tmp_path / "top_level.yml"
    top_level.write_text("corr_target: identity\nmean_shrinkage: 0.1\n")
    defaults = tmp_path / "sweep.yml"
    defaults.write_text("index_file: index.csv\noutput_path: Sweep.xlsx\n")
    scalar = tmp_path / "scalar.yml"
    scalar.write_text("- not-a-mapping\n")
    csv_path = tmp_path / "params.csv"
    yaml_out = tmp_path / "params.yml"
    csv_path.write_text(
        "Parameter,Value\n"
        "Number of simulations,10\n"
        "In-House annual return (%),12\n"
        "Risk metrics,Return;Risk\n"
    )

    assert pa._load_calibration_overrides(mapping) == {"corr_target": "constant"}
    assert pa._load_calibration_overrides(top_level) == {
        "corr_target": "identity",
        "mean_shrinkage": 0.1,
    }
    assert pa._load_calibration_overrides(scalar) == {}
    assert pa._load_sweep_io_defaults(defaults) == ("index.csv", "Sweep.xlsx")
    assert pa._load_sweep_io_defaults(scalar) == (None, None)
    assert pa._coerce_calibration_setting(None, allowed={"identity"}, label="target") is None
    assert (
        pa._coerce_calibration_setting("identity", allowed={"identity"}, label="target")
        == "identity"
    )
    with pytest.raises(ValueError, match="target must be one of"):
        pa._coerce_calibration_setting("constant", allowed={"identity"}, label="target")

    pa._convert_csv_to_yaml(str(csv_path), str(yaml_out))
    converted = yaml_out.read_text()
    assert "N_SIMULATIONS: 10" in converted
    assert "mu_H:" in converted or "risk_metrics:" in converted
    # Friendly %-labelled values are scaled into decimal form.
    assert "0.12" in converted
    assert get_field_mappings()["Number of simulations"] == "N_SIMULATIONS"


def test_facade_normalizes_scalars_and_applies_all_run_overrides() -> None:
    class Config:
        def __init__(self, data: dict[str, object]) -> None:
            self.data = data

        def model_dump(self) -> dict[str, object]:
            return dict(self.data)

        @classmethod
        def model_validate(cls, data: dict[str, object]) -> "Config":
            return cls(data)

    original = Config({"return_distribution": "normal"})
    updated = facade.apply_run_options(
        original,
        facade.RunOptions(
            return_distribution="student_t",
            return_t_df=7.0,
            return_copula="t",
            covariance_shrinkage="ledoit_wolf",
            vol_regime="two_state",
            vol_regime_window=12,
            analysis_mode="capital",
        ),
        {"mu_H": 0.01},
    )

    assert updated.data == {
        "return_distribution": "student_t",
        "return_t_df": 7.0,
        "return_copula": "t",
        "covariance_shrinkage": "ledoit_wolf",
        "vol_regime": "two_state",
        "vol_regime_window": 12,
        "analysis_mode": "capital",
        "mu_H": 0.01,
    }
    assert facade.apply_run_options(original, None, None) is original
    assert facade._to_builtin_scalar({"nan": np.nan, "values": np.array([np.int64(2)])}) == {
        "nan": None,
        "values": [2],
    }
    assert facade._to_builtin_scalar(float("inf")) is None
    assert facade._to_builtin_scalar((np.float64(1.5),)) == [1.5]
    assert facade._records_to_builtin([{"x": np.int64(3)}]) == [{"x": 3}]

    inputs: dict[str, object] = {
        "_agent_semantics_df": pd.DataFrame({"Agent": ["A"], "Weight": [np.float64(1.0)]})
    }
    facade._serialize_agent_semantics_input(inputs)
    assert inputs["_agent_semantics_df"] == [{"Agent": "A", "Weight": 1.0}]


def test_units_cover_annual_round_trips_and_invalid_units() -> None:
    series = pd.Series([0.01, 0.02])

    assert np.isclose(
        units.convert_mean(0.01, from_unit="monthly", to_unit="annual", method="geometric"),
        1.01**12 - 1,
    )
    assert np.isclose(
        units.convert_mean(0.12, from_unit="annual", to_unit="monthly", method="simple"),
        0.01,
    )
    assert units.convert_mean(0.05, from_unit="monthly", to_unit="monthly") == 0.05
    assert np.isclose(
        units.convert_volatility(0.02, from_unit="monthly", to_unit="annual"),
        0.02 * np.sqrt(12),
    )
    assert np.isclose(
        units.convert_volatility(0.12, from_unit="annual", to_unit="monthly"),
        0.12 / np.sqrt(12),
    )
    assert np.allclose(
        units.convert_covariance(np.eye(2), from_unit="annual", to_unit="monthly"),
        np.eye(2) / 12,
    )
    assert np.allclose(
        units.convert_covariance(np.eye(2), from_unit="monthly", to_unit="monthly"),
        np.eye(2),
    )
    assert np.allclose(
        units.convert_return_series(
            series, from_unit="monthly", to_unit="annual", method="geometric"
        ),
        (1.0 + series) ** 12 - 1.0,
    )
    assert np.allclose(
        units.convert_return_series(series, from_unit="annual", to_unit="monthly", method="simple"),
        series / 12.0,
    )
    annual_idx = units.normalize_index_series(series, "annual")
    assert np.allclose(annual_idx, series / 12.0)
    assert units.normalize_index_series(series, "monthly").equals(pd.Series(series))
    assert units.get_config_unit() == "monthly"
    assert units.get_index_series_unit() == "monthly"
    assert units.format_unit_label("annual") == "annualised"
    assert units.format_unit_label("monthly") == "monthly"
    assert units.get_summary_table_unit(periods_per_year=1) == "monthly"
    assert units.get_summary_table_unit() == "annual"
    assert units.get_threshold_unit()["breach_threshold"] == "monthly"
    with pytest.raises(ValueError, match="unit must be"):
        units.convert_mean(0.1, from_unit="weekly", to_unit="monthly")  # type: ignore[arg-type]


def test_internal_financing_validates_inputs_and_generates_per_path_draws() -> None:
    for kwargs, message in [
        ({"n_months": 0, "n_sim": 1}, "n_months"),
        ({"n_months": 1, "n_sim": 0}, "n_sim"),
        ({"n_months": 1, "n_sim": 1, "financing_mode": "bad"}, "financing_mode"),
    ]:
        with pytest.raises(ValueError, match=message):
            resolve_internal_pa_financing_series(**kwargs)

    with pytest.raises(ValueError, match="must equal n_months"):
        resolve_internal_pa_financing_series(n_months=2, n_sim=1, series=[0.01])

    series_matrix = resolve_internal_pa_financing_series(n_months=2, n_sim=3, series=[-0.01, 0.02])
    assert series_matrix.shape == (3, 2)
    assert np.allclose(series_matrix[0], [-0.01, 0.02])

    index_matrix = resolve_internal_pa_financing_series(n_months=2, n_sim=2, index="SPX")
    assert index_matrix.shape == (2, 2)
    assert np.allclose(index_matrix[0], index_matrix[1])

    draws = resolve_internal_pa_financing_series(
        n_months=3,
        n_sim=2,
        sigma_month=0.01,
        financing_mode="per_path",
        rng=np.random.default_rng(4),
    )
    assert draws.shape == (2, 3)
    assert not np.array_equal(draws[0], draws[1])

    broadcast = resolve_internal_pa_financing_series(
        n_months=2,
        n_sim=3,
        mean_month=0.001,
        sigma_month=0.01,
        financing_mode="broadcast",
        rng=np.random.default_rng(5),
    )
    assert broadcast.shape == (3, 2)
    assert np.allclose(broadcast[0], broadcast[1])

    constant = resolve_internal_pa_financing_series(n_months=2, n_sim=2, mean_month=-0.002)
    assert np.allclose(constant, -0.002)


def test_manifest_falls_back_when_git_revision_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    def unavailable(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr("pa_core.manifest.subprocess.check_output", unavailable)
    config_path = tmp_path / "config.yml"
    config_path.write_text("N_SIMULATIONS: 2\n")
    data_file = tmp_path / "index.csv"
    data_file.write_text("Monthly_TR\n0.01\n")

    output = tmp_path / "manifest.json"
    with pytest.warns(UserWarning, match="seed"):
        ManifestWriter(output).write(
            config_path=config_path,
            config_snapshot=None,
            config_snapshot_bytes=None,
            data_files=[data_file],
            seed=None,
            cli_args={"output": "out.xlsx"},
        )

    text = output.read_text()
    assert '"git_commit": "unknown"' in text
    assert '"seed": null' in text
    assert str(data_file) in text or data_file.name in text


def test_loader_frequency_and_parameter_error_paths(tmp_path: Path) -> None:
    assert infer_index_frequency(pd.Series(["not-a-date", "2020-01-01"])) == "irregular"
    assert infer_index_frequency(pd.DatetimeIndex(["2020-01-01"])) == "unknown"
    daily = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
    assert infer_index_frequency(daily) == "daily"
    monthly = pd.DatetimeIndex(["2020-01-31", "2020-02-29", "2020-03-31"])
    assert infer_index_frequency(monthly) == "monthly"
    with pytest.raises(ValueError, match="DatetimeIndex"):
        resample_to_monthly(pd.Series([0.01]))

    series = pd.Series([0.01, 0.02], index=pd.DatetimeIndex(["2020-01-31", "2020-02-29"]))
    series.attrs["frequency"] = "daily"
    with pytest.raises(FrequencyValidationError, match="Expected monthly"):
        validate_frequency(series, expected="monthly", strict=True)
    unknown = pd.Series([0.01])
    with pytest.warns(UserWarning, match="Could not detect"):
        validate_frequency(unknown, expected="monthly", strict=True)

    with pytest.raises(FileNotFoundError, match="Parameter file not found"):
        load_parameters(tmp_path / "missing.csv", {})

    bad_params = tmp_path / "empty.csv"
    bad_params.write_text("")
    with pytest.raises(ValueError, match="Failed to read parameter CSV"):
        load_parameters(bad_params, {"Number of simulations": "N_SIMULATIONS"})

    with pytest.raises(FileNotFoundError, match="Index returns file not found"):
        load_index_returns(tmp_path / "missing_index.csv")

    wrong_col = tmp_path / "index.csv"
    wrong_col.write_text("Return\n0.01\n")
    with pytest.raises(ValueError, match="Expected index returns column"):
        load_index_returns(wrong_col)


def test_sweep_helpers_cover_grid_random_and_invalid_inputs(monkeypatch) -> None:
    params = {
        "a": SimpleNamespace(values=[1.0, 2.0], min=None, max=None, step=None),
        "b": SimpleNamespace(values=None, min=0.0, max=1.0, step=1.0),
    }
    grid = SimpleNamespace(parameters=params)
    assert list(sweep._iter_sweep_grid(grid)) == [
        {"a": 1.0, "b": 0.0},
        {"a": 1.0, "b": 1.0},
        {"a": 2.0, "b": 0.0},
        {"a": 2.0, "b": 1.0},
    ]
    assert sweep._sample_sweep_value(params["a"], np.random.default_rng(0)) in {1.0, 2.0}
    assert (
        sweep._sample_sweep_value(
            SimpleNamespace(values=None, min=0.0, max=1.0, step=None),
            np.random.default_rng(1),
        )
        >= 0.0
    )
    with pytest.raises(ValueError, match="min.*max"):
        sweep._sample_sweep_value(
            SimpleNamespace(values=None, min=None, max=None, step=1.0), np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="min.*max"):
        sweep._sample_sweep_value(
            SimpleNamespace(values=None, min=None, max=None, step=None), np.random.default_rng(0)
        )
    monkeypatch.setattr(sweep, "_HAS_TQDM", False)
    assert sweep.progress_bar([1, 2]) == [1, 2]
    assert sweep._count_range(0.0, 1.0, 0.5) == 3

    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        sweep=SweepConfig(
            method="random",
            samples=2,
            seed=3,
            parameters={"mu_H": SweepParameter(values=[0.01, 0.02])},
        ),
    )
    assert sweep._estimate_total_combinations(cfg) == 2
    assert sweep._override_keys_for_config(cfg) == {"mu_H"}
    combos = list(sweep.generate_parameter_combinations(cfg))
    assert len(combos) == 2
    assert all(set(combo) == {"mu_H"} for combo in combos)

    capital_cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        analysis_mode="capital",
        max_external_combined_pct=0.1,
        external_step_size_pct=0.1,
    )
    assert sweep._estimate_total_combinations(capital_cfg) == 3
    assert "external_pa_capital" in sweep._override_keys_for_config(capital_cfg)


def test_calibration_report_and_series_id_validation() -> None:
    stats = SeriesEstimate(0.01, 0.02, EstimateCI(0.0, 0.02), None, 2)
    result = CalibrationResult(
        index_id="IDX",
        series={"IDX": stats},
        correlations=[CorrelationEstimate(("IDX", "H"), 0.2, EstimateCI(0.1, 0.3), 2)],
        corr_target="identity",
        mean_shrinkage=0.2,
        corr_shrinkage=0.2,
        confidence_level=0.95,
        annualize=True,
    )

    report = build_calibration_report(result, alpha_map={"H": "H"}, min_obs=3)
    assert report["series"]["IDX"]["below_min_obs"] is True  # type: ignore[index]
    assert report["correlations"][0]["rho_ci"] == [0.1, 0.3]  # type: ignore[index]
    assert report["alpha_map"] == {"H": "H"}
    bare = build_calibration_report(result)
    assert "below_min_obs" not in bare["series"]["IDX"]  # type: ignore[index]
    assert "alpha_map" not in bare

    assert infer_series_ids(["IDX", "H", "E", "M"], index_id=None, alpha_ids=None) == (
        "IDX",
        ("H", "E", "M"),
    )
    assert infer_series_ids(["A", "B", "C", "D"], index_id="A", alpha_ids=["B", "C", "D"]) == (
        "A",
        ("B", "C", "D"),
    )
    with pytest.raises(ValueError, match="exactly three"):
        infer_series_ids(["IDX", "H"], index_id="IDX", alpha_ids=None)
    with pytest.raises(ValueError, match="no columns"):
        infer_series_ids([], index_id=None, alpha_ids=None)
    with pytest.raises(ValueError, match="exactly three"):
        infer_series_ids(["IDX"], index_id="IDX", alpha_ids=["H", "E"])
