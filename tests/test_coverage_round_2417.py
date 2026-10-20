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
from pa_core.data.loaders import infer_index_frequency, load_parameters, resample_to_monthly
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
    assert result_explain._manifest_cli_output_ref({"cli_args": {"output": ""}}) is None
    assert (
        result_explain._manifest_cli_output_ref({"cli_args": {"output": "output.xlsx"}})
        == "output.xlsx"
    )


def test_pa_mapping_helpers_distinguish_mapping_and_top_level_defaults(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.yml"
    mapping.write_text("calibration:\n  corr_target: constant\n")
    defaults = tmp_path / "sweep.yml"
    defaults.write_text("index_file: index.csv\noutput_path: Sweep.xlsx\n")
    scalar = tmp_path / "scalar.yml"
    scalar.write_text("- not-a-mapping\n")

    assert pa._load_calibration_overrides(mapping) == {"corr_target": "constant"}
    assert pa._load_calibration_overrides(scalar) == {}
    assert pa._load_sweep_io_defaults(defaults) == ("index.csv", "Sweep.xlsx")
    assert pa._coerce_calibration_setting(None, allowed={"identity"}, label="target") is None
    with pytest.raises(ValueError, match="target must be one of"):
        pa._coerce_calibration_setting("constant", allowed={"identity"}, label="target")


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
        facade.RunOptions(return_distribution="student_t", vol_regime="two_state"),
        {"return_t_df": 7.0},
    )

    assert updated.data == {
        "return_distribution": "student_t",
        "vol_regime": "two_state",
        "return_t_df": 7.0,
    }
    assert facade._to_builtin_scalar({"nan": np.nan, "values": np.array([np.int64(2)])}) == {
        "nan": None,
        "values": [2],
    }


def test_units_cover_annual_round_trips_and_invalid_units() -> None:
    series = pd.Series([0.01, 0.02])

    assert (
        units.convert_mean(0.01, from_unit="monthly", to_unit="annual", method="geometric") > 0.12
    )
    assert units.convert_volatility(0.02, from_unit="monthly", to_unit="annual") > 0.02
    assert np.allclose(
        units.convert_covariance(np.eye(2), from_unit="annual", to_unit="monthly"),
        np.eye(2) / 12,
    )
    assert (
        units.convert_return_series(
            series, from_unit="monthly", to_unit="annual", method="geometric"
        ).iloc[0]
        > 0.12
    )
    assert units.get_config_unit() == "monthly"
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

    draws = resolve_internal_pa_financing_series(
        n_months=3,
        n_sim=2,
        sigma_month=0.01,
        financing_mode="per_path",
        rng=np.random.default_rng(4),
    )
    assert draws.shape == (2, 3)
    assert not np.array_equal(draws[0], draws[1])


def test_manifest_falls_back_when_git_revision_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    def unavailable(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr("pa_core.manifest.subprocess.check_output", unavailable)
    output = tmp_path / "manifest.json"
    ManifestWriter(output).write(
        config_path=tmp_path / "config.yml",
        config_snapshot="N_SIMULATIONS: 1\n",
        config_snapshot_bytes=None,
        data_files=[],
        seed=1,
        cli_args={},
    )

    assert '"git_commit": "unknown"' in output.read_text()


def test_loader_frequency_and_parameter_error_paths(tmp_path: Path) -> None:
    assert infer_index_frequency(pd.Series(["not-a-date", "2020-01-01"])) == "irregular"
    assert infer_index_frequency(pd.DatetimeIndex(["2020-01-01"])) == "unknown"
    with pytest.raises(ValueError, match="DatetimeIndex"):
        resample_to_monthly(pd.Series([0.01]))

    with pytest.raises(FileNotFoundError, match="Parameter file not found"):
        load_parameters(tmp_path / "missing.csv", {})


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
    with pytest.raises(ValueError, match="min.*max"):
        sweep._sample_sweep_value(
            SimpleNamespace(values=None, min=None, max=None, step=1.0), np.random.default_rng(0)
        )
    monkeypatch.setattr(sweep, "_HAS_TQDM", False)
    assert sweep.progress_bar([1, 2]) == [1, 2]


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
    with pytest.raises(ValueError, match="exactly three"):
        infer_series_ids(["IDX", "H"], index_id="IDX", alpha_ids=None)
