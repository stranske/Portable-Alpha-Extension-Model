from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pa_core.pa import main
from pa_core.schema import load_scenario
from pa_core.config import load_config

yaml = pytest.importorskip("yaml")


def _write_returns_csv(path: Path, dates: pd.DatetimeIndex, data: dict[str, list[float]]) -> None:
    df = pd.DataFrame({"Date": dates, **data})
    df.to_csv(path, index=False)


def test_pa_calibrate_two_state_vol_regime(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    low = [0.01, 0.01, 0.01]
    high = [0.10, -0.10, 0.12]
    csv_path = tmp_path / "returns.csv"
    _write_returns_csv(csv_path, dates, {"IDX": low + high, "A": low + high})

    output_path = tmp_path / "asset_library.yaml"
    main(
        [
            "calibrate",
            "--input",
            str(csv_path),
            "--index-id",
            "IDX",
            "--output",
            str(output_path),
            "--min-obs",
            "1",
            "--vol-regime",
            "two_state",
            "--vol-regime-window",
            "3",
        ]
    )

    payload = yaml.safe_load(output_path.read_text())
    asset_a = next(row for row in payload["assets"] if row["id"] == "A")
    recent_sigma = pd.Series(high).std(ddof=1) * (12.0**0.5)
    assert asset_a["sigma"] == pytest.approx(float(recent_sigma))


def test_pa_calibrate_mapping_vol_regime(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    low = [0.01, 0.01, 0.01]
    high = [0.10, -0.10, 0.12]
    csv_path = tmp_path / "returns.csv"
    _write_returns_csv(csv_path, dates, {"IDX": low + high, "A": low + high})

    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(
        "\n".join(
            [
                "date_col: Date",
                "wide: true",
                "value_type: returns",
                "frequency: monthly",
                "min_obs: 1",
                "vol_regime: two_state",
                "vol_regime_window: 3",
            ]
        )
    )

    output_path = tmp_path / "asset_library.yaml"
    main(
        [
            "calibrate",
            "--input",
            str(csv_path),
            "--index-id",
            "IDX",
            "--output",
            str(output_path),
            "--mapping",
            str(mapping),
        ]
    )

    payload = yaml.safe_load(output_path.read_text())
    asset_a = next(row for row in payload["assets"] if row["id"] == "A")
    recent_sigma = pd.Series(high).std(ddof=1) * (12.0**0.5)
    assert asset_a["sigma"] == pytest.approx(float(recent_sigma))


def test_pa_calibrate_mapping_cov_shrinkage(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    df = pd.DataFrame(
        {
            "Date": dates,
            "IDX": [0.01, 0.02, 0.03],
            "A": [0.02, 0.03, ""],
        }
    )
    csv_path = tmp_path / "returns.csv"
    df.to_csv(csv_path, index=False)

    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(
        "\n".join(
            [
                "date_col: Date",
                "wide: true",
                "value_type: returns",
                "frequency: monthly",
                "min_obs: 1",
                "covariance_shrinkage: ledoit_wolf",
            ]
        )
    )

    output_path = tmp_path / "asset_library.yaml"
    main(
        [
            "calibrate",
            "--input",
            str(csv_path),
            "--index-id",
            "IDX",
            "--output",
            str(output_path),
            "--mapping",
            str(mapping),
        ]
    )

    payload = yaml.safe_load(output_path.read_text())
    asset_idx = next(row for row in payload["assets"] if row["id"] == "IDX")
    expected_sigma = pd.Series([0.01, 0.02]).std(ddof=0) * (12.0**0.5)
    assert asset_idx["sigma"] == pytest.approx(float(expected_sigma))


def test_pa_calibrate_returns_params(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-31", periods=8, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "IDX": [0.01, 0.02, 0.03, 0.04, 0.01, -0.01, 0.02, 0.03],
            "H": [0.02, 0.01, 0.03, 0.02, 0.01, 0.00, 0.02, 0.01],
            "E": [0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.02, 0.02],
            "M": [0.03, 0.02, 0.01, 0.00, 0.02, 0.03, 0.01, 0.00],
        }
    )
    csv_path = tmp_path / "managers.csv"
    df.to_csv(csv_path, index=False)

    output_path = tmp_path / "params.yml"
    scenario_path = tmp_path / "scenario.yml"
    main(
        [
            "calibrate",
            "--returns",
            str(csv_path),
            "--output",
            str(output_path),
            "--scenario-output",
            str(scenario_path),
        ]
    )

    payload = yaml.safe_load(output_path.read_text())
    assert payload["mu_H"] != 0.0
    assert payload["sigma_E"] > 0.0
    assert "calibration" in payload
    load_config(output_path)
    scenario = load_scenario(scenario_path)
    assert scenario.index.id == "IDX"
