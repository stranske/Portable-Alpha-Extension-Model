from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pa_core.pa import main

# ruff: noqa: E402


def test_pa_validate(tmp_path: Path) -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [{"id": "A", "mu": 0.05, "sigma": 0.1}],
        "correlations": [{"pair": ["IDX", "A"], "rho": 0.1}],
        "portfolios": [{"id": "p1", "weights": {"A": 1.0}}],
    }
    path = tmp_path / "scen.yaml"
    path.write_text(yaml.safe_dump(data))
    main(["validate", str(path)])


def test_pa_validate_fail(tmp_path: Path) -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [],
        "correlations": [],
        "portfolios": [{"id": "p1", "weights": {"IDX": 0.5}}],
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(SystemExit):
        main(["validate", str(path)])


def test_pa_validate_config(tmp_path: Path) -> None:
    data = {"N_SIMULATIONS": 1, "N_MONTHS": 1, "mu_H": 0.04, "sigma_H": 0.01}
    path = tmp_path / "conf.yml"
    path.write_text(yaml.safe_dump(data))
    main(["validate", "--schema", "config", str(path)])


def test_pa_validate_config_fail(tmp_path: Path) -> None:
    data = {"N_SIMULATIONS": 1}
    path = tmp_path / "conf.yml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(SystemExit):
        main(["validate", "--schema", "config", str(path)])
