from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import pa_core.validate as validate

yaml: Any = pytest.importorskip("yaml")


def test_validate_cli_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [{"id": "A", "mu": 0.05, "sigma": 0.1}],
        "correlations": [{"pair": ["IDX", "A"], "rho": 0.1}],
        "portfolios": [{"id": "p1", "weights": {"A": 1.0}}],
    }
    path = tmp_path / "scen.yaml"
    path.write_text(yaml.safe_dump(data))
    validate.main([str(path)])
    captured = capsys.readouterr()
    assert captured.out.strip() == "OK"
    assert captured.err == ""


def test_validate_cli_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [{"id": "A", "mu": 0.05, "sigma": 0.1}],
        "correlations": [],
        "portfolios": [{"id": "p1", "weights": {"A": 1.0}}],
    }
    path = tmp_path / "scen.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(SystemExit) as exc:
        validate.main([str(path)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "missing correlations" in captured.out


def test_validate_cli_config_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "financing_mode": "broadcast",
        "mu_H": 0.04,
        "sigma_H": 0.01,
    }
    path = tmp_path / "conf.yml"
    path.write_text(yaml.safe_dump(data))
    validate.main(["--schema", "config", str(path)])
    captured = capsys.readouterr()
    assert captured.out.strip() == "OK"
    assert captured.err == ""


def test_validate_cli_config_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = {"N_SIMULATIONS": 1, "financing_mode": "broadcast"}
    path = tmp_path / "conf.yml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(SystemExit) as exc:
        validate.main(["--schema", "config", str(path)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "N_MONTHS" in captured.out
