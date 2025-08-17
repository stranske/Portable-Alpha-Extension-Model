from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pa_core import validate


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
