import subprocess
import sys
from pathlib import Path


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content)


def test_validate_cli_ok(tmp_path: Path) -> None:
    yaml_path = tmp_path / "scen.yaml"
    _write_yaml(
        yaml_path,
        """
index:
  id: IDX
  mu: 0.1
  sigma: 0.2
assets:
  - id: A
    mu: 0.05
    sigma: 0.1
correlations:
  - pair: [IDX, A]
    rho: 0.1
portfolios:
  - id: p1
    weights: {A: 1.0}
""",
    )
    result = subprocess.run(
        [sys.executable, "-m", "pa_core.validate", str(yaml_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "OK"


def test_validate_cli_failure(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    _write_yaml(
        yaml_path,
        """
index:
  id: IDX
  mu: 0.1
  sigma: 0.2
assets: []
correlations: []
portfolios:
  - id: p1
    weights: {A: 1.0}
""",
    )
    result = subprocess.run(
        [sys.executable, "-m", "pa_core.validate", str(yaml_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unknown assets" in result.stdout
