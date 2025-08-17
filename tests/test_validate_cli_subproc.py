import subprocess
import sys
from pathlib import Path


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content)


def _run_validate(args: list[str]) -> subprocess.CompletedProcess[str]:
    pkg_root = Path(__file__).resolve().parents[1] / "pa_core"
    script = (
        "import runpy,sys,types;"
        "pkg=types.ModuleType('pa_core');"
        "pkg.__path__=[sys.argv[1]];"
        "sys.modules['pa_core']=pkg;"
        "sys.argv=['pa_core.validate']+sys.argv[2:];"
        "runpy.run_module('pa_core.validate', run_name='__main__')"
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(pkg_root), *args],
        capture_output=True,
        text=True,
    )


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
    result = _run_validate([str(yaml_path)])
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
    result = _run_validate([str(yaml_path)])
    assert result.returncode != 0
    assert "unknown assets" in result.stdout


def test_validate_cli_config(tmp_path: Path) -> None:
    yaml_path = tmp_path / "conf.yml"
    _write_yaml(
        yaml_path,
        """
N_SIMULATIONS: 1
N_MONTHS: 1
mu_H: 0.04
sigma_H: 0.01
""",
    )
    result = _run_validate(["--schema", "config", str(yaml_path)])
    assert result.returncode == 0
    assert result.stdout.strip() == "OK"
