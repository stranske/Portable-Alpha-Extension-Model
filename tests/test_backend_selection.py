import sys
import types
from pathlib import Path

import pytest
import yaml

sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))
pptx_mod = types.ModuleType("pptx")
pptx_util = types.ModuleType("pptx.util")
pptx_mod.Presentation = object  # type: ignore[attr-defined]
pptx_util.Inches = lambda x: x  # type: ignore[attr-defined]
pptx_mod.util = pptx_util  # type: ignore[attr-defined]
sys.modules.setdefault("pptx", pptx_mod)
sys.modules.setdefault("pptx.util", pptx_util)

from pa_core.cli import main  # noqa: E402


@pytest.fixture(autouse=True)
def _fast_sweep(monkeypatch):
    def _stub_run_parameter_sweep(*_args, **_kwargs):
        return []

    def _stub_export_sweep_results(_results, filename="Sweep.xlsx", **_kwargs):
        Path(filename).write_text("")

    monkeypatch.setattr("pa_core.sweep.run_parameter_sweep", _stub_run_parameter_sweep)
    monkeypatch.setattr(
        "pa_core.reporting.sweep_excel.export_sweep_results",
        _stub_export_sweep_results,
    )


def _write_cfg(tmp_path, backend=None):
    cfg = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "in_house_return_min_pct": 2.0,
        "in_house_return_max_pct": 2.0,
        "in_house_return_step_pct": 1.0,
        "in_house_vol_min_pct": 1.0,
        "in_house_vol_max_pct": 1.0,
        "in_house_vol_step_pct": 1.0,
        "alpha_ext_return_min_pct": 1.0,
        "alpha_ext_return_max_pct": 1.0,
        "alpha_ext_return_step_pct": 1.0,
        "alpha_ext_vol_min_pct": 2.0,
        "alpha_ext_vol_max_pct": 2.0,
        "alpha_ext_vol_step_pct": 1.0,
    }
    if backend:
        cfg["backend"] = backend
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    idx_csv = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"
    return cfg_path, idx_csv


def test_backend_cli_numpy(tmp_path):
    cfg_path, idx_csv = _write_cfg(tmp_path)
    out_file = tmp_path / "out.xlsx"
    main(
        [
            "--config",
            str(cfg_path),
            "--index",
            str(idx_csv),
            "--output",
            str(out_file),
            "--backend",
            "numpy",
        ]
    )
    assert out_file.exists()


def test_backend_config_rejects_unknown(tmp_path):
    cfg_path, idx_csv = _write_cfg(tmp_path, backend="cupy")
    with pytest.raises(ValueError, match=r"backend must be .*numpy"):
        main(
            [
                "--config",
                str(cfg_path),
                "--index",
                str(idx_csv),
            ]
        )
