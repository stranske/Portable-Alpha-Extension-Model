import json
from pathlib import Path
import sys
import types

import pytest
import yaml

sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))
pptx_mod = types.ModuleType("pptx")
pptx_util = types.ModuleType("pptx.util")
pptx_mod.Presentation = object
pptx_util.Inches = lambda x: x
pptx_mod.util = pptx_util
sys.modules.setdefault("pptx", pptx_mod)
sys.modules.setdefault("pptx.util", pptx_util)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa_core.cli import main


def _write_cfg(tmp_path, backend=None):
    cfg = {"N_SIMULATIONS": 1, "N_MONTHS": 1}
    if backend:
        cfg["backend"] = backend
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    idx_csv = Path(__file__).resolve().parents[1] / "sp500tr_fred_divyield.csv"
    return cfg_path, idx_csv


def test_backend_cli_numpy(tmp_path):
    cfg_path, idx_csv = _write_cfg(tmp_path)
    out_file = tmp_path / "out.xlsx"
    main([
        "--config",
        str(cfg_path),
        "--index",
        str(idx_csv),
        "--output",
        str(out_file),
        "--backend",
        "numpy",
    ])
    assert out_file.exists()


def test_backend_cli_cupy_missing(tmp_path):
    cfg_path, idx_csv = _write_cfg(tmp_path)
    with pytest.raises(ImportError, match="CuPy backend requested"):
        main([
            "--config",
            str(cfg_path),
            "--index",
            str(idx_csv),
            "--backend",
            "cupy",
        ])


def test_backend_config_cupy_missing(tmp_path):
    cfg_path, idx_csv = _write_cfg(tmp_path, backend="cupy")
    with pytest.raises(ImportError, match="CuPy backend requested"):
        main([
            "--config",
            str(cfg_path),
            "--index",
            str(idx_csv),
        ])
