#!/usr/bin/env python3
"""Integration tests for backend selection between CLI and backend module."""

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

# Mock streamlit and pptx early to avoid import issues
sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))
pptx_mod = types.ModuleType("pptx")
pptx_util = types.ModuleType("pptx.util")
pptx_mod.Presentation = object  # type: ignore[attr-defined]
pptx_util.Inches = lambda x: x  # type: ignore[attr-defined]
pptx_mod.util = pptx_util  # type: ignore[attr-defined]
sys.modules.setdefault("pptx", pptx_mod)
sys.modules.setdefault("pptx.util", pptx_util)

from pa_core.backend import get_backend  # noqa: E402
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


class TestBackendCLIIntegration:
    """Integration tests for CLI backend selection addressing issue #600."""

    def _create_test_config(self, tmp_path, backend=None):
        """Create a test config file."""
        cfg = {
            "N_SIMULATIONS": 1,
            "N_MONTHS": 1,
            "financing_mode": "broadcast",
            "risk_metrics": ["Return", "Risk", "terminal_ShortfallProb"],
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
            **({"backend": backend} if backend else {}),
        }

        cfg_path = tmp_path / "test_cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        # Get the actual index CSV from repository
        repo_root = Path(__file__).resolve().parents[1]
        idx_csv = repo_root / "data" / "sp500tr_fred_divyield.csv"

        return cfg_path, idx_csv

    def test_backend_numpy_cli_flag(self, tmp_path, capsys):
        """Test --backend numpy works and is echoed at start."""
        cfg_path, idx_csv = self._create_test_config(tmp_path)
        out_file = tmp_path / "numpy_test.xlsx"
        manifest_file = tmp_path / "manifest.json"

        # Run CLI with numpy backend
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

        # Verify output file created
        assert out_file.exists()

        # Verify backend echo message
        captured = capsys.readouterr()
        assert "[BACKEND] Using backend: numpy" in captured.out

        # Verify backend is recorded in manifest
        if manifest_file.exists():
            manifest_data = json.loads(manifest_file.read_text())
            assert manifest_data.get("backend") == "numpy"
            assert manifest_data.get("cli_args", {}).get("backend") == "numpy"

    def test_backend_from_config_file(self, tmp_path, capsys):
        """Test backend selection from config file when no CLI flag."""
        cfg_path, idx_csv = self._create_test_config(tmp_path, backend="numpy")
        out_file = tmp_path / "config_backend_test.xlsx"

        # Run CLI without backend flag (should use config)
        main(
            [
                "--config",
                str(cfg_path),
                "--index",
                str(idx_csv),
                "--output",
                str(out_file),
            ]
        )

        # Verify output file created
        assert out_file.exists()

        # Verify backend echo message shows config backend
        captured = capsys.readouterr()
        assert "[BACKEND] Using backend: numpy" in captured.out

    def test_backend_default_fallback(self, tmp_path, capsys):
        """Test default backend when neither CLI nor config specify."""
        # Create config without backend field
        cfg_path, idx_csv = self._create_test_config(tmp_path)
        out_file = tmp_path / "default_backend_test.xlsx"

        # Run without backend flag
        main(
            [
                "--config",
                str(cfg_path),
                "--index",
                str(idx_csv),
                "--output",
                str(out_file),
            ]
        )

        # Should default to numpy
        captured = capsys.readouterr()
        assert "[BACKEND] Using backend: numpy" in captured.out
        assert out_file.exists()

    def test_backend_in_manifest_matches_actual(self, tmp_path):
        """Test that backend recorded in manifest matches actual backend used."""
        cfg_path, idx_csv = self._create_test_config(tmp_path)
        out_file = tmp_path / "manifest_backend_test.xlsx"
        manifest_file = out_file.with_name("manifest.json")

        # Run with specific backend
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

        # Verify actual backend is set correctly
        assert get_backend() == "numpy"

        # Verify manifest contains correct backend
        if manifest_file.exists():
            manifest_data = json.loads(manifest_file.read_text())
            assert manifest_data.get("backend") == "numpy"


if __name__ == "__main__":
    pytest.main([__file__])
