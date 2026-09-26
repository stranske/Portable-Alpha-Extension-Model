from pathlib import Path

import pandas as pd
import yaml

from pa_core.cli import main


def test_sweep_mode_honors_png_flag(monkeypatch, tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "N_SIMULATIONS": 1,
                "N_MONTHS": 1,
                "financing_mode": "broadcast",
                "analysis_mode": "returns",
            }
        )
    )

    summary = pd.DataFrame(
        {
            "Agent": ["Base"],
            "terminal_AnnReturn": [0.06],
            "monthly_AnnVol": [0.11],
            "terminal_ShortfallProb": [0.08],
        }
    )
    monkeypatch.setattr(
        "pa_core.sweep.run_parameter_sweep",
        lambda *_args, **_kwargs: [{"summary": summary, "combination_id": 1}],
    )
    monkeypatch.setattr(
        "pa_core.reporting.sweep_excel.export_sweep_results",
        lambda _results, filename, **_kwargs: Path(filename).write_text("stub"),
    )
    monkeypatch.setattr(
        "pa_core.viz.export_backend.write_figure_image",
        lambda _fig, path, **_kwargs: Path(path).write_bytes(b"png"),
    )

    repo_root = Path(__file__).resolve().parents[1]
    out_file = tmp_path / "sweep.xlsx"
    main(
        [
            "--config",
            str(config_path),
            "--index",
            str(repo_root / "data" / "sp500tr_fred_divyield.csv"),
            "--seed",
            "42",
            "--png",
            "--output",
            str(out_file),
        ]
    )

    assert out_file.exists()
    assert out_file.with_suffix(".png").read_bytes() == b"png"
