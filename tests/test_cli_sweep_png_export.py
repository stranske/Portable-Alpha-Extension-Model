from pathlib import Path

import pandas as pd
import yaml

from pa_core.cli import main


def _run_sweep(
    monkeypatch,
    tmp_path,
    *export_flags: str,
    output_name: str = "sweep.xlsx",
    png_exporter=None,
    result_count: int = 1,
):
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
    results = [
        {"summary": summary.copy(), "combination_id": combination_id}
        for combination_id in range(1, result_count + 1)
    ]
    monkeypatch.setattr("pa_core.sweep.run_parameter_sweep", lambda *_args, **_kwargs: results)
    monkeypatch.setattr(
        "pa_core.reporting.sweep_excel.export_sweep_results",
        lambda _results, filename, **_kwargs: Path(filename).write_text("stub"),
    )
    monkeypatch.setattr(
        "pa_core.viz.export_backend.write_figure_image",
        png_exporter or (lambda _fig, path, **_kwargs: Path(path).write_bytes(b"png")),
    )

    repo_root = Path(__file__).resolve().parents[1]
    out_file = tmp_path / output_name
    main(
        [
            "--config",
            str(config_path),
            "--index",
            str(repo_root / "data" / "sp500tr_fred_divyield.csv"),
            "--seed",
            "42",
            *export_flags,
            "--output",
            str(out_file),
        ]
    )

    return out_file, results


def test_sweep_mode_honors_png_flag(monkeypatch, tmp_path):
    out_file, _results = _run_sweep(monkeypatch, tmp_path, "--png")

    assert out_file.exists()
    assert out_file.with_suffix(".png").read_bytes() == b"png"


def test_sweep_export_preserves_multi_dot_output_stem(monkeypatch, tmp_path):
    out_file, _results = _run_sweep(monkeypatch, tmp_path, "--png", output_name="sweep.q3.xlsx")

    assert out_file.exists()
    assert (tmp_path / "sweep.q3.png").read_bytes() == b"png"
    assert not (tmp_path / "sweep.png").exists()


def test_sweep_export_failure_does_not_abort_completed_run(monkeypatch, tmp_path, capsys):
    def fail_export(*_args, **_kwargs):
        raise OSError("destination unavailable")

    out_file, _results = _run_sweep(monkeypatch, tmp_path, "--png", png_exporter=fail_export)

    assert out_file.exists()
    assert "PNG export failed: destination unavailable" in capsys.readouterr().out


def test_sweep_pptx_contains_one_figure_per_scenario(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def save_pptx(figures, path, **kwargs):
        captured["figures"] = figures
        captured["path"] = path
        captured["kwargs"] = kwargs
        Path(path).write_bytes(b"pptx")

    monkeypatch.setattr("pa_core.viz.pptx_export.save", save_pptx)
    out_file, results = _run_sweep(monkeypatch, tmp_path, "--pptx", result_count=2)

    assert out_file.with_suffix(".pptx").read_bytes() == b"pptx"
    assert len(captured["figures"]) == len(results)
