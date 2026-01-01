from pathlib import Path

import yaml

from pa_core.cli import main
from pa_core.run_artifact_bundle import RunArtifactBundle


def test_main_with_yaml(tmp_path):
    cfg = {"N_SIMULATIONS": 2, "N_MONTHS": 1}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    idx_csv = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"
    out_file = tmp_path / "out.xlsx"
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
    assert out_file.exists()


def test_main_with_png(tmp_path, monkeypatch):
    cfg = {"N_SIMULATIONS": 2, "N_MONTHS": 1}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    idx_csv = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"
    out_file = tmp_path / "out.xlsx"
    monkeypatch.chdir(tmp_path)
    main(
        [
            "--config",
            str(cfg_path),
            "--index",
            str(idx_csv),
            "--output",
            str(out_file),
            "--png",
        ]
    )
    assert out_file.exists()


def test_main_with_bundle(tmp_path):
    cfg = {"N_SIMULATIONS": 2, "N_MONTHS": 1}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_text = yaml.safe_dump(cfg)
    cfg_path.write_text(cfg_text)
    idx_csv = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"
    out_file = tmp_path / "out.xlsx"
    bundle_dir = tmp_path / "bundle"
    main(
        [
            "--config",
            str(cfg_path),
            "--index",
            str(idx_csv),
            "--output",
            str(out_file),
            "--bundle",
            str(bundle_dir),
        ]
    )
    assert out_file.exists()

    bundle = RunArtifactBundle.load(bundle_dir)
    assert bundle.verify()
    assert (bundle_dir / "config.yaml").read_text() == cfg_text
