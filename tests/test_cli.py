from pathlib import Path
from unittest.mock import create_autospec

import plotly.graph_objects as go
import yaml

from pa_core.cli import main
from pa_core.run_artifact_bundle import RunArtifactBundle


def test_main_with_yaml(tmp_path):
    cfg = {"N_SIMULATIONS": 2, "N_MONTHS": 1, "financing_mode": "broadcast"}
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
    """Run the CLI and image backend while isolating the external browser renderer."""

    def write_image(figure, path, **_kwargs):
        Path(path).write_bytes(b"rendered-png")

    renderer = create_autospec(go.Figure.write_image, side_effect=write_image)
    monkeypatch.setattr(go.Figure, "write_image", renderer)
    cfg = {"N_SIMULATIONS": 2, "N_MONTHS": 1, "financing_mode": "broadcast"}
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
    renderer.assert_called_once()
    figure, image_path = renderer.call_args.args
    assert isinstance(figure, go.Figure)
    assert figure.data
    assert image_path == out_file.with_suffix(".png")
    assert renderer.call_args.kwargs == {}
    assert image_path.read_bytes() == b"rendered-png"


def test_main_with_bundle(tmp_path):
    cfg = {"N_SIMULATIONS": 2, "N_MONTHS": 1, "financing_mode": "broadcast"}
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
