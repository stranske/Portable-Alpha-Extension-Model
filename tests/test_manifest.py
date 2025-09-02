import json
from pathlib import Path

import yaml

from pa_core.cli import main


def test_manifest_written(tmp_path):
    cfg = {"N_SIMULATIONS": 1, "N_MONTHS": 1}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    idx_csv = Path(__file__).resolve().parents[1] / "sp500tr_fred_divyield.csv"
    out_file = tmp_path / "out.xlsx"
    seed = 123

    main(
        [
            "--config",
            str(cfg_path),
            "--index",
            str(idx_csv),
            "--output",
            str(out_file),
            "--seed",
            str(seed),
        ]
    )

    manifest_path = out_file.with_name("manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["seed"] == seed
    assert manifest["config"]["N_SIMULATIONS"] == 1
    assert str(cfg_path) in manifest["data_files"]
