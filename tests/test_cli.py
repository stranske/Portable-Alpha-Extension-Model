from pathlib import Path
import yaml
from pa_core.__main__ import main


def test_main_with_yaml(tmp_path):
    cfg = {"N_SIMULATIONS": 2, "N_MONTHS": 1}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    idx_csv = Path(__file__).resolve().parents[1] / "sp500tr_fred_divyield.csv"
    out_file = tmp_path / "out.xlsx"
    main([
        "--config",
        str(cfg_path),
        "--index",
        str(idx_csv),
        "--output",
        str(out_file),
    ])
    assert out_file.exists()


