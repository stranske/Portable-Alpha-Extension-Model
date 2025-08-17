import pandas as pd
import pytest
from dashboard.app import load_history


def test_load_history(tmp_path):
    df = pd.DataFrame({"Sim": [0, 0, 1], "Return": [0.1, 0.2, 0.3]})
    path = tmp_path / "Outputs.parquet"
    df.to_parquet(path)
    hist = load_history(str(path))
    assert hist is not None
    assert "mean_return" in hist.columns
    assert hist.loc[0, "mean_return"] == pytest.approx(0.15)
