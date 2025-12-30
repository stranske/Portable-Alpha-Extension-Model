import pandas as pd
import pytest

from dashboard.app import load_history, save_history


def test_load_history(tmp_path):
    df = pd.DataFrame({"Sim": [0, 0, 1], "Return": [0.1, 0.2, 0.3]})
    path = tmp_path / "Outputs.parquet"
    save_history(df, path)
    hist = load_history(str(path))
    assert hist is not None
    assert "mean_return" in hist.columns
    assert hist.loc[0, "mean_return"] == pytest.approx(0.15)


def test_load_history_csv_fallback(tmp_path, monkeypatch):
    df = pd.DataFrame({"Sim": [0, 0, 1], "Return": [0.1, 0.2, 0.3]})
    path = tmp_path / "Outputs.parquet"
    save_history(df, path)

    def _raise(*args, **kwargs):
        raise ImportError("no pyarrow")

    monkeypatch.setattr(pd, "read_parquet", _raise)
    hist = load_history(str(path))
    assert hist is not None
    assert hist.loc[0, "mean_return"] == pytest.approx(0.15)


def test_save_history_writes_csv_on_parquet_importerror(tmp_path, monkeypatch):
    df = pd.DataFrame({"Sim": [0, 1], "Return": [0.1, 0.2]})
    path = tmp_path / "Outputs.parquet"

    def _raise(*_args, **_kwargs):
        raise ImportError("no engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise)
    monkeypatch.setattr("dashboard.app.st.info", lambda *_args, **_kwargs: None)

    save_history(df, path)

    csv_path = path.with_suffix(".csv")
    assert csv_path.exists()
    assert "Sim" in csv_path.read_text()
