import pandas as pd
import pathlib
from pa_core.reporting import export_to_excel


def test_shortfall_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({
        "AnnReturn": [0.05],
        "AnnVol": [0.02],
        "TrackingErr": [0.01],
        "Agent": ["Base"],
        "ShortfallProb": [0.02],
    })
    export_to_excel({}, df, {}, filename="Outputs.xlsx")
    fn = pathlib.Path("Outputs.xlsx")
    assert fn.exists(), "Outputs.xlsx missing"
    cols = pd.read_excel(fn, sheet_name="Summary").columns
    assert "ShortfallProb" in cols
