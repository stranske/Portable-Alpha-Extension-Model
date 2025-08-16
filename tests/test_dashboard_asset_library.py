from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
PKG = types.ModuleType("pa_core")
PKG.__path__ = [str(root / "pa_core")]
sys.modules.setdefault("pa_core", PKG)


class Uploaded:
    def __init__(self, path: Path):
        self.name = path.name
        self._data = path.read_bytes()

    def getvalue(self) -> bytes:
        return self._data


def test_asset_library_calibration(monkeypatch):
    module = runpy.run_path("dashboard/pages/1_Asset_Library.py", run_name="page")
    uploaded = Uploaded(Path("templates/asset_timeseries_wide_returns.csv"))
    st_mod = module["st"]
    monkeypatch.setattr(
        st_mod.sidebar, "text_input", lambda *a, **k: module["_DEF_THEME"]
    )
    module["main"].__globals__["apply_theme"] = lambda *a, **k: None
    monkeypatch.setattr(st_mod, "title", lambda *a, **k: None)
    monkeypatch.setattr(st_mod, "file_uploader", lambda *a, **k: uploaded)
    monkeypatch.setattr(st_mod, "dataframe", lambda *a, **k: None)
    monkeypatch.setattr(st_mod, "json", lambda *a, **k: None)
    monkeypatch.setattr(st_mod, "selectbox", lambda *a, **k: "SP500_TR")
    monkeypatch.setattr(st_mod, "button", lambda *a, **k: True)
    captured: dict[str, str] = {}

    def fake_download(label, data, **kwargs):
        captured["data"] = data

    monkeypatch.setattr(st_mod, "download_button", fake_download)
    from pa_core.data import DataImportAgent as RealImporter

    module["main"].__globals__["DataImportAgent"] = lambda *a, **k: RealImporter(
        min_obs=1
    )
    module["main"]()
    assert "SP500_TR" in captured["data"]
