from __future__ import annotations

import runpy
from pathlib import Path


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

    uploads = [uploaded, None]

    def fake_uploader(*a, **k):
        return uploads.pop(0)

    monkeypatch.setattr(st_mod, "file_uploader", fake_uploader)
    monkeypatch.setattr(st_mod, "dataframe", lambda *a, **k: None)
    monkeypatch.setattr(st_mod, "json", lambda *a, **k: None)

    selects = ["SP500_TR", ""]
    monkeypatch.setattr(st_mod, "selectbox", lambda *a, **k: selects.pop(0))

    buttons = [True, False]
    monkeypatch.setattr(st_mod, "button", lambda *a, **k: buttons.pop(0))

    monkeypatch.setattr(st_mod, "subheader", lambda *a, **k: None)
    st_mod.session_state = {}

    class DummyForm:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(st_mod, "form", lambda *a, **k: DummyForm())
    monkeypatch.setattr(st_mod, "text_input", lambda *a, **k: "")
    monkeypatch.setattr(st_mod, "number_input", lambda *a, **k: 0.0)
    monkeypatch.setattr(st_mod, "form_submit_button", lambda *a, **k: False)

    captured: dict[str, str] = {}
    labels: list[str] = []

    def fake_download(label, data, **kwargs):
        labels.append(label)
        if label == "Download Asset Library YAML":
            captured["label"] = label
            captured["data"] = data

    monkeypatch.setattr(st_mod, "download_button", fake_download)
    from pa_core.data import DataImportAgent as RealImporter

    module["main"].__globals__["DataImportAgent"] = lambda *a, **k: RealImporter(
        min_obs=1
    )
    module["main"]()
    assert "Download Asset Library YAML" in labels
    assert "Download Presets JSON" in labels
    assert "SP500_TR" in captured["data"]
