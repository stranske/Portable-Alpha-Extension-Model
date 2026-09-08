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
    monkeypatch.setattr(st_mod.sidebar, "text_input", lambda *a, **k: module["_DEF_THEME"])
    module["main"].__globals__["apply_theme"] = lambda *a, **k: None
    monkeypatch.setattr(st_mod, "title", lambda *a, **k: None)

    uploads = [uploaded, None]

    def fake_uploader(*a, **k):
        return uploads.pop(0)

    monkeypatch.setattr(st_mod, "file_uploader", fake_uploader)
    monkeypatch.setattr(st_mod, "dataframe", lambda *a, **k: None)
    monkeypatch.setattr(st_mod, "json", lambda *a, **k: None)

    # Provide enough values for all selectbox calls (cov_shrinkage, vol_regime, index_id, del_id)
    selects = ["none", "single", "SP500_TR", ""]
    monkeypatch.setattr(st_mod, "selectbox", lambda *a, **k: selects.pop(0))

    buttons = [True, False]
    monkeypatch.setattr(st_mod, "button", lambda *a, **k: buttons.pop(0))

    monkeypatch.setattr(st_mod, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(st_mod, "session_state", {})

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

    module["main"].__globals__["DataImportAgent"] = lambda *a, **k: RealImporter(min_obs=1)
    module["main"]()
    assert "Download Asset Library YAML" in labels
    assert "Download Presets JSON" in labels
    assert "SP500_TR" in captured["data"]


def test_asset_library_bundled_sample_at_real_defaults():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("dashboard/pages/1_Asset_Library.py").run()
    assert not app.exception
    app.checkbox[0].check().run()

    assert not app.exception
    assert not app.error
    minimum = next(w for w in app.number_input if w.label == "Min observations per id")
    assert minimum.value == 36
    assert any("Data loaded successfully" in message.value for message in app.success)
    loaded = app.dataframe[0].value
    assert set(loaded["id"]) == {"FUND_A", "FUND_B", "SP500_TR"}
    assert (loaded.groupby("id").size() >= minimum.value).all()


def test_asset_library_insufficient_upload_guidance(monkeypatch, tmp_path):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    # AppTest has no file-upload setter; replace only that boundary. The page,
    # importer, widget defaults, CSV parsing and validation remain real.
    rows = Path("templates/asset_timeseries_wide_returns.csv").read_text().splitlines()
    short_csv = tmp_path / "short_returns.csv"
    short_csv.write_text("\n".join(rows[:25]) + "\n")
    uploaded = Uploaded(short_csv)
    original_uploader = st.file_uploader
    monkeypatch.setattr(
        st,
        "file_uploader",
        lambda label, *a, **k: (
            uploaded if label == "Drag-and-drop CSV/XLSX" else original_uploader(label, *a, **k)
        ),
    )
    # Track the real upload tempfile so returning early must still clean it up.
    import tempfile

    original_mkstemp = tempfile.mkstemp
    upload_paths = []

    def track_mkstemp(*a, **k):
        fd, path = original_mkstemp(*a, **k)
        if k.get("suffix") == ".csv":
            upload_paths.append(Path(path))
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", track_mkstemp)
    app = AppTest.from_file("dashboard/pages/1_Asset_Library.py").run()

    assert not app.exception
    assert not app.success
    assert not app.dataframe
    assert len(app.error) == 1
    message = app.error[0].value
    assert "36" in message
    assert "Upload" in message
    assert "monthly observations" in message
    for series_id in ("FUND_A", "FUND_B", "SP500_TR"):
        assert series_id in message
    assert not any(button.label == "Calibrate" for button in app.button)
    assert upload_paths and all(not path.exists() for path in upload_paths)
