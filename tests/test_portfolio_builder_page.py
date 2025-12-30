from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


class _RerunCalled(Exception):
    pass


class _FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeSidebar:
    def __init__(self, calls: list[tuple[str, Any]]):
        self._calls = calls

    def text_input(self, label: str, value: str) -> str:
        self._calls.append(("sidebar_text_input", label))
        return value


class FakeStreamlit(ModuleType):
    def __init__(
        self,
        name: str,
        number_inputs: list[float] | None = None,
        file_uploader: Any | None = None,
        button_value: bool = False,
    ) -> None:
        super().__init__(name)
        self.calls: list[tuple[str, Any]] = []
        self.session_state: dict[str, Any] = {}
        self._number_inputs = list(number_inputs or [])
        self._file_uploader = file_uploader
        self._button_value = button_value
        self.sidebar = _FakeSidebar(self.calls)

    def title(self, message: str) -> None:
        self.calls.append(("title", message))

    def info(self, message: str) -> None:
        self.calls.append(("info", message))

    def warning(self, message: str) -> None:
        self.calls.append(("warning", message))

    def subheader(self, message: str) -> None:
        self.calls.append(("subheader", message))

    def write(self, message: str) -> None:
        self.calls.append(("write", message))

    def error(self, message: str) -> None:
        self.calls.append(("error", message))

    def expander(self, label: str, expanded: bool = False) -> _FakeContext:
        self.calls.append(("expander", label))
        return _FakeContext()

    def number_input(self, label: str, **kwargs: Any) -> float:
        self.calls.append(("number_input", label))
        if self._number_inputs:
            return float(self._number_inputs.pop(0))
        return float(kwargs.get("value", 0.0))

    def file_uploader(self, label: str, **kwargs: Any) -> Any | None:
        self.calls.append(("file_uploader", label))
        return self._file_uploader

    def button(self, label: str) -> bool:
        self.calls.append(("button", label))
        return self._button_value

    def download_button(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("download_button", kwargs.get("file_name")))

    def rerun(self) -> None:
        self.calls.append(("rerun", None))
        raise _RerunCalled("rerun")


class _FakeApp(ModuleType):
    def __init__(self) -> None:
        super().__init__("dashboard.app")
        self._DEF_THEME = "theme.yaml"
        self.applied_theme: str | None = None

    def apply_theme(self, path: str) -> None:
        self.applied_theme = path


class _Upload:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def getvalue(self) -> bytes:
        return self._payload


class _Scenario:
    def __init__(self, assets: list[Any], correlations: Any | None = None) -> None:
        self.assets = assets
        self.correlations = correlations
        self.portfolios: list[Any] = []

    def model_dump(self) -> dict[str, Any]:
        return {"assets": [asset.id for asset in self.assets]}


class _Asset:
    def __init__(self, asset_id: str) -> None:
        self.id = asset_id


def _load_module(monkeypatch: pytest.MonkeyPatch, fake_st: FakeStreamlit) -> dict:
    fake_app = _FakeApp()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setitem(sys.modules, "dashboard.app", fake_app)
    module = runpy.run_path(str(Path("dashboard/pages/2_Portfolio_Builder.py")))
    return module


def _patch_module_global(module: dict, name: str, value: Any) -> None:
    module[name] = value
    module["main"].__globals__[name] = value


def test_portfolio_builder_promoted_values_trigger_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = FakeStreamlit("streamlit")
    fake_st.session_state["scenario_grid_selection"] = {
        "active_share": 40.0,
        "theta_extpa": 60.0,
    }
    fake_st.session_state["scenario_grid_promotion_token"] = 1
    module = _load_module(monkeypatch, fake_st)

    with pytest.raises(_RerunCalled):
        module["main"]()

    assert fake_st.session_state["alpha_shares_active_share"] == 0.4
    assert fake_st.session_state["alpha_shares_theta_extpa"] == 0.6
    assert fake_st.session_state["portfolio_builder_autorun"] is True


def test_portfolio_builder_requires_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = FakeStreamlit("streamlit", file_uploader=None)
    module = _load_module(monkeypatch, fake_st)

    module["main"]()

    assert any(call == ("info", "Upload an asset library YAML to begin.") for call in fake_st.calls)


def test_portfolio_builder_empty_assets_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = FakeStreamlit("streamlit", file_uploader=_Upload(b"assets: []"))
    module = _load_module(monkeypatch, fake_st)
    _patch_module_global(module, "load_scenario", lambda _: _Scenario([]))

    module["main"]()

    assert any(call == ("warning", "No assets found in file.") for call in fake_st.calls)


def test_portfolio_builder_zero_weights_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = FakeStreamlit(
        "streamlit",
        number_inputs=[0.5, 0.5, 0.0, 0.0],
        file_uploader=_Upload(b"assets: []"),
    )
    module = _load_module(monkeypatch, fake_st)
    _patch_module_global(
        module, "load_scenario", lambda _: _Scenario([_Asset("A"), _Asset("B")])
    )

    module["main"]()

    assert any(
        call == ("warning", "Enter weights for at least one asset.")
        for call in fake_st.calls
    )


def test_portfolio_builder_autorun_generates_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = FakeStreamlit(
        "streamlit",
        number_inputs=[0.25, 0.75, 1.0, 1.0],
        file_uploader=_Upload(b"assets: []"),
    )
    fake_st.session_state["portfolio_builder_autorun"] = True
    module = _load_module(monkeypatch, fake_st)
    scenario = _Scenario([_Asset("A"), _Asset("B")], correlations="corr")
    _patch_module_global(module, "load_scenario", lambda _: scenario)

    class _FakeAggregator:
        def __init__(self, assets, correlations) -> None:
            self.assets = assets
            self.correlations = correlations

        def aggregate(self, weights):
            return 0.12, 0.34

    _patch_module_global(module, "PortfolioAggregator", _FakeAggregator)

    module["main"]()

    assert any(call[0] == "download_button" for call in fake_st.calls)
    assert any("Expected return" in call[1] for call in fake_st.calls if call[0] == "write")
    assert fake_st.session_state["portfolio_builder_autorun"] is False
    assert scenario.portfolios
