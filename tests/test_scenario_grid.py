from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import yaml

from pa_core.config import ModelConfig


class _RerunSignal(Exception):
    """Raised by the fake ``st.rerun`` when a test allows reruns."""


class _FakeStreamlit(ModuleType):
    def __init__(
        self,
        name: str,
        *,
        uploads: dict[str, Any] | None = None,
        checkbox_value: bool = False,
        button_values: dict[str, bool] | None = None,
        selection: Any = None,
        number_values: dict[str, float] | None = None,
        selectbox_values: dict[str, Any] | None = None,
        allow_rerun: bool = False,
    ) -> None:
        super().__init__(name)
        self.session_state: dict = {}
        self.errors: list[str] = []
        self.infos: list[str] = []
        self.successes: list[str] = []
        self.uploads = uploads or {}
        self.checkbox_value = checkbox_value
        self.button_values = button_values or {}
        self.selection = selection
        self.number_values = number_values or {}
        self.selectbox_values = selectbox_values or {}
        self.allow_rerun = allow_rerun
        self.downloads: list[tuple[str, str]] = []
        self.sidebar = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def error(self, message: str) -> None:
        self.errors.append(message)

    def info(self, message: str) -> None:
        self.infos.append(message)

    def success(self, message: str) -> None:
        self.successes.append(message)

    def title(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def markdown(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def subheader(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def tabs(self, _labels: list[str]) -> tuple[_FakeStreamlit, _FakeStreamlit]:
        return self, self

    def file_uploader(self, label: str, **_kwargs: Any) -> Any:
        return self.uploads.get(label)

    def checkbox(self, _label: str, **_kwargs: Any) -> bool:
        return self.checkbox_value

    def number_input(self, label: str, **kwargs: Any) -> Any:
        return self.number_values.get(label, kwargs["value"])

    def selectbox(self, label: str, options: list[Any], **kwargs: Any) -> Any:
        if label in self.selectbox_values:
            return self.selectbox_values[label]
        return options[kwargs.get("index", 0)]

    def button(self, label: str, **_kwargs: Any) -> bool:
        return self.button_values.get(label, False)

    def progress(self, *_args: Any, **_kwargs: Any) -> _FakeStreamlit:
        return self

    def empty(self) -> None:
        pass

    def plotly_chart(self, *_args: Any, **_kwargs: Any) -> Any:
        return self.selection

    def page_link(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def download_button(self, label: str, data: str = "", *_args: Any, **_kwargs: Any) -> None:
        self.downloads.append((label, data))

    def rerun(self) -> None:
        if self.allow_rerun:
            raise _RerunSignal()
        raise AssertionError("rerun should not be called by this test")


class _FakeApp(ModuleType):
    def __init__(self) -> None:
        super().__init__("dashboard.app")

    def apply_theme(self, _path: str) -> None:
        pass

    def render_settings_sidebar(self) -> tuple[None, str]:
        return None, "theme.yaml"


def _load_module(monkeypatch) -> tuple[dict, _FakeStreamlit]:
    fake_st = _FakeStreamlit("streamlit")
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setitem(sys.modules, "dashboard.app", _FakeApp())
    module = runpy.run_path(str(Path("dashboard/pages/5_Scenario_Grid.py")))
    return module, fake_st


def test_get_empty_dataframe_cached(monkeypatch) -> None:
    module, _ = _load_module(monkeypatch)
    first = module["_get_empty_dataframe"]()
    second = module["_get_empty_dataframe"]()

    assert first is second
    assert isinstance(first, pd.DataFrame)


def test_grid_cache_round_trip_and_eviction(monkeypatch) -> None:
    module, fake_st = _load_module(monkeypatch)
    fake_st.session_state.clear()
    df = pd.DataFrame({"x": [1.0], "y": [2.0]})

    module["_set_grid_cache"]("alpha", df, "y_axis", 100.0)
    cached = module["_get_grid_cache"]("alpha")
    assert cached["grid_df"].equals(df)
    assert cached["y_col"] == "y_axis"
    assert cached["total_fund"] == 100.0

    module["_set_grid_cache"]("beta", df, "y_axis", 100.0)
    module["_set_grid_cache"]("gamma", df, "y_axis", 100.0)
    module["_set_grid_cache"]("delta", df, "y_axis", 100.0)

    cache = fake_st.session_state[module["_GRID_CACHE_KEY"]]
    assert "alpha" not in cache["entries"]
    assert cache["order"] == ["beta", "gamma", "delta"]


def test_extract_plotly_click(monkeypatch) -> None:
    module, _ = _load_module(monkeypatch)
    selection = {"selection": {"points": [{"x": "1.5", "y": 2}]}}
    assert module["_extract_plotly_click"](selection) == (1.5, 2.0)

    assert module["_extract_plotly_click"]({}) is None
    assert module["_extract_plotly_click"]({"selection": {"points": []}}) is None


def test_grid_helpers_handle_legacy_and_malformed_cache_state(monkeypatch) -> None:
    module, fake_st = _load_module(monkeypatch)
    frame = pd.DataFrame({"value": [1.0]})

    legacy = {"key": "legacy", "grid_df": frame, "y_col": "value", "total_fund": 10.0}
    fake_st.session_state[module["_GRID_CACHE_KEY"]] = legacy
    assert module["_get_grid_cache"]("legacy") is legacy

    fake_st.session_state[module["_GRID_CACHE_KEY"]] = {"entries": [], "order": {"bad"}}
    module["_set_grid_cache"]("repaired", frame, "value", 10.0)
    repaired = fake_st.session_state[module["_GRID_CACHE_KEY"]]
    assert repaired["order"] == ["repaired"]
    assert module["_get_grid_cache"]("repaired")["grid_df"].equals(frame)

    assert (
        module["_extract_plotly_click"]({"selection": {"points": [{"x": object(), "y": 1}]}})
        is None
    )


def test_read_csv_from_bytes(monkeypatch) -> None:
    module, _ = _load_module(monkeypatch)

    class _Upload:
        def getvalue(self) -> bytes:
            return b"col\n1\n2\n"

    df = module["_read_csv"](_Upload())
    assert df["col"].tolist() == [1, 2]


class _Upload:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def getvalue(self) -> bytes:
        return self.payload


class _FakeFigure:
    def __init__(self) -> None:
        self.scatter_calls: list[dict[str, Any]] = []
        self.hovertemplate: str | None = None

    def add_scatter(self, **kwargs: Any) -> None:
        self.scatter_calls.append(kwargs)

    def update_traces(self, **kwargs: Any) -> None:
        self.hovertemplate = kwargs["hovertemplate"]


def _patch_global(module: dict, name: str, value: Any) -> None:
    module[name] = value
    module["main"].__globals__[name] = value


def test_main_renders_uploaded_grid_frontier_and_missing_compute_input(monkeypatch) -> None:
    upload = _Upload(b"AE_leverage,ExtPA_frac,Sharpe\n1,0.2,1.1\n2,0.4,1.4\n")
    fake_st = _FakeStreamlit("streamlit", uploads={"Grid CSV": upload})
    module, _ = _load_module(monkeypatch)
    figure = _FakeFigure()
    _patch_global(module, "st", fake_st)
    _patch_global(module, "render_settings_sidebar", lambda: (None, "theme.yaml"))
    _patch_global(module, "apply_theme", lambda _path: None)
    _patch_global(module, "grid_heatmap", type("Heatmap", (), {"make": lambda *_a, **_k: figure}))

    module["main"]()

    assert figure.scatter_calls[0]["name"] == "Frontier"
    assert (
        fake_st.infos[-1]
        == "Upload an index returns CSV or tick 'Use bundled sample data' to run the sweep."
    )


def test_main_runs_cached_sweep_and_promotes_clicked_selection(monkeypatch) -> None:
    fake_st = _FakeStreamlit(
        "streamlit",
        checkbox_value=True,
        button_values={"Run sweep": True},
        selection={"selection": {"points": [{"x": 0.4, "y": 0.2}]}},
    )
    module, _ = _load_module(monkeypatch)
    figure = _FakeFigure()
    _patch_global(module, "st", fake_st)
    _patch_global(module, "render_settings_sidebar", lambda: (None, "theme.yaml"))
    _patch_global(module, "apply_theme", lambda _path: None)
    _patch_global(module, "load_bundled_sample_index", lambda: pd.Series([0.01, 0.02]))
    _patch_global(
        module,
        "build_sample_model_config",
        lambda **_kwargs: type("Cfg", (), {"total_fund_capital": 100.0})(),
    )
    _patch_global(module, "make_grid_cache_key", lambda *_args: "grid-key")
    _patch_global(module, "run_parameter_sweep_cached", lambda *_args, **_kwargs: object())
    _patch_global(
        module,
        "sweep_results_to_dataframe",
        lambda _results: pd.DataFrame(
            {
                "Agent": ["Base", "Total", "Total", "Other"],
                "active_share": [0.4, 0.4, 0.6, 0.4],
                "theta_extpa": [0.2, 0.2, 0.3, 0.2],
                "terminal_AnnReturn": [0.5, 0.08, 0.12, 0.01],
                "monthly_AnnVol": [0.5, 0.04, 0.06, 0.01],
            }
        ),
    )
    _patch_global(module, "grid_heatmap", type("Heatmap", (), {"make": lambda *_a, **_k: figure}))
    _patch_global(module, "render_png_download", lambda *_args, **_kwargs: None)
    _patch_global(
        module,
        "build_alpha_shares_payload",
        lambda active, theta: {"active_share": active, "theta_extpa": theta},
    )
    _patch_global(module, "bump_session_token", lambda state, key: state.__setitem__(key, 1))

    module["main"]()

    assert fake_st.session_state["scenario_grid_selection"] == {
        "active_share": 0.4,
        "theta_extpa": 0.2,
    }
    assert fake_st.session_state["scenario_grid_promotion_token"] == 1
    assert figure.scatter_calls[0]["name"] == "Frontier"
    assert figure.hovertemplate is not None


_X_LABEL = "Active share"
_Y_LABEL = "External PA value"
_GRID_NUMBERS = {
    "ExtPA alpha min [%]": 25.0,
    "ExtPA alpha max [%]": 50.0,
    "ExtPA alpha step [%]": 25.0,
    "Active share min [%]": 50.0,
    "Active share max [%]": 100.0,
    "Active share step [%]": 50.0,
}


class _HeatmapRecorder:
    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []
        self.figure = _FakeFigure()

    def make(self, df: pd.DataFrame, **_kwargs: Any) -> _FakeFigure:
        self.frames.append(df.copy())
        return self.figure


def _run_grid_page(
    monkeypatch,
    session_state: dict,
    *,
    buttons: dict[str, bool],
    number_values: dict[str, float] | None = None,
    selectbox_values: dict[str, Any] | None = None,
    selection: Any = None,
    allow_rerun: bool = False,
    sweep_spy: list | None = None,
) -> tuple[_FakeStreamlit, _HeatmapRecorder]:
    """Run the real page (real sweep, bundled index) against a fake Streamlit."""
    fake_st = _FakeStreamlit(
        "streamlit",
        checkbox_value=True,
        button_values=buttons,
        selection=selection,
        number_values={**_GRID_NUMBERS, **(number_values or {})},
        selectbox_values=selectbox_values,
        allow_rerun=allow_rerun,
    )
    fake_st.session_state = session_state
    module, _ = _load_module(monkeypatch)
    heatmap = _HeatmapRecorder()
    _patch_global(module, "st", fake_st)
    _patch_global(module, "render_settings_sidebar", lambda: (None, "theme.yaml"))
    _patch_global(module, "apply_theme", lambda _path: None)
    _patch_global(module, "grid_heatmap", heatmap)
    _patch_global(module, "render_png_download", lambda *_a, **_k: None)
    if sweep_spy is not None:
        real_sweep = module["run_parameter_sweep_cached"]

        def _spy(*args: Any, **kwargs: Any) -> Any:
            results = real_sweep(*args, **kwargs)
            sweep_spy.extend(results)
            return results

        _patch_global(module, "run_parameter_sweep_cached", _spy)
    try:
        module["main"]()
    except _RerunSignal:
        pass
    return fake_st, heatmap


def _scenario_download(fake_st: _FakeStreamlit) -> ModelConfig:
    labels = [label for label, _ in fake_st.downloads]
    assert "Download Scenario YAML" in labels
    data = dict(fake_st.downloads)["Download Scenario YAML"]
    return ModelConfig.model_validate(yaml.safe_load(data))


def test_funded_grid_totals_and_config_roundtrip(monkeypatch) -> None:
    """Four-cell bundled-index grid: funded sleeves, varying Total, valid export."""
    session: dict = {}
    sweep_results: list = []
    fake_st, heatmap = _run_grid_page(
        monkeypatch,
        session,
        buttons={"Run sweep": True},
        sweep_spy=sweep_results,
    )

    # Funded ExternalPA/ActiveExt sleeves are present in every grid cell.
    assert len(sweep_results) == 4
    for result in sweep_results:
        agents = set(result["summary"]["Agent"])
        assert {"ExternalPA", "ActiveExt", "Total"} <= agents

    # Compatibility control: the Base benchmark stays invariant across cells while
    # the aggregated portfolio Total varies with both grid axes.
    def _metric(agent: str, column: str) -> list[float]:
        values = []
        for result in sweep_results:
            frame = result["summary"]
            values.append(float(frame.loc[frame["Agent"] == agent, column].iloc[0]))
        return values

    assert len({round(v, 12) for v in _metric("Base", "terminal_AnnReturn")}) == 1
    assert len({round(v, 12) for v in _metric("Total", "terminal_AnnReturn")}) == 4

    grid = heatmap.frames[-1]
    assert len(grid) == 4
    assert grid["Sharpe"].nunique() == 4
    assert grid["terminal_AnnReturn"].nunique() == 4
    assert sorted(grid["active_share"].unique()) == [0.5, 1.0]

    # Promote each cell through both supported paths with the same cached basis.
    for y_mode in ("External alpha fraction (theta)", "External PA $ (mm)"):
        mode_session: dict = {}
        _, mode_heatmap = _run_grid_page(
            monkeypatch,
            mode_session,
            buttons={"Run sweep": True},
            selectbox_values={"Y axis": y_mode},
        )
        mode_grid = mode_heatmap.frames[-1]
        entry = next(iter(mode_session["scenario_grid_cache"]["entries"].values()))
        total_fund = float(entry["total_fund"])
        assert total_fund > 0
        for _, cell in mode_grid.iterrows():
            share = float(cell["active_share"])
            y_value = float(cell["y_axis"])
            theta = y_value if "theta" in y_mode else y_value / total_fund
            select_values = {"Y axis": y_mode, _X_LABEL: share, _Y_LABEL: y_value}

            # Click path.
            click_state = dict(mode_session)
            click_st, _ = _run_grid_page(
                monkeypatch,
                click_state,
                buttons={},
                selectbox_values=select_values,
                selection={"selection": {"points": [{"x": share, "y": y_value}]}},
            )
            click_payload = click_state["promoted_alpha_shares"]

            # Selectbox path: promote, rerun, then read back the export.
            box_state = dict(mode_session)
            _run_grid_page(
                monkeypatch,
                box_state,
                buttons={"Promote to session": True},
                selectbox_values=select_values,
                allow_rerun=True,
            )
            box_payload = box_state["promoted_alpha_shares"]
            box_st, _ = _run_grid_page(
                monkeypatch, box_state, buttons={}, selectbox_values=select_values
            )

            assert click_payload == box_payload
            assert click_payload["active_share"] == share
            assert abs(click_payload["theta_extpa"] - theta) < 1e-9
            assert theta in (0.25, 0.5)

            # Exported YAML reloads via ModelConfig with financing mode + selection.
            for exported in (_scenario_download(click_st), _scenario_download(box_st)):
                assert exported.financing_mode == "per_path"
                assert exported.analysis_mode == "alpha_shares"
                assert exported.active_share == share
                assert abs(exported.theta_extpa - theta) < 1e-9
                assert exported.external_pa_capital > 0
                assert exported.active_ext_capital > 0
                names = {agent.name for agent in exported.agents}
                assert {"ExternalPA", "ActiveExt"} <= names
