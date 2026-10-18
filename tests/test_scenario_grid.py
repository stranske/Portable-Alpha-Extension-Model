from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd


class _FakeStreamlit(ModuleType):
    def __init__(
        self,
        name: str,
        *,
        uploads: dict[str, Any] | None = None,
        checkbox_value: bool = False,
        button_values: dict[str, bool] | None = None,
        selection: Any = None,
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

    def number_input(self, _label: str, **kwargs: Any) -> Any:
        return kwargs["value"]

    def selectbox(self, _label: str, options: list[Any], **kwargs: Any) -> Any:
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

    def download_button(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def rerun(self) -> None:
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
                "Agent": ["Base", "Base", "Other"],
                "active_share": [0.4, 0.6, 0.4],
                "theta_extpa": [0.2, 0.3, 0.2],
                "terminal_AnnReturn": [0.08, 0.12, 0.01],
                "monthly_AnnVol": [0.04, 0.06, 0.01],
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
