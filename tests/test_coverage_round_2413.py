"""Coverage for the Scenario Wizard's top-level navigation contract."""

import runpy
from pathlib import Path
from typing import Any

import pytest

from pa_core.wizard_schema import AnalysisMode, get_default_config


class _RerunRequested(Exception):
    """Sentinel raised by the Streamlit fake when main() requests a rerun."""


class _StopRequested(Exception):
    """Sentinel raised by the Streamlit fake when main() calls st.stop()."""


class _UploadedIndex:
    """Minimal stand-in for an uploaded index CSV."""

    def getvalue(self) -> bytes:
        return b"date,return\n2020-01-01,0.01\n"


class _SessionState(dict[str, Any]):
    def __getattr__(self, key: str) -> Any:
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class _Container:
    def __enter__(self) -> "_Container":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


class _Streamlit:
    def __init__(
        self,
        step: int | None = 1,
        *,
        pressed: str | None = None,
        seed_session: bool = True,
    ) -> None:
        self.session_state = _SessionState()
        if seed_session:
            if step is not None:
                self.session_state.wizard_step = step
            self.session_state.wizard_config = get_default_config(AnalysisMode.RETURNS)
        self.sidebar = self
        self._pressed = pressed
        self.rerun_count = 0

    def button(self, label: str = "", *_args: Any, **_kwargs: Any) -> bool:
        return bool(self._pressed and label == self._pressed)

    def file_uploader(self, *_args: Any, **_kwargs: Any) -> Any:
        return None

    def columns(self, count: int, *_args: Any, **_kwargs: Any) -> list[_Container]:
        return [_Container() for _ in range(count)]

    def expander(self, *_args: Any, **_kwargs: Any) -> _Container:
        return _Container()

    def markdown(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def subheader(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def text_input(self, label: str = "", *_args: Any, **_kwargs: Any) -> str:
        return "out.xlsx"

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def stop(self) -> None:
        raise _StopRequested()

    def rerun(self) -> None:
        self.rerun_count += 1
        raise _RerunRequested()


WIZARD = Path(__file__).resolve().parents[1] / "dashboard/pages/3_Scenario_Wizard.py"

_STEP_RENDERERS = (
    "_render_step_1_analysis_mode",
    "_render_step_2_capital",
    "_render_step_3_returns_risk",
    "_render_step_4_correlations",
)


def _install_main_fakes(
    module: dict[str, Any],
    fake_st: _Streamlit,
    *,
    step_5_returns: bool = False,
) -> list[tuple[str, int]]:
    """Wire main() to isolated fakes; return the shared call log."""
    calls: list[tuple[str, int]] = []
    globals_ = module["main"].__globals__
    globals_["st"] = fake_st
    globals_["render_settings_sidebar"] = lambda: (None, None)
    globals_["apply_theme"] = lambda _path: None
    globals_["_render_config_chat_sidebar"] = lambda: None
    globals_["_render_progress_bar"] = lambda current: calls.append(("progress", current))
    for number, name in enumerate(_STEP_RENDERERS, start=1):
        globals_[name] = lambda config, number=number: calls.append(("renderer", number)) or config
    globals_["_render_step_5_review"] = lambda _config: (
        calls.append(("renderer", 5)) or step_5_returns
    )
    return calls


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
def test_main_routes_each_step_to_its_renderer(step: int) -> None:
    """A wizard step must render its matching page without advancing by itself."""
    module = runpy.run_path(str(WIZARD))
    fake_st = _Streamlit(step)
    calls = _install_main_fakes(module, fake_st)

    module["main"]()

    assert calls == [("progress", step), ("renderer", step)]
    assert fake_st.session_state.wizard_step == step
    assert fake_st.rerun_count == 0


def test_main_cold_start_initializes_and_routes_to_step_1() -> None:
    """Empty session state must land on step 1 and dispatch its renderer."""
    module = runpy.run_path(str(WIZARD))
    fake_st = _Streamlit(seed_session=False)
    calls = _install_main_fakes(module, fake_st)

    module["main"]()

    assert fake_st.session_state.wizard_step == 1
    assert "wizard_config" in fake_st.session_state
    assert calls == [("progress", 1), ("renderer", 1)]
    assert fake_st.rerun_count == 0


@pytest.mark.parametrize("step", [0, 6, 99])
def test_main_unknown_step_renders_progress_without_step_body(step: int) -> None:
    """Steps outside 1–5 must not fall through into any step renderer."""
    module = runpy.run_path(str(WIZARD))
    fake_st = _Streamlit(step)
    calls = _install_main_fakes(module, fake_st)

    module["main"]()

    assert calls == [("progress", step)]
    assert fake_st.session_state.wizard_step == step
    assert fake_st.rerun_count == 0


@pytest.mark.parametrize(
    ("step", "pressed", "expected_step"),
    [
        (1, "➡️ Next: Capital Allocation", 2),
        (2, "⬅️ Back: Analysis Mode", 1),
        (2, "➡️ Next: Return & Risk", 3),
        (3, "⬅️ Back: Capital Allocation", 2),
        (3, "➡️ Next: Correlations", 4),
        (4, "⬅️ Back: Return & Risk", 3),
        (4, "➡️ Next: Review & Run", 5),
        (5, "⬅️ Back: Correlations", 4),
    ],
)
def test_main_step_buttons_route_to_adjacent_steps(
    step: int, pressed: str, expected_step: int
) -> None:
    """Top-level Next/Back buttons must move wizard_step and request a rerun."""
    module = runpy.run_path(str(WIZARD))
    fake_st = _Streamlit(step, pressed=pressed)
    calls = _install_main_fakes(module, fake_st)

    with pytest.raises(_RerunRequested):
        module["main"]()

    assert calls == [("progress", step), ("renderer", step)]
    assert fake_st.session_state.wizard_step == expected_step
    assert fake_st.rerun_count == 1


@pytest.mark.parametrize("step", [2, 3, 4, 5])
def test_main_sidebar_reset_routes_to_step_1(step: int) -> None:
    """Sidebar Reset to Step 1 must jump from any later step without rendering it."""
    module = runpy.run_path(str(WIZARD))
    fake_st = _Streamlit(step, pressed="🏠 Reset to Step 1")
    calls = _install_main_fakes(module, fake_st)

    with pytest.raises(_RerunRequested):
        module["main"]()

    assert calls == []
    assert fake_st.session_state.wizard_step == 1
    assert fake_st.rerun_count == 1


def test_main_step_5_run_gate_opens_index_upload_without_executing() -> None:
    """Review Run must open the index-upload gate; missing CSV must not execute sim."""
    module = runpy.run_path(str(WIZARD))
    fake_st = _Streamlit(5)
    # text_input is only reached inside the run_simulation gate.
    recorded: list[str] = []
    fake_st.text_input = lambda label, *_a, **_k: recorded.append(label) or "out.xlsx"
    calls = _install_main_fakes(module, fake_st, step_5_returns=True)

    module["main"]()

    assert calls == [("progress", 5), ("renderer", 5)]
    assert fake_st.session_state.last_wizard_config is fake_st.session_state.wizard_config
    assert recorded == ["Output workbook"]
    assert fake_st.rerun_count == 0


def test_main_step_5_run_gate_stops_when_loaded_config_is_invalid() -> None:
    """Uploaded index with an invalid wizard YAML must surface the error and stop."""
    module = runpy.run_path(str(WIZARD))
    fake_st = _Streamlit(5)
    fake_st.file_uploader = lambda label="", *_a, **_k: (
        _UploadedIndex() if "Index" in str(label) else None
    )
    errors: list[str] = []
    fake_st.error = lambda message, *_a, **_k: errors.append(str(message))
    calls = _install_main_fakes(module, fake_st, step_5_returns=True)
    globals_ = module["main"].__globals__
    globals_["_build_yaml_from_config"] = lambda _config: {"invalid": True}
    globals_["load_config"] = lambda _data: (_ for _ in ()).throw(ValueError("bad config"))

    with pytest.raises(_StopRequested):
        module["main"]()

    assert calls == [("progress", 5), ("renderer", 5)]
    assert fake_st.session_state.last_wizard_config is fake_st.session_state.wizard_config
    assert errors and "correction before the dashboard can run" in errors[0]
    assert fake_st.rerun_count == 0
