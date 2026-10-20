"""Coverage for the Scenario Wizard's top-level navigation contract."""

import runpy
from pathlib import Path
from typing import Any

import pytest

from pa_core.wizard_schema import AnalysisMode, get_default_config


class _RerunRequested(Exception):
    """Sentinel raised by the Streamlit fake when main() requests a rerun."""


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
    def __init__(self, step: int, *, pressed: str | None = None) -> None:
        self.session_state = _SessionState(
            wizard_step=step, wizard_config=get_default_config(AnalysisMode.RETURNS)
        )
        self.sidebar = self
        self._pressed = pressed
        self.rerun_count = 0

    def button(self, label: str = "", *_args: Any, **_kwargs: Any) -> bool:
        return bool(self._pressed and label == self._pressed)

    def file_uploader(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def columns(self, count: int, *_args: Any, **_kwargs: Any) -> list[_Container]:
        return [_Container() for _ in range(count)]

    def expander(self, *_args: Any, **_kwargs: Any) -> _Container:
        return _Container()

    def markdown(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def subheader(self, *_args: Any, **_kwargs: Any) -> None:
        pass

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


def _install_main_fakes(module: dict[str, Any], fake_st: _Streamlit) -> list[tuple[str, int]]:
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
    globals_["_render_step_5_review"] = lambda _config: calls.append(("renderer", 5)) or False
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


def test_main_sidebar_reset_routes_to_step_1() -> None:
    """Sidebar Reset to Step 1 must jump from a later step without rendering it."""
    module = runpy.run_path(str(WIZARD))
    fake_st = _Streamlit(4, pressed="🏠 Reset to Step 1")
    calls = _install_main_fakes(module, fake_st)

    with pytest.raises(_RerunRequested):
        module["main"]()

    assert calls == []
    assert fake_st.session_state.wizard_step == 1
    assert fake_st.rerun_count == 1
