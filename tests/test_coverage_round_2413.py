"""Coverage for the Scenario Wizard's top-level navigation contract."""

import runpy
from pathlib import Path
from typing import Any

import pytest

from pa_core.wizard_schema import AnalysisMode, get_default_config


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
    def __init__(self, step: int) -> None:
        self.session_state = _SessionState(
            wizard_step=step, wizard_config=get_default_config(AnalysisMode.RETURNS)
        )
        self.sidebar = self

    def button(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

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


WIZARD = Path(__file__).resolve().parents[1] / "dashboard/pages/3_Scenario_Wizard.py"


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
def test_main_routes_each_step_to_its_renderer(step: int) -> None:
    """A wizard step must render its matching page without advancing by itself."""
    module = runpy.run_path(str(WIZARD))
    fake_st = _Streamlit(step)
    calls: list[tuple[str, int]] = []
    globals_ = module["main"].__globals__
    globals_["st"] = fake_st
    globals_["render_settings_sidebar"] = lambda: (None, None)
    globals_["apply_theme"] = lambda _path: None
    globals_["_render_config_chat_sidebar"] = lambda: None
    globals_["_render_progress_bar"] = lambda current: calls.append(("progress", current))
    for number in range(1, 5):
        globals_[
            f"_render_step_{number}_{['analysis_mode', 'capital', 'returns_risk', 'correlations'][number - 1]}"
        ] = (lambda config, number=number: calls.append(("renderer", number)) or config)
    globals_["_render_step_5_review"] = lambda _config: calls.append(("renderer", 5)) or False

    module["main"]()

    assert calls == [("progress", step), ("renderer", step)]
    assert fake_st.session_state.wizard_step == step
