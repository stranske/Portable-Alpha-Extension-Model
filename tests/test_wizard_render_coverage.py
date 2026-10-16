import runpy
from pathlib import Path
from typing import Any

import pytest

from pa_core.wizard_schema import AnalysisMode, get_default_config


class _Container:
    def __init__(self, calls: list[tuple[str, Any]] | None = None) -> None:
        self.calls = calls

    def __enter__(self) -> "_Container":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def metric(self, value: str, detail: str, *args: Any, **kwargs: Any) -> None:
        if self.calls is not None:
            self.calls.append(("metric", (value, detail)))


class _SessionState(dict[str, Any]):
    def __getattr__(self, key: str) -> Any:
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class _FakeStreamlit:
    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.session_state = _SessionState()
        self.values = values or {}
        self.calls: list[tuple[str, Any]] = []

    def columns(self, count: int, *args: Any, **kwargs: Any) -> list[_Container]:
        return [_Container(self.calls) for _ in range(count)]

    def expander(self, *args: Any, **kwargs: Any) -> _Container:
        return _Container()

    def number_input(self, label: str, *args: Any, **kwargs: Any) -> Any:
        value = self.values.get(label, kwargs.get("value"))
        if key := kwargs.get("key"):
            self.session_state[key] = value
        return value

    def slider(self, label: str, *args: Any, **kwargs: Any) -> Any:
        return self.number_input(label, *args, **kwargs)

    def selectbox(self, label: str, options: list[Any], index: int = 0, **kwargs: Any) -> Any:
        value = self.values.get(label, options[index])
        if key := kwargs.get("key"):
            self.session_state[key] = value
        return value

    def file_uploader(self, *args: Any, **kwargs: Any) -> None:
        return None

    def checkbox(self, label: str, *args: Any, **kwargs: Any) -> bool:
        value = bool(self.values.get(label, kwargs.get("value", False)))
        if key := kwargs.get("key"):
            self.session_state[key] = value
        return value

    def button(self, label: str, *args: Any, **kwargs: Any) -> bool:
        return bool(self.values.get(label, False))

    def markdown(self, value: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("markdown", value))

    def progress(self, value: float, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("progress", value))

    def success(self, value: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("success", value))

    def info(self, value: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("info", value))

    def warning(self, value: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("warning", value))

    def error(self, value: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("error", value))

    def write(self, value: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("write", value))

    def subheader(self, value: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("subheader", value))

    def plotly_chart(self, value: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        self.calls.append(("plotly_chart", value))

    def metric(self, value: str, detail: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("metric", (value, detail)))

    def dataframe(self, value: Any, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("dataframe", value))


WIZARD_PAGE = Path(__file__).resolve().parents[1] / "dashboard" / "pages" / "3_Scenario_Wizard.py"


@pytest.fixture
def wizard_module() -> dict[str, Any]:
    # A fresh namespace per test (so a patched `st` cannot leak), found from this file, not from the working directory.
    return runpy.run_path(str(WIZARD_PAGE))


def test_progress_bar_marks_completed_current_and_future_steps(wizard_module: dict[str, Any]) -> None:
    fake_st = _FakeStreamlit()
    wizard_module["_render_progress_bar"].__globals__["st"] = fake_st

    wizard_module["_render_progress_bar"](2)

    assert ("progress", 0.4) in fake_st.calls
    assert ("success", "✅ 1. Analysis Mode") in fake_st.calls
    assert ("info", "📍 2. Capital Allocation") in fake_st.calls
    assert ("write", "⭕ 5. Review & Run") in fake_st.calls


@pytest.mark.parametrize(
    ("internal_capital", "expected_status"),
    [(50.0, "error"), (60.0, "success")],
)
def test_capital_step_reports_allocation_status_and_balanced_chart(
    wizard_module: dict[str, Any],
    internal_capital: float,
    expected_status: str,
) -> None:
    fake_st = _FakeStreamlit(
        {
            "Total Fund Capital": 100.0,
            "External PA Capital [$M]": 20.0,
            "Active Extension Capital [$M]": 20.0,
            "Internal PA Capital [$M]": internal_capital,
        }
    )
    fake_st.session_state.financing_settings = {
        "financing_model": "simple_proxy",
        "reference_sigma": 0.01,
        "volatility_multiple": 3.0,
        "term_months": 1.0,
        "schedule_path": None,
    }
    func_globals = wizard_module["_render_step_2_capital"].__globals__
    func_globals["st"] = fake_st
    func_globals["calculate_margin_requirement"] = lambda **_kwargs: 5.0
    config = get_default_config(AnalysisMode.RETURNS)

    rendered = wizard_module["_render_step_2_capital"](config)

    assert rendered is config
    assert any(kind == "success" and "Buffer after margin" in text for kind, text in fake_st.calls)
    assert any(kind == expected_status and "Capital allocation" in text for kind, text in fake_st.calls)
    if internal_capital == 60.0:
        chart = next(value for kind, value in fake_st.calls if kind == "plotly_chart")
        assert chart["data"][0]["values"] == [20.0, 20.0, 60.0]
    else:
        assert not any(kind == "plotly_chart" for kind, _value in fake_st.calls)


@pytest.mark.parametrize("uploaded", [True, False])
def test_capital_step_handles_uploaded_and_persisted_margin_schedules(
    wizard_module: dict[str, Any], tmp_path: Any, uploaded: bool
) -> None:
    schedule_path = tmp_path / "schedule.csv"
    schedule_path.write_text("term,multiplier\n1,3\n")
    fake_st = _FakeStreamlit({"Financing model": "schedule"})
    fake_st.session_state.financing_settings = {
        "financing_model": "schedule",
        "reference_sigma": 0.01,
        "volatility_multiple": 3.0,
        "term_months": 1.0,
        "schedule_path": str(schedule_path),
    }
    if uploaded:
        fake_st.file_uploader = lambda *args, **kwargs: type(
            "Upload", (), {"getvalue": lambda self: b"term,multiplier\n1,3\n"}
        )()
    loaded = wizard_module["pd"].DataFrame({"term": [1.0], "multiplier": [3.0]})
    margin_calls: list[dict[str, Any]] = []

    def fake_margin(**kwargs: Any) -> float:
        margin_calls.append(kwargs)
        return 5.0

    func_globals = wizard_module["_render_step_2_capital"].__globals__
    func_globals["st"] = fake_st
    func_globals["load_margin_schedule"] = lambda _path: loaded
    func_globals["calculate_margin_requirement"] = fake_margin

    wizard_module["_render_step_2_capital"](get_default_config(AnalysisMode.RETURNS))

    # Every margin computation on this page uses the schedule model with the LOADED schedule; the last one is the
    # interpolation the "Interpolated k" metric is derived from.
    assert margin_calls
    assert all(call["financing_model"] == "schedule" for call in margin_calls)
    assert all(call["margin_schedule"] is loaded for call in margin_calls)
    interpolation = margin_calls[-1]
    assert interpolation["term_months"] == 1.0
    assert interpolation["reference_sigma"] == 0.01
    # k = margin / (reference_sigma * total capital): the metric carries that value, not merely its label.
    expected_k = 5.0 / (0.01 * interpolation["total_capital"])
    assert ("metric", ("Interpolated k (multiplier)", f"{expected_k:.2f}")) in fake_st.calls
    assert ("metric", ("Estimated Margin Requirement", "$5.0M")) in fake_st.calls
    tables = [value for kind, value in fake_st.calls if kind == "dataframe"]
    assert len(tables) == 1 and tables[0] is loaded
    if uploaded:
        assert ("success", "Schedule validated ✓") in fake_st.calls
    else:
        assert any(kind == "info" and "Using schedule at" in value for kind, value in fake_st.calls)


@pytest.mark.parametrize("suggestions", [None, "empty"])
def test_sleeve_suggestor_explains_missing_or_empty_results(
    wizard_module: dict[str, Any], suggestions: str | None
) -> None:
    fake_st = _FakeStreamlit()
    if suggestions == "empty":
        fake_st.session_state["sleeve_suggestions"] = wizard_module["pd"].DataFrame()
    func_globals = wizard_module["_render_sleeve_suggestor"].__globals__
    func_globals["st"] = fake_st

    wizard_module["_render_sleeve_suggestor"](get_default_config(AnalysisMode.RETURNS))

    if suggestions is None:
        assert not any(kind == "warning" for kind, _value in fake_st.calls)
        # Positive evidence that the renderer reached the end of its inputs (the last widget before the "Run Suggestor"
        # button stored its value) and that no suggestions were produced or stored.
        assert ("subheader", "Sleeve Suggestor") in fake_st.calls
        assert fake_st.session_state["sleeve_validate_on_run"] is False
        assert "sleeve_suggestions" not in fake_st.session_state
    else:
        assert ("warning", "No feasible sleeve allocations found.") in fake_st.calls


def test_margin_schedule_loader_hides_invalid_schedule_errors(
    wizard_module: dict[str, Any], tmp_path: Any
) -> None:
    schedule_path = tmp_path / "invalid.csv"
    schedule_path.write_text("not,a,schedule\n")
    func_globals = wizard_module["_load_margin_schedule_from_path"].__globals__
    func_globals["load_margin_schedule"] = lambda _path: (_ for _ in ()).throw(ValueError("bad"))

    schedule, resolved_path = wizard_module["_load_margin_schedule_from_path"](schedule_path)

    assert schedule is None
    assert resolved_path is None
