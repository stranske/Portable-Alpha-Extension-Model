from typing import Any, Dict, List

import pytest

from dashboard import validation_ui
from pa_core.validators import ValidationResult


class _FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeColumn(_FakeContext):
    pass


class _FakeSidebar:
    def __init__(self, checkbox_values: List[bool], calls: List[tuple[str, Any]]):
        self._checkbox_values = list(checkbox_values)
        self._calls = calls

    def subheader(self, message: str) -> None:
        self._calls.append(("sidebar_subheader", message))

    def checkbox(self, label: str, value: bool, help: str) -> bool:  # noqa: A002
        self._calls.append(("sidebar_checkbox", label))
        if self._checkbox_values:
            return self._checkbox_values.pop(0)
        return value


class FakeStreamlit:
    def __init__(self, checkbox_values: List[bool] | None = None):
        self.calls: List[tuple[str, Any]] = []
        self.sidebar = _FakeSidebar(checkbox_values or [], self.calls)

    def success(self, message: str) -> None:
        self.calls.append(("success", message))

    def error(self, message: str) -> None:
        self.calls.append(("error", message))

    def warning(self, message: str) -> None:
        self.calls.append(("warning", message))

    def info(self, message: str) -> None:
        self.calls.append(("info", message))

    def subheader(self, message: str) -> None:
        self.calls.append(("subheader", message))

    def json(self, payload: Dict[str, Any]) -> None:
        self.calls.append(("json", payload))

    def write(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("write", args))

    def metric(
        self, label: str, value: str, delta: str | None = None, help: str | None = None
    ) -> None:  # noqa: A002
        self.calls.append(("metric", label))

    def columns(self, count: int) -> List[_FakeColumn]:
        self.calls.append(("columns", count))
        return [_FakeColumn() for _ in range(count)]

    def expander(self, label: str, expanded: bool = False) -> _FakeContext:
        self.calls.append(("expander", label))
        return _FakeContext()


@pytest.fixture()
def fake_st(monkeypatch: pytest.MonkeyPatch) -> FakeStreamlit:
    fake = FakeStreamlit()
    monkeypatch.setattr(validation_ui, "st", fake)
    return fake


def test_display_validation_results_empty(fake_st: FakeStreamlit) -> None:
    validation_ui.display_validation_results([])
    assert any(call[0] == "success" for call in fake_st.calls)


def test_display_validation_results_mixed(fake_st: FakeStreamlit) -> None:
    results = [
        ValidationResult(False, "bad", "error", {"field": "x"}),
        ValidationResult(True, "warn", "warning", {}),
        ValidationResult(True, "info", "info", {}),
    ]
    validation_ui.display_validation_results(results)
    call_types = {call[0] for call in fake_st.calls}
    assert "error" in call_types
    assert "warning" in call_types
    assert "info" in call_types
    assert "expander" in call_types


def test_create_validation_sidebar_returns_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeStreamlit([False, True, False])
    monkeypatch.setattr(validation_ui, "st", fake)

    settings = validation_ui.create_validation_sidebar()

    assert settings == {
        "validate_on_change": False,
        "show_details": True,
        "show_warnings": False,
    }


def test_validate_scenario_config_filters_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_validate_correlations(_: Dict[str, float]) -> List[ValidationResult]:
        return [ValidationResult(False, "bad", "error", {})]

    def fake_validate_capital_allocation(**_: Any) -> List[ValidationResult]:
        return [ValidationResult(True, "warn", "warning", {})]

    def fake_validate_simulation_parameters(**_: Any) -> List[ValidationResult]:
        return [ValidationResult(True, "info", "info", {})]

    monkeypatch.setattr(
        validation_ui, "validate_correlations", fake_validate_correlations
    )
    monkeypatch.setattr(
        validation_ui, "validate_capital_allocation", fake_validate_capital_allocation
    )
    monkeypatch.setattr(
        validation_ui,
        "validate_simulation_parameters",
        fake_validate_simulation_parameters,
    )

    config = {
        "rho_idx_H": 1.2,
        "external_pa_capital": 100.0,
        "active_ext_capital": 200.0,
        "internal_pa_capital": 300.0,
        "N_SIMULATIONS": 100,
    }
    settings = {"validate_on_change": True, "show_warnings": False}

    results = validation_ui.validate_scenario_config(config, settings)

    severities = {result.severity for result in results}
    assert "warning" not in severities
    assert severities == {"error", "info"}


def test_display_psd_projection_info_branches(fake_st: FakeStreamlit) -> None:
    validation_ui.display_psd_projection_info({"was_projected": False})
    assert any(call[0] == "success" for call in fake_st.calls)

    fake_st.calls.clear()
    validation_ui.display_psd_projection_info(
        {
            "was_projected": True,
            "max_delta": 0.1,
            "max_eigenvalue_delta": 0.2,
            "original_min_eigenvalue": -0.3,
            "projected_min_eigenvalue": 0.0,
        }
    )
    assert any(call[0] == "warning" for call in fake_st.calls)
    assert any(call[0] == "metric" for call in fake_st.calls)


def test_create_margin_buffer_display_branches(fake_st: FakeStreamlit) -> None:
    validation_ui.create_margin_buffer_display(1100.0, -50.0, total_capital=1000.0)
    assert any(call[0] == "error" for call in fake_st.calls)

    fake_st.calls.clear()
    validation_ui.create_margin_buffer_display(900.0, 50.0, total_capital=1000.0)
    assert any(call[0] == "warning" for call in fake_st.calls)

    fake_st.calls.clear()
    validation_ui.create_margin_buffer_display(900.0, 150.0, total_capital=1000.0)
    assert any(call[0] == "info" for call in fake_st.calls)

    fake_st.calls.clear()
    validation_ui.create_margin_buffer_display(900.0, 500.0, total_capital=1000.0)
    assert any(call[0] == "success" for call in fake_st.calls)


def test_validation_status_indicator() -> None:
    assert validation_ui.validation_status_indicator([]) == "✅"
    assert (
        validation_ui.validation_status_indicator(
            [ValidationResult(False, "bad", "error", {})]
        )
        == "❌"
    )
    assert (
        validation_ui.validation_status_indicator(
            [ValidationResult(True, "warn", "warning", {})]
        )
        == "⚠️"
    )
