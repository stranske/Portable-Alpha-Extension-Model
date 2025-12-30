from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

from pa_core.validators import ValidationResult


class _FakeExpander:
    def __init__(self, calls, label: str, expanded: bool) -> None:
        self._calls = calls
        self._calls.append(("expander", label, expanded))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeSidebar:
    def __init__(self, calls, checkbox_values: dict[str, bool] | None = None) -> None:
        self._calls = calls
        self._checkbox_values = checkbox_values or {}

    def subheader(self, message: str) -> None:
        self._calls.append(("sidebar_subheader", message))

    def checkbox(self, label: str, value: bool, help: str | None = None) -> bool:
        self._calls.append(("sidebar_checkbox", label, value, help))
        return self._checkbox_values.get(label, value)


class FakeStreamlit(ModuleType):
    def __init__(self, name: str, sidebar_values: dict[str, bool] | None = None) -> None:
        super().__init__(name)
        self.calls: list[tuple] = []
        self.sidebar = _FakeSidebar(self.calls, sidebar_values)

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

    def metric(
        self, label: str, value: str, delta: str | None = None, help: str | None = None
    ) -> None:
        self.calls.append(("metric", label, value, delta, help))

    def columns(self, count: int):
        self.calls.append(("columns", count))
        return [_FakeColumn() for _ in range(count)]

    def expander(self, label: str, expanded: bool = False) -> _FakeExpander:
        return _FakeExpander(self.calls, label, expanded)

    def json(self, payload) -> None:
        self.calls.append(("json", payload))

    def write(self, *args, **kwargs) -> None:
        self.calls.append(("write", args, kwargs))


def _load_validation_ui(monkeypatch, sidebar_values: dict[str, bool] | None = None):
    fake_st = FakeStreamlit("streamlit", sidebar_values=sidebar_values)
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    module = importlib.import_module("dashboard.validation_ui")
    module = importlib.reload(module)
    return module, fake_st


def test_display_validation_results_empty(monkeypatch) -> None:
    module, fake_st = _load_validation_ui(monkeypatch)

    module.display_validation_results([])

    assert ("success", "✅ All validations passed!") in fake_st.calls


def test_display_validation_results_severity_sections(monkeypatch) -> None:
    module, fake_st = _load_validation_ui(monkeypatch)

    results = [
        ValidationResult(
            is_valid=False,
            message="Bad correlation",
            severity="error",
            details={"rho": 1.5},
        ),
        ValidationResult(
            is_valid=True,
            message="High correlation",
            severity="warning",
            details={"rho": 0.96},
        ),
        ValidationResult(
            is_valid=True,
            message="Informational",
            severity="info",
            details={"note": "ok"},
        ),
    ]

    module.display_validation_results(results)

    call_types = {call[0] for call in fake_st.calls}
    assert "error" in call_types
    assert "warning" in call_types
    assert "info" in call_types
    assert "expander" in call_types
    assert "json" in call_types


def test_create_validation_sidebar(monkeypatch) -> None:
    module, fake_st = _load_validation_ui(
        monkeypatch,
        sidebar_values={
            "Real-time validation": False,
            "Show validation details": True,
            "Show warnings": False,
        },
    )

    settings = module.create_validation_sidebar()

    assert settings == {
        "validate_on_change": False,
        "show_details": True,
        "show_warnings": False,
    }
    assert ("sidebar_subheader", "🔍 Validation Settings") in fake_st.calls


def test_validate_scenario_config_disabled(monkeypatch) -> None:
    module, _ = _load_validation_ui(monkeypatch)

    results = module.validate_scenario_config({}, {"validate_on_change": False})

    assert results == []


def test_validate_scenario_config_filters_and_exception(monkeypatch) -> None:
    module, _ = _load_validation_ui(monkeypatch)

    def fake_validate_correlations(correlations):
        assert "rho_idx_H" in correlations
        return [
            ValidationResult(
                is_valid=True,
                message="High correlation",
                severity="warning",
                details={},
            )
        ]

    def fake_validate_capital_allocation(**kwargs):
        raise RuntimeError("boom")

    def fake_validate_simulation_parameters(n_simulations, step_sizes=None):
        assert n_simulations == 10
        assert step_sizes == {"external_step_size_pct": 0.05}
        return [
            ValidationResult(
                is_valid=True,
                message="Informational",
                severity="info",
                details={},
            )
        ]

    monkeypatch.setattr(module, "validate_correlations", fake_validate_correlations)
    monkeypatch.setattr(module, "validate_capital_allocation", fake_validate_capital_allocation)
    monkeypatch.setattr(
        module,
        "validate_simulation_parameters",
        fake_validate_simulation_parameters,
    )

    config_data = {
        "rho_idx_H": 1.1,
        "external_pa_capital": 100.0,
        "active_ext_capital": 200.0,
        "internal_pa_capital": 700.0,
        "total_fund_capital": 1000.0,
        "N_SIMULATIONS": 10,
        "external_step_size_pct": 0.05,
    }
    settings = {"validate_on_change": True, "show_warnings": False}

    results = module.validate_scenario_config(config_data, settings)

    assert all(result.severity != "warning" for result in results)
    assert any(result.severity == "info" for result in results)
    assert any(
        result.severity == "error" and "Capital validation failed" in result.message
        for result in results
    )


def test_display_psd_projection_info(monkeypatch) -> None:
    module, fake_st = _load_validation_ui(monkeypatch)

    module.display_psd_projection_info({"was_projected": False})
    assert ("success", "✅ Covariance matrix is positive semidefinite") in fake_st.calls

    fake_st.calls.clear()
    module.display_psd_projection_info(
        {
            "was_projected": True,
            "max_eigenvalue_delta": 0.1,
            "max_delta": 0.2,
            "original_min_eigenvalue": -0.1,
            "projected_min_eigenvalue": 0.0,
        }
    )

    call_types = {call[0] for call in fake_st.calls}
    assert "warning" in call_types
    assert "metric" in call_types
    assert "expander" in call_types
    assert "write" in call_types
    assert "info" in call_types


@pytest.mark.parametrize(
    "available_buffer,expected_call",
    [
        (-10.0, "error"),
        (50.0, "warning"),
        (150.0, "info"),
        (300.0, "success"),
    ],
)
def test_create_margin_buffer_display_branches(
    monkeypatch, available_buffer: float, expected_call: str
) -> None:
    module, fake_st = _load_validation_ui(monkeypatch)

    module.create_margin_buffer_display(
        margin_requirement=10.0,
        available_buffer=available_buffer,
        total_capital=1000.0,
    )

    assert any(call[0] == expected_call for call in fake_st.calls)


def test_validation_status_indicator(monkeypatch) -> None:
    module, _ = _load_validation_ui(monkeypatch)

    assert module.validation_status_indicator([]) == "✅"
    assert (
        module.validation_status_indicator(
            [
                ValidationResult(
                    is_valid=True,
                    message="Warn",
                    severity="warning",
                    details={},
                )
            ]
        )
        == "⚠️"
    )
    assert (
        module.validation_status_indicator(
            [
                ValidationResult(
                    is_valid=False,
                    message="Error",
                    severity="error",
                    details={},
                )
            ]
        )
        == "❌"
    )
