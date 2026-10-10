"""Protect validation-only failure diagnostics and the configuration boundary."""

import logging
import warnings

import pytest

from pa_core import cli, validators
from pa_core.config import ModelConfig


@pytest.fixture
def validation_config(monkeypatch):
    cfg = ModelConfig(N_SIMULATIONS=100, N_MONTHS=1, financing_mode="broadcast")
    monkeypatch.setattr("pa_core.config.load_config", lambda _: cfg)
    # Inject failures into the CLI's validation pass after model validation.
    # Facade option revalidation uses the same validator functions earlier.
    monkeypatch.setattr("pa_core.facade.apply_run_options", lambda config, _: config)
    return cfg


def invoke_validation():
    root = logging.getLogger()
    handler = logging.NullHandler()
    original_showwarning = warnings.showwarning

    def caller_showwarning(*args, **kwargs):
        return original_showwarning(*args, **kwargs)

    root.addHandler(handler)
    handlers = tuple(root.handlers)
    warnings.showwarning = caller_showwarning
    try:
        exit_code = None
        try:
            cli.main(["--config", "cfg.yaml", "--validate-only"], emit_deprecation_warning=False)
        except SystemExit as exc:
            exit_code = exc.code
        assert tuple(root.handlers) == handlers, "validation changed caller logging handlers"
        assert (
            warnings.showwarning is caller_showwarning
        ), "validation changed caller warning handler"
        return exit_code
    finally:
        root.removeHandler(handler)
        warnings.showwarning = original_showwarning


def test_capital_exception_is_a_failed_result_with_original_details(
    validation_config, monkeypatch, capsys
):
    captured = []
    original_formatter = validators.format_validation_messages

    def fail_capital(**kwargs):
        raise RuntimeError("margin schedule unavailable")

    def format_results(results):
        captured.extend(results)
        return original_formatter(results)

    monkeypatch.setattr(validators, "validate_capital_allocation", fail_capital)
    monkeypatch.setattr(validators, "format_validation_messages", format_results)
    assert invoke_validation() == 1, "capital validator exceptions must reject validation"
    errors = [result for result in captured if not result.is_valid and result.severity == "error"]
    assert len(errors) == 1
    assert errors[0].message == "Capital validation failed: margin schedule unavailable"
    assert errors[0].details == {
        "exception": "margin schedule unavailable"
    }, "exception detail lost"
    output = capsys.readouterr().out
    assert "Capital validation failed: margin schedule unavailable" in output
    assert "Validation completed successfully." not in output


@pytest.mark.parametrize(
    "severity,expected_exit", [("warning", None), ("info", None), ("error", 1)]
)
def test_validation_only_rejects_errors_without_rejecting_advisories(
    validation_config, monkeypatch, capsys, severity, expected_exit
):
    result = validators.ValidationResult(
        is_valid=False, severity=severity, message=f"boundary {severity}", details={}
    )
    monkeypatch.setattr(validators, "validate_correlations", lambda _: [result])
    assert invoke_validation() == expected_exit, "validation severity changed exit policy"
    output = capsys.readouterr().out
    assert f"boundary {severity}" in output
    assert ("Validation completed successfully." in output) is (expected_exit is None)


def test_validation_forwards_effective_financing_and_simulation_settings(
    validation_config, monkeypatch, tmp_path
):
    schedule = tmp_path / "margin.csv"
    schedule.write_text("term,multiplier\n1,2\n24,4\n")
    cfg = validation_config.model_copy(
        update={
            "financing_model": "schedule",
            "financing_schedule_path": schedule,
            "financing_term_months": 18,
            "N_SIMULATIONS": 321,
            "external_step_size_pct": 7.0,
            "in_house_return_step_pct": 3.0,
            "in_house_vol_step_pct": 0.5,
            "alpha_ext_return_step_pct": 4.0,
            "alpha_ext_vol_step_pct": 0.75,
            "external_pa_alpha_step_pct": 6.0,
            "active_share_step_pct": 8.0,
            "sd_multiple_step": 0.5,
        }
    )
    applied_configs = []

    def apply_options(config, options):
        applied_configs.append(config)
        return cfg

    # Keep loaded defaults distinct from the facade's effective configuration.
    monkeypatch.setattr("pa_core.facade.apply_run_options", apply_options)
    capital_calls, simulation_calls = [], []
    original_capital = validators.validate_capital_allocation
    original_simulation = validators.validate_simulation_parameters

    def capital(**kwargs):
        capital_calls.append(kwargs)
        return original_capital(**kwargs)

    def simulation(**kwargs):
        simulation_calls.append(kwargs)
        return original_simulation(**kwargs)

    monkeypatch.setattr(validators, "validate_capital_allocation", capital)
    monkeypatch.setattr(validators, "validate_simulation_parameters", simulation)
    assert invoke_validation() is None
    assert len(applied_configs) == 1
    assert applied_configs[0] is validation_config
    assert capital_calls == [
        {
            "external_pa_capital": cfg.external_pa_capital,
            "active_ext_capital": cfg.active_ext_capital,
            "internal_pa_capital": cfg.internal_pa_capital,
            "total_fund_capital": cfg.total_fund_capital,
            "reference_sigma": cfg.reference_sigma,
            "volatility_multiple": cfg.volatility_multiple,
            "financing_model": cfg.financing_model,
            "margin_schedule_path": cfg.financing_schedule_path,
            "term_months": 18,
        }
    ], "effective financing settings were not forwarded"
    keys = (
        "external_step_size_pct",
        "in_house_return_step_pct",
        "in_house_vol_step_pct",
        "alpha_ext_return_step_pct",
        "alpha_ext_vol_step_pct",
        "external_pa_alpha_step_pct",
        "active_share_step_pct",
        "sd_multiple_step",
    )
    assert simulation_calls == [
        {
            "n_simulations": cfg.N_SIMULATIONS,
            "step_sizes": {key: getattr(cfg, key) for key in keys},
        }
    ]
