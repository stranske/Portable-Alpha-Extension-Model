from __future__ import annotations

from pa_core.config import ModelConfig
from pa_core.units import CONFIG_TIME_HORIZON_FIELDS


def _collect_time_sensitive_fields() -> set[str]:
    keyword_fields: set[str] = set()
    for name, field in ModelConfig.model_fields.items():
        alias = str(field.alias or "")
        description = str(field.description or "")
        text = f"{name} {alias} {description}".lower()
        if any(token in text for token in ("annual", "monthly", "month", "months")):
            keyword_fields.add(name)
            continue
        if name.endswith(("_month", "_months")):
            keyword_fields.add(name)

    explicit_fields = {
        "N_MONTHS",
        "return_unit",
        "return_unit_input",
        "mu_H",
        "sigma_H",
        "mu_E",
        "sigma_E",
        "mu_M",
        "sigma_M",
        "vol_regime_window",
        "reference_sigma",
        "financing_term_months",
    }
    return keyword_fields | explicit_fields


def test_config_time_horizon_fields_are_covered() -> None:
    audited = _collect_time_sensitive_fields()
    missing = audited - set(CONFIG_TIME_HORIZON_FIELDS)
    assert not missing, f"Missing time-horizon fields in units audit: {sorted(missing)}"
