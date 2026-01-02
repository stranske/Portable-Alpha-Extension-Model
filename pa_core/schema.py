from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticUndefined

from .share_utils import SHARE_MAX, SHARE_MIN, SHARE_SUM_TOLERANCE, normalize_share

CORRELATION_LOWER_BOUND = -0.999
CORRELATION_UPPER_BOUND = 0.999


class Index(BaseModel):
    id: str
    label: str | None = None
    mu: float
    sigma: float


class Asset(BaseModel):
    id: str
    label: str | None = None
    mu: float
    sigma: float


class Correlation(BaseModel):
    pair: Tuple[str, str]
    rho: float

    @field_validator("rho")
    @classmethod
    def _check_rho(cls, v: float) -> float:
        if not CORRELATION_LOWER_BOUND <= v <= CORRELATION_UPPER_BOUND:
            raise ValueError(
                f"rho must be between {CORRELATION_LOWER_BOUND} and {CORRELATION_UPPER_BOUND}"
            )
        return v


class Portfolio(BaseModel):
    id: str
    weights: Dict[str, float]

    @model_validator(mode="after")
    def _check_weights(self) -> "Portfolio":
        total = sum(self.weights.values())
        if abs(total - 1.0) > SHARE_SUM_TOLERANCE:
            raise ValueError("portfolio weights must sum to 1")
        return self


class Sleeve(BaseModel):
    """Capital and alpha share metadata for a sleeve within a Scenario."""

    alpha_source: str
    capital_share: float
    theta: float | None = None
    active_share: float | None = None

    @field_validator("capital_share", "theta", "active_share", mode="before")
    @classmethod
    def _normalize_share_inputs(cls, value: float | None) -> float | None:
        return normalize_share(value)

    @model_validator(mode="after")
    def _check_share_bounds(self) -> "Sleeve":
        for name, value in (
            ("capital_share", self.capital_share),
            ("theta", self.theta),
            ("active_share", self.active_share),
        ):
            if value is None:
                continue
            if not SHARE_MIN <= value <= SHARE_MAX:
                raise ValueError(f"{name} must be between 0 and 1")
        return self


class Scenario(BaseModel):
    """Market data and portfolio structure for a single simulation run.

    Use ``Scenario`` to define index/asset inputs, correlations, and sleeves.
    Use :class:`pa_core.config.ModelConfig` to define simulation parameters
    such as run length, capital allocation, and risk metrics. They intentionally
    serve different roles: ``Scenario`` supplies market inputs and portfolio
    structure, while ``ModelConfig`` controls how the simulation runs. Pair
    with :func:`pa_core.config.load_config` for a full simulation setup.
    """

    index: Index
    assets: List[Asset] = Field(default_factory=list)
    correlations: List[Correlation] = Field(default_factory=list)
    portfolios: List[Portfolio] = Field(default_factory=list)
    sleeves: Dict[str, Sleeve] | None = None

    @model_validator(mode="after")
    def _check_assets_and_portfolios(self) -> "Scenario":
        asset_ids = [a.id for a in self.assets]
        dup_assets = [i for i, c in Counter(asset_ids).items() if c > 1]
        if dup_assets:
            raise ValueError(f"duplicate asset ids: {sorted(dup_assets)}")

        port_ids = [p.id for p in self.portfolios]
        dup_ports = [i for i, c in Counter(port_ids).items() if c > 1]
        if dup_ports:
            raise ValueError(f"duplicate portfolio ids: {sorted(dup_ports)}")

        asset_id_set = set(asset_ids)
        for p in self.portfolios:
            unknown = set(p.weights) - asset_id_set
            if unknown:
                raise ValueError(f"portfolio {p.id} references unknown assets: {sorted(unknown)}")
        return self

    @model_validator(mode="after")
    def _check_correlations(self) -> "Scenario":
        ids = [self.index.id] + [a.id for a in self.assets]
        expected = {tuple(sorted(p)) for p in combinations(ids, 2)}
        pairs = [tuple(sorted(c.pair)) for c in self.correlations]
        dupes = [p for p, count in Counter(pairs).items() if count > 1]
        if dupes:
            raise ValueError(f"duplicate correlations for pairs: {sorted(dupes)}")
        provided = set(pairs)
        missing = expected - provided
        if missing:
            raise ValueError(f"missing correlations for pairs: {sorted(missing)}")
        extra = provided - expected
        if extra:
            raise ValueError(f"unexpected correlations for pairs: {sorted(extra)}")
        return self

    @model_validator(mode="after")
    def _check_sleeves(self) -> "Scenario":
        if self.sleeves:
            total = sum(s.capital_share for s in self.sleeves.values())
            if abs(total - 1.0) > SHARE_SUM_TOLERANCE:
                raise ValueError("sleeves capital_share must sum to 1")
        return self


def load_scenario(path: str | Path) -> Scenario:
    """Return ``Scenario`` parsed from YAML file.

    Use :class:`pa_core.config.ModelConfig` for run-level settings such as
    simulation length, capital allocation, and risk metrics.
    """
    data = yaml.safe_load(Path(path).read_text())
    return Scenario.model_validate(data)


def save_scenario(scenario: Scenario, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(scenario.model_dump()))


def _stringify_type(annotation: Any) -> str:
    if annotation is None:
        return "Any"
    return str(annotation).replace("typing.", "")


def _validation_aliases(field_info: Any) -> list[str]:
    validation_alias = getattr(field_info, "validation_alias", None)
    if validation_alias is None:
        return []
    if isinstance(validation_alias, AliasChoices):
        return [str(choice) for choice in validation_alias.choices]
    return [str(validation_alias)]


def _export_model_fields(model_class: type[BaseModel]) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for field_name, field_info in model_class.model_fields.items():
        alias_value = field_info.alias or field_name
        validation_aliases = _validation_aliases(field_info)
        aliases = [alias_value] + [
            candidate for candidate in validation_aliases if candidate != alias_value
        ]
        default_factory = field_info.default_factory
        default_factory_name = (
            None
            if default_factory is None
            else getattr(default_factory, "__name__", repr(default_factory))
        )
        is_required = field_info.is_required()
        default_value = None
        if not is_required and field_info.default is not PydanticUndefined:
            default_value = field_info.default
        fields[field_name] = {
            "alias": alias_value,
            "aliases": aliases,
            "required": is_required,
            "default": default_value,
            "default_factory": default_factory_name,
            "type": _stringify_type(field_info.annotation),
            "description": field_info.description,
            "exclude": field_info.exclude,
        }
    return fields


def export_schema_definitions(schema: str = "all") -> dict[str, Any]:
    """Return field/alias definitions for supported schemas."""
    from .config import AgentConfig, ModelConfig

    scenario_models = [Index, Asset, Correlation, Portfolio, Sleeve, Scenario]
    config_models = [AgentConfig, ModelConfig]
    if schema == "scenario":
        models = scenario_models
    elif schema == "config":
        models = config_models
    elif schema == "all":
        models = config_models + scenario_models
    else:
        raise ValueError(f"Unknown schema selection: {schema}")

    payload: dict[str, Any] = {"models": {}}
    for model in models:
        payload["models"][model.__name__] = {
            "fields": _export_model_fields(cast(type[BaseModel], model)),
        }
    return payload


def _dump_payload(payload: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        import json

        return json.dumps(payload, indent=2, sort_keys=True)
    if fmt == "yaml":
        return yaml.safe_dump(payload, sort_keys=True)
    raise ValueError(f"Unsupported export format: {fmt}")


def main(argv: list[str] | None = None) -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Scenario schema utilities")
    parser.add_argument("--export", action="store_true", help="Export schema definitions")
    parser.add_argument(
        "--schema",
        choices=["all", "config", "scenario"],
        default="all",
        help="Schema subset to export",
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="Export format",
    )
    parser.add_argument("--output", help="Write export to a file instead of stdout")
    args = parser.parse_args(argv)

    if not args.export:
        parser.print_help()
        sys.exit(2)

    payload = export_schema_definitions(schema=args.schema)
    output = _dump_payload(payload, args.format)
    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output)


if __name__ == "__main__":  # pragma: no cover
    main()
