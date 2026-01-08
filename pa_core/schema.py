from __future__ import annotations

import types
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union, cast, get_args, get_origin

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
        if self.index.id in asset_ids:
            raise ValueError(f"assets must not include index id {self.index.id!r}")
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


def _field_key(field_name: str, field_info: Any, *, use_aliases: bool) -> str:
    alias_value = field_info.alias or field_name
    return alias_value if use_aliases else field_name


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


def _placeholder_for_type(annotation: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return _build_model_template(annotation, use_aliases=False)
        if annotation is str:
            return "REPLACE_ME"
        if annotation is float:
            return 0.0
        if annotation is int:
            return 0
        if annotation is bool:
            return False
        return None
    if origin in (list, List):
        return []
    if origin in (dict, Dict):
        return {}
    if origin in (tuple, Tuple):
        if not args:
            return []
        return [_placeholder_for_type(arg) for arg in args]
    if origin in (Union, getattr(types, "UnionType", Union)):
        non_none = [arg for arg in args if arg is not type(None)]
        if non_none:
            return _placeholder_for_type(non_none[0])
        return None
    if origin is Literal:
        return args[0] if args else None
    return None


def _field_default(field_info: Any) -> Any:
    if field_info.default is not PydanticUndefined:
        return field_info.default
    if field_info.default_factory is not None:
        return field_info.default_factory()
    return _placeholder_for_type(field_info.annotation)


def _build_model_template(
    model_class: type[BaseModel],
    *,
    use_aliases: bool,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for field_name, field_info in model_class.model_fields.items():
        if field_info.exclude:
            continue
        key = _field_key(field_name, field_info, use_aliases=use_aliases)
        data[key] = _field_default(field_info)
    if overrides:
        data.update(overrides)
    return data


def _build_scenario_template(asset_count: int = 2) -> dict[str, Any]:
    index = _build_model_template(
        Index,
        use_aliases=False,
        overrides={"id": "INDEX", "label": "Sample Index", "mu": 0.07, "sigma": 0.15},
    )
    assets = []
    for idx in range(1, asset_count + 1):
        asset_id = f"ASSET_{idx}"
        assets.append(
            _build_model_template(
                Asset,
                use_aliases=False,
                overrides={
                    "id": asset_id,
                    "label": f"Sample Asset {idx}",
                    "mu": 0.04 + 0.01 * (idx - 1),
                    "sigma": 0.08 + 0.01 * (idx - 1),
                },
            )
        )
    correlations = []
    for asset in assets:
        correlations.append(
            _build_model_template(
                Correlation,
                use_aliases=False,
                overrides={"pair": [index["id"], asset["id"]], "rho": 0.15},
            )
        )
    if asset_count > 1:
        first = assets[0]["id"]
        second = assets[1]["id"]
        correlations.append(
            _build_model_template(
                Correlation,
                use_aliases=False,
                overrides={"pair": [first, second], "rho": 0.1},
            )
        )
    portfolios = []
    for idx, asset in enumerate(assets, start=1):
        portfolios.append(
            _build_model_template(
                Portfolio,
                use_aliases=False,
                overrides={"id": f"portfolio_{idx}", "weights": {asset["id"]: 1.0}},
            )
        )
    sleeves: dict[str, Any] = {}
    share = round(1.0 / max(len(portfolios), 1), 6)
    for idx, portfolio in enumerate(portfolios, start=1):
        sleeves[f"sleeve_{idx}"] = _build_model_template(
            Sleeve,
            use_aliases=False,
            overrides={
                "alpha_source": f"portfolio:{portfolio['id']}",
                "capital_share": share,
            },
        )
    return {
        "index": index,
        "assets": assets,
        "correlations": correlations,
        "portfolios": portfolios,
        "sleeves": sleeves,
    }


def _dump_yaml_template(data: dict[str, Any], header: str) -> str:
    body = yaml.safe_dump(data, sort_keys=False)
    return f"{header}\n{body}"


def _dump_csv_template(rows: list[tuple[str, str]], header: str) -> str:
    import csv
    import io

    output = io.StringIO()
    output.write(f"{header}\n")
    writer = csv.writer(output)
    writer.writerow(["Parameter", "Value"])
    writer.writerows(rows)
    return output.getvalue()


def generate_schema_templates(output_dir: str | Path) -> None:
    """Generate schema-driven YAML/CSV templates in the target directory."""
    from .config import ModelConfig

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = "# Generated by pa_core.schema --generate-templates. Do not edit by hand."

    scenario_full = _build_scenario_template(asset_count=2)
    scenario_min = _build_scenario_template(asset_count=1)
    (out_dir / "scenario_template.yaml").write_text(_dump_yaml_template(scenario_full, header))
    (out_dir / "scenario_example.yaml").write_text(_dump_yaml_template(scenario_min, header))

    config_overrides = {
        "Number of simulations": 1000,
        "Number of months": 12,
    }
    config_data = _build_model_template(
        ModelConfig,
        use_aliases=True,
        overrides=config_overrides,
    )
    (out_dir / "params_template.yaml").write_text(_dump_yaml_template(config_data, header))

    rows: list[tuple[str, str]] = []
    for field_name, field_info in ModelConfig.model_fields.items():
        if field_info.exclude:
            continue
        key = _field_key(field_name, field_info, use_aliases=True)
        value = _field_default(field_info)
        if field_name == "N_SIMULATIONS":
            value = config_overrides["Number of simulations"]
        elif field_name == "N_MONTHS":
            value = config_overrides["Number of months"]
        if field_name == "risk_metrics" and isinstance(value, list):
            rendered = ";".join(str(item) for item in value)
        elif isinstance(value, (list, dict)):
            rendered = ""
        elif value is None:
            rendered = ""
        else:
            rendered = str(value)
        rows.append((key, rendered))
    (out_dir / "parameters_template.csv").write_text(_dump_csv_template(rows, header))


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
        return str(yaml.safe_dump(payload, sort_keys=True))
    raise ValueError(f"Unsupported export format: {fmt}")


def _escape_markdown(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _format_default_for_markdown(field: dict[str, Any]) -> str:
    if field.get("required"):
        return ""
    default_factory = field.get("default_factory")
    if default_factory:
        return f"default_factory: {default_factory}"
    default = field.get("default")
    if default is None:
        return "None"
    if isinstance(default, (list, dict)):
        import json

        return json.dumps(default)
    return str(default)


def generate_parameter_dictionary(output_path: str | Path, schema: str = "config") -> None:
    """Generate a markdown parameter dictionary from schema definitions."""
    payload = export_schema_definitions(schema=schema)
    header = (
        "# Parameter Dictionary\n\n"
        "> Generated by `python -m pa_core.schema --generate-parameter-dictionary`. "
        "Do not edit by hand.\n\n"
        "This reference lists canonical field names, accepted aliases, and defaults "
        "for schema inputs.\n"
    )
    lines: list[str] = [header]
    for model_name, model_info in payload["models"].items():
        lines.append(f"\n## {model_name}\n")
        lines.append("| Field | Aliases | Type | Required | Default | Description |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for field_name, field in model_info["fields"].items():
            if field.get("exclude"):
                continue
            aliases = field.get("aliases") or []
            alias_text = ", ".join(f"`{_escape_markdown(alias)}`" for alias in aliases)
            type_text = field.get("type") or "Any"
            required_text = "yes" if field.get("required") else "no"
            default_text = _format_default_for_markdown(field)
            default_cell = f"`{_escape_markdown(default_text)}`" if default_text else ""
            description = field.get("description") or ""
            lines.append(
                "| "
                f"`{_escape_markdown(field_name)}` | "
                f"{alias_text} | "
                f"`{_escape_markdown(type_text)}` | "
                f"{required_text} | "
                f"{default_cell} | "
                f"{_escape_markdown(description)} |"
            )
    Path(output_path).write_text("\n".join(lines).rstrip() + "\n")


def main(argv: list[str] | None = None) -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Scenario schema utilities")
    parser.add_argument("--export", action="store_true", help="Export schema definitions")
    parser.add_argument(
        "--generate-templates",
        action="store_true",
        help="Generate YAML/CSV templates from schema definitions",
    )
    parser.add_argument(
        "--generate-parameter-dictionary",
        action="store_true",
        help="Generate a markdown parameter dictionary from schema definitions",
    )
    parser.add_argument(
        "--template-dir",
        default="templates",
        help="Output directory for generated templates",
    )
    parser.add_argument(
        "--dictionary-output",
        default="docs/guides/PARAMETER_DICTIONARY.md",
        help="Output path for parameter dictionary",
    )
    parser.add_argument(
        "--schema",
        choices=["all", "config", "scenario"],
        default="all",
        help="Schema subset to export",
    )
    parser.add_argument(
        "--dictionary-schema",
        choices=["all", "config", "scenario"],
        default="config",
        help="Schema subset to include in the parameter dictionary",
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="Export format",
    )
    parser.add_argument("--output", help="Write export to a file instead of stdout")
    args = parser.parse_args(argv)

    if not args.export and not args.generate_templates and not args.generate_parameter_dictionary:
        parser.print_help()
        sys.exit(2)

    if args.export:
        payload = export_schema_definitions(schema=args.schema)
        output = _dump_payload(payload, args.format)
        if args.output:
            Path(args.output).write_text(output)
        else:
            print(output)

    if args.generate_templates:
        generate_schema_templates(args.template_dir)

    if args.generate_parameter_dictionary:
        generate_parameter_dictionary(args.dictionary_output, schema=args.dictionary_schema)


if __name__ == "__main__":  # pragma: no cover
    main()
