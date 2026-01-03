from __future__ import annotations

import argparse
import runpy
from typing import Iterable

import streamlit as st

from pa_core.config import ModelConfig
from pa_core.wizard_schema import AnalysisMode, get_default_config

WIZARD_MODULE_PATH = "dashboard/pages/3_Scenario_Wizard.py"


def _model_config_fields() -> set[str]:
    fields: set[str] = set()
    for name, field in ModelConfig.model_fields.items():
        if field.exclude:
            continue
        fields.add(name)
    return fields


def _wizard_yaml_keys() -> set[str]:
    st.session_state.clear()
    module = runpy.run_path(WIZARD_MODULE_PATH)
    build_yaml = module["_build_yaml_from_config"]
    config = get_default_config(AnalysisMode.RETURNS)
    yaml_dict = build_yaml(config)
    st.session_state.clear()
    return set(yaml_dict.keys())


def compute_gap() -> dict[str, list[str]]:
    model_fields = _model_config_fields()
    yaml_keys = _wizard_yaml_keys()
    return {
        "wired": sorted(model_fields & yaml_keys),
        "missing": sorted(model_fields - yaml_keys),
        "extra": sorted(yaml_keys - model_fields),
    }


def _format_lines(gap: dict[str, list[str]], *, model_fields: Iterable[str], yaml_keys: Iterable[str]) -> str:
    model_fields = list(model_fields)
    yaml_keys = list(yaml_keys)
    lines = [
        "ModelConfig UI Gap Report",
        f"ModelConfig fields: {len(model_fields)}",
        f"Wizard YAML keys: {len(yaml_keys)}",
        f"Wired fields: {len(gap['wired'])}",
        f"Missing fields: {len(gap['missing'])}",
        f"Extra wizard keys: {len(gap['extra'])}",
    ]
    if gap["missing"]:
        lines.append("")
        lines.append("Missing ModelConfig fields:")
        lines.extend(f"- {name}" for name in gap["missing"])
    if gap["extra"]:
        lines.append("")
        lines.append("Extra wizard keys:")
        lines.extend(f"- {name}" for name in gap["extra"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report ModelConfig fields not exposed by the scenario wizard."
    )
    parser.add_argument("--output", type=str, default="", help="Optional output file path.")
    args = parser.parse_args()

    model_fields = _model_config_fields()
    yaml_keys = _wizard_yaml_keys()
    gap = compute_gap()
    report = _format_lines(gap, model_fields=model_fields, yaml_keys=yaml_keys)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(report)
            handle.write("\n")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
