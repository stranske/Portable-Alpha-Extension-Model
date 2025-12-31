from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml

from ..config import get_field_mappings, load_config
from .loaders import load_parameters


def convert(csv_path: str | Path, yaml_path: str | Path) -> None:
    """Convert legacy parameters CSV to YAML configuration."""

    # Get the field mappings from the ModelConfig Pydantic model
    label_map = get_field_mappings()

    raw = load_parameters(csv_path, label_map)
    cfg = load_config(raw)
    Path(yaml_path).write_text(yaml.safe_dump(cfg.model_dump()))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert parameters CSV to YAML")
    parser.add_argument("csv", help="Input parameters CSV file")
    parser.add_argument("yaml", nargs="?", help="Output YAML file (default: params.yml)")
    args = parser.parse_args(argv)
    out = args.yaml or Path(args.csv).with_suffix(".yml")
    convert(args.csv, out)


if __name__ == "__main__":  # pragma: no cover
    main()
