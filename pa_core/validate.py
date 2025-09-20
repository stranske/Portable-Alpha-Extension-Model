from __future__ import annotations

import argparse
import importlib
import sys
from types import ModuleType

from pydantic import ValidationError

from .config import load_config
from .schema import load_scenario

yaml: ModuleType | None
try:
    yaml = importlib.import_module("yaml")
except ImportError:  # pragma: no cover - dependency optional
    yaml = None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate YAML against a schema")
    parser.add_argument(
        "--schema",
        choices=["scenario", "config"],
        default="scenario",
        help="Type of YAML file to validate",
    )
    parser.add_argument("path", help="YAML file")
    args = parser.parse_args(argv)
    try:
        if args.schema == "scenario":
            load_scenario(args.path)
        else:
            load_config(args.path)
    except (
        ValidationError,
        ValueError,
    ) as exc:  # pragma: no cover - reached via SystemExit
        print(exc)
        # For config schema, also surface canonical field names to aid users/tests
        if args.schema == "config" and yaml is not None:
            try:
                from pathlib import Path

                data = yaml.safe_load(Path(args.path).read_text()) or {}
                required = ("N_SIMULATIONS", "N_MONTHS")
                missing = [k for k in required if k not in data]
                if missing:
                    print("Missing required field(s): " + ", ".join(missing))
            except (FileNotFoundError, PermissionError, yaml.YAMLError):
                pass
        sys.exit(1)
    print("OK")


if __name__ == "__main__":  # pragma: no cover
    main()
