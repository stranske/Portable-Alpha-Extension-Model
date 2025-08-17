from __future__ import annotations

import argparse
import sys
from pydantic import ValidationError

from .schema import load_scenario
from .config import load_config


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
    except (ValidationError, ValueError) as exc:  # pragma: no cover - reached via SystemExit
        print(exc)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":  # pragma: no cover
    main()
