from __future__ import annotations

import argparse
import sys
from pydantic import ValidationError

from .schema import load_scenario


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate scenario YAML")
    parser.add_argument("path", help="Scenario YAML file")
    args = parser.parse_args(argv)
    try:
        load_scenario(args.path)
    except ValidationError as exc:  # pragma: no cover - reached in tests via SystemExit
        print(exc)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":  # pragma: no cover
    main()
