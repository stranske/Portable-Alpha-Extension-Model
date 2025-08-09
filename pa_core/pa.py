from __future__ import annotations

import argparse
import sys
from typing import Sequence

def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="pa")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Run simulation")
    sub.add_parser("validate", help="Validate scenario YAML")
    args, remaining = parser.parse_known_args(argv)
    if args.command == "run":
        from .cli import main as run_main

        run_main(list(remaining))
    else:
        from .validate import main as validate_main

        validate_main(list(remaining))


if __name__ == "__main__":  # pragma: no cover
    main()
