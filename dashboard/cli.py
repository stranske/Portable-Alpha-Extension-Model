"""Command-line helpers for launching the dashboard."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def build_pa_core_args(
    config_path: str | Path,
    index_path: str | Path,
    output_path: str | Path,
    *,
    use_seed: bool,
    seed_value: int | None,
    extra_args: Iterable[str] | None = None,
) -> list[str]:
    """Build pa_core CLI args for a dashboard run."""
    args = [
        "--config",
        str(config_path),
        "--index",
        str(index_path),
        "--output",
        str(output_path),
    ]
    if use_seed:
        if seed_value is None:
            raise ValueError("Seed value is required when use_seed=True.")
        args.extend(["--seed", str(int(seed_value))])
    if extra_args:
        args.extend([str(arg) for arg in extra_args])
    return args


def main(argv: list[str] | None = None) -> None:
    """Launch the Streamlit dashboard."""
    if argv is None:
        # Avoid leaking pytest's own CLI args into Streamlit when called from tests.
        argv = [] if "pytest" in sys.modules else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Launch the Streamlit dashboard")
    parser.add_argument(
        "--app-path",
        default=None,
        help="Override the Streamlit app path (default: dashboard/app.py)",
    )
    args, streamlit_args = parser.parse_known_args(argv)

    app_path = Path(args.app_path) if args.app_path else Path(__file__).with_name("app.py")
    subprocess.run(["streamlit", "run", str(app_path), *streamlit_args], check=True)


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
