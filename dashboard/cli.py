"""Command-line helpers for launching the dashboard."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """Launch the Streamlit dashboard."""
    if argv is None:
        argv = sys.argv[1:]
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
