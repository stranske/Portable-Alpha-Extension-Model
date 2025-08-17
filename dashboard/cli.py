"""Command-line helpers for launching the dashboard."""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    """Launch the Streamlit dashboard."""
    app_path = Path(__file__).with_name("app.py")
    subprocess.run(["streamlit", "run", str(app_path)], check=True)


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
