"""Generate simple OS launchers for console scripts.

This utility creates Windows ``.bat`` and macOS ``.command`` files that
forward to existing console script entry points.  It is useful when
shipping a portable archive where users may double-click a launcher
instead of invoking the console script from a shell.
"""

from __future__ import annotations

import argparse
import stat
from pathlib import Path


def _make_windows_launcher(name: str, target: Path) -> None:
    """Create a ``.bat`` launcher calling *name* in *target* directory."""
    content = f'@echo off\n"{name}" %*\nif %errorlevel% neq 0 exit /b %errorlevel%\n'
    (target / f"{name}.bat").write_text(content, newline="\r\n")


def _make_mac_launcher(name: str, target: Path) -> None:
    """Create a ``.command`` launcher calling *name* in *target* directory."""
    path = target / f"{name}.command"
    content = (
        f'#!/bin/bash\n'
        f'set -e\n'
        f'if ! command -v {name} >/dev/null 2>&1; then\n'
        f'  echo "Error: {name} not found in PATH." >&2\n'
        f'  exit 1\n'
        f'fi\n'
        f'{name} "$@"\n'
    )
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Windows and macOS launchers for console scripts",
    )
    parser.add_argument("scripts", nargs="+", help="Console script names to wrap")
    parser.add_argument(
        "--output", default=".", help="Directory to place the generated launchers"
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for script in args.scripts:
        _make_windows_launcher(script, out_dir)
        _make_mac_launcher(script, out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
