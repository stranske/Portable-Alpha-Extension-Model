"""Create a portable Windows zip archive of the project.

This utility is a starting point for generating a self-contained zip
that can be distributed on Windows systems without an installer.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create portable zip archive")
    parser.add_argument(
        "--output", default="portable_windows.zip", help="Output zip file name"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    shutil.make_archive(str(Path(args.output).with_suffix("")), "zip", root)


if __name__ == "__main__":  # pragma: no cover - utility script
    main()
