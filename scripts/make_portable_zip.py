"""Create a portable Windows zip archive of the project.

This utility filters out development artifacts to produce a
self-contained archive suitable for distribution without an installer.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import zipfile
from pathlib import Path
from typing import Set


def get_default_excludes() -> Set[str]:
    """Return patterns to exclude from the portable archive."""
    return {
        # Version control
        ".git",
        ".gitignore",
        ".gitattributes",
        ".github",
        # Python caches and builds
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.egg-info",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        # Virtual environments
        ".venv",
        "venv",
        "dev-env",
        # Development tools and configs
        ".pre-commit-config.yaml",
        ".vscode",
        ".idea",
        "*.swp",
        "*.swo",
        ".ipynb_checkpoints",
        ".jupyter",
        "pyrightconfig.json",
        # Development documentation and logs
        "DEVELOPMENT_*.md",
        "CODEX_*.md",
        "TUTORIAL_*_TESTING_RESULTS.md",
        "AUTOMATION_QUICK_START.md",
        "COMPLETE_TUTORIAL_TESTING_SUMMARY.md",
        "DEMO_TESTING_LOG.md",
        "FUNCTIONALITY_GAP_ANALYSIS.md",
        "PARAMETER_SWEEP_STATUS_REPORT.md",
        "STREAMLIT_TUTORIAL_INTEGRATION_PLAN.md",
        "TUTORIAL_IMPLEMENTATION_DEPENDENCY.md",
        "TUTORIAL_UPDATE_DRAFT.md",
        "TEST_PERMISSIONS.md",
        "code-quality.log",
        "debug_*.md",
        "debug_*.py",
        "streamlined_*.md",
        "streamlined-*.yml",
        "test_debug_*.md",
        "tutorial*_issues.md",
        "user_testing_issues.md",
        "codex.patch",
        # Build artifacts and outputs
        "docs/_build",
        "plots",
        "*.xlsx",
        "*.tmp",
        "*.temp",
        ".env",
        ".env.local",
        "Outputs.parquet",
        "get-pip.py",
        # OS files
        ".DS_Store",
        "Thumbs.db",
        # Development containers and tools
        ".devcontainer",
        ".gate_smoke",
        "Makefile",
        "archive",
        # Jupyter notebooks
        "*.ipynb",
    }


def should_exclude_path(path: Path, root: Path, excludes: Set[str]) -> bool:
    """Return True if *path* should be excluded from the archive."""
    try:
        rel_path = path.relative_to(root)
    except ValueError:
        return True  # Path is outside the root directory

    rel_str = str(rel_path)
    parts = rel_path.parts

    if (
        path.is_file()
        and path.stat().st_size == 0
        and path.suffix == ".py"
        and path.name != "__init__.py"
    ):
        return True

    for pattern in excludes:
        if rel_str == pattern or path.name == pattern:
            return True
        if "*" in pattern and (
            fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(rel_str, pattern)
        ):
            return True
        if any(part == pattern for part in parts):
            return True
        if pattern.endswith("*") and path.name.startswith(pattern[:-1]):
            return True
    return False


def create_filtered_zip(
    root_dir: Path, output_path: Path, excludes: Set[str], *, verbose: bool = False
) -> None:
    """Create a zip archive excluding paths that match *excludes*."""
    files_added = 0
    files_excluded = 0

    print(f"Creating portable archive: {output_path}")
    print(f"Source directory: {root_dir}")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in root_dir.rglob("*"):
            if path.is_file():
                if should_exclude_path(path, root_dir, excludes):
                    files_excluded += 1
                    if verbose:
                        print(f"Excluded: {path.relative_to(root_dir)}")
                    continue
                arcname = path.relative_to(root_dir)
                zipf.write(path, arcname)
                files_added += 1
                if verbose:
                    print(f"Included: {arcname}")

    size_mb = output_path.stat().st_size / 1024 / 1024
    print("\nArchive created successfully!")
    print(f"Files included: {files_added}")
    print(f"Files excluded: {files_excluded}")
    print(f"Archive size: {size_mb:.2f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a portable zip archive with development files filtered out",
    )
    parser.add_argument(
        "--output",
        default="portable_windows.zip",
        help="Name of the generated archive",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="Additional patterns to exclude (may be used multiple times)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show every file included or excluded",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_path = Path(args.output).resolve()

    excludes = get_default_excludes().union(args.exclude_pattern)

    try:
        create_filtered_zip(root, output_path, excludes, verbose=args.verbose)
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"Error creating archive: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
