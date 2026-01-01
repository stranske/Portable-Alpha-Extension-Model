"""Create a portable Windows zip archive of the project.

Default behavior creates a source-only archive (no interpreter). On Windows,
use ``--with-python`` to bundle the embeddable CPython runtime and generate
launchers that run the CLI and dashboard without requiring a prior install.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import subprocess
import sys
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
        "build",
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


def _download(url: str, dest: Path) -> None:  # pragma: no cover - network/deps
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def _unzip(zip_path: Path, dest_dir: Path) -> None:  # pragma: no cover - platform
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _enable_embedded_site(py_dir: Path) -> None:  # pragma: no cover - windows only
    # Enable site-packages in Windows embeddable distribution by editing the
    # pythonXY._pth file: add 'Lib\\site-packages' and ensure 'import site'.
    pth_files = list(py_dir.glob("python*._pth"))
    if not pth_files:
        return
    pth = pth_files[0]
    lines = pth.read_text(encoding="utf-8").splitlines()
    has_import_site = any(line.strip() == "import site" for line in lines)
    if not has_import_site:
        lines.append("import site")
    if all("Lib\\\\site-packages" not in line for line in lines):
        lines.insert(0, "Lib\\site-packages")
    pth.write_text("\n".join(lines), encoding="utf-8")


def _write_launcher_bats(staging: Path) -> None:  # pragma: no cover - windows only
    launchers = {
        "pa.bat": '@echo off\n"%~dp0python\\python.exe" -m pa_core.pa %*\n',
        "pa-dashboard.bat": (
            "@echo off\n" '"%~dp0python\\python.exe" -m streamlit run dashboard\\app.py %*\n'
        ),
        "pa-validate.bat": '@echo off\n"%~dp0python\\python.exe" -m pa_core.validate %*\n',
        "pa-convert-params.bat": (
            '@echo off\n"%~dp0python\\python.exe" -m pa_core.data.convert %*\n'
        ),
    }
    for name, content in launchers.items():
        (staging / name).write_text(content, newline="\r\n")


def build_windows_portable_zip(
    project_root: Path,
    output_path: Path,
    *,
    python_version: str = "3.12.11",
    verbose: bool = False,
) -> None:  # pragma: no cover - platform/network
    """Build a Windows portable zip including embeddable Python and deps.

    Steps (Windows only):
    - Download CPython embeddable zip for the selected version
    - Enable site-packages import
    - Bootstrap pip (get-pip.py)
    - pip install -r requirements.txt into the embedded env
    - Copy project sources
    - Generate .bat launchers
    - Zip the staging directory
    """
    if sys.platform != "win32":
        # Fallback to source-only zip when not on Windows
        excludes = get_default_excludes()
        create_filtered_zip(project_root, output_path, excludes, verbose=verbose)
        print("Note: --with-python requires Windows; created source-only archive.")
        return

    staging = output_path.with_suffix("").parent / "portable_build"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    # 1) Download embeddable Python
    embed_name = f"python-{python_version}-embed-amd64.zip"
    base = f"https://www.python.org/ftp/python/{python_version}/{embed_name}"
    embed_zip = staging / embed_name
    print(f"Downloading embeddable Python {python_version}...")
    _download(base, embed_zip)
    (staging / "python").mkdir(exist_ok=True)
    _unzip(embed_zip, staging / "python")
    _enable_embedded_site(staging / "python")

    # 2) Bootstrap pip - SECURE VERSION using subprocess.run()
    print("Bootstrapping pip in embedded Python...")
    get_pip = staging / "get-pip.py"
    _download("https://bootstrap.pypa.io/get-pip.py", get_pip)

    # SECURITY FIX: Use subprocess.run() with list of arguments instead of os.system()
    # This prevents command injection even if paths contain malicious characters
    python_exe = staging / "python" / "python.exe"
    try:
        subprocess.run([str(python_exe), str(get_pip)], check=True, cwd=staging)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to bootstrap pip: {e}") from e

    # 3) Install dependencies - SECURE VERSION using subprocess.run()
    req = project_root / "requirements.txt"
    if req.exists():
        # SECURITY FIX: Use subprocess.run() with list of arguments instead of os.system()
        # This prevents command injection even if paths contain malicious characters
        try:
            subprocess.run(
                [
                    str(python_exe),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(req),
                    "--no-warn-script-location",
                ],
                check=True,
                cwd=staging,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install dependencies: {e}") from e

    # 4) Copy project sources
    print("Copying project files...")
    for item in project_root.iterdir():
        if item.name in {".git", "dev-env", "archive"}:
            continue
        if item.is_dir() and item.name in {".venv", "__pycache__"}:
            continue
        dest = staging / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    # 5) Write launchers
    _write_launcher_bats(staging)

    # 6) Create final zip from staging
    print("Creating final portable zip...")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging))
    print("Portable zip created:", output_path)


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
    parser.add_argument(
        "--with-python",
        action="store_true",
        help="On Windows, include embeddable Python runtime and dependencies",
    )
    parser.add_argument(
        "--python-version",
        default="3.12.11",
        help="Python version for embeddable runtime (Windows only)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_path = Path(args.output).resolve()

    if args.with_python:
        try:
            build_windows_portable_zip(
                root,
                output_path,
                python_version=args.python_version,
                verbose=args.verbose,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"Error creating portable zip: {exc}")
            return 1
    else:
        excludes = get_default_excludes().union(args.exclude_pattern)
        try:
            create_filtered_zip(root, output_path, excludes, verbose=args.verbose)
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"Error creating archive: {exc}")
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
