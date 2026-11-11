"""Run the #2495 current-main baseline commands once each and preserve raw evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
TIMEOUT_SECONDS = 600
PRODUCTION = ["pa_core/cli.py"]
PACKAGES = ["pytest", "pytest-cov", "coverage", "numpy", "pandas", "scipy", "openpyxl", "kaleido", "plotly"]
FOCUSED = [
    "tests/test_cli_png_failures.py",
    "tests/test_cli_pdf_failures.py",
    "tests/test_cli_sweep_png_export.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def run(name: str, argv: list[str]) -> dict[str, object]:
    stdout_path = OUT / f"{name}-stdout.log"
    stderr_path = OUT / f"{name}-stderr.log"
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=err)
        try:
            exit_code: int | None = proc.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
            exit_code = None
    return {
        "name": name,
        "argv": argv,
        "exit": exit_code,
        "timeout": timed_out,
        "completed": not timed_out,
        "timeout_seconds": TIMEOUT_SECONDS,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "max_attempts": 1,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "commit": git("rev-parse", "HEAD"),
        "working_tree_porcelain": git("status", "--porcelain", "--", "pa_core", "dashboard", "tests"),
        "interpreter": sys.executable,
        "python_version": sys.version,
        "dependency_versions": versions(),
        "production_sha256_before": {p: sha256(ROOT / p) for p in PRODUCTION},
    }
    full = run(
        "full",
        [
            sys.executable, "-m", "pytest",
            "--cov=pa_core", "--cov=dashboard", "--cov-report=json:coverage.json", "tests/",
            f"--junitxml={OUT / 'full-junit.xml'}",
        ],
    )
    coverage = ROOT / "coverage.json"
    if coverage.exists():
        shutil.move(coverage, OUT / "coverage.json")
    focused = run(
        "focused",
        [sys.executable, "-m", "pytest", *FOCUSED, "-q", f"--junitxml={OUT / 'focused-junit.xml'}"],
    )
    record["runs"] = [full, focused]
    record["production_sha256_after"] = {p: sha256(ROOT / p) for p in PRODUCTION}
    (OUT / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
