"""Capture bounded current-main product coverage with reproducible run metadata.

Run from this checkout; test changes are retained as a patch, while product
changes relative to the selected main revision are rejected before pytest runs.
"""

import argparse
from collections import Counter
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from summarize import summarize


def capture(repo: Path, output: Path, revision: str, timeout_seconds: int) -> int:
    if timeout_seconds <= 0:
        raise ValueError("Timeout must be positive")
    environment = os.environ.copy()
    # A clean tree alone does not guarantee the recorded command's selection or
    # denominator. In particular, inherited -x can leave valid partial JUnit and
    # coverage reports with exit 1, which the summary otherwise accepts.
    for name in ("PYTEST_ADDOPTS", "COVERAGE_RCFILE", "COVERAGE_PROCESS_START"):
        if environment.get(name):
            raise ValueError(f"Environment override changes test or coverage settings: {name}")
    repo = repo.resolve()
    output = output.resolve()

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=repo)

    main_sha = (
        git("rev-parse", "--verify", "--end-of-options", revision + "^{commit}").decode().strip()
    )
    product_patch = git("diff", main_sha, "--", "pa_core", "dashboard")
    untracked_product = git("ls-files", "--others", "--exclude-standard", "pa_core", "dashboard")
    if product_patch or untracked_product:
        raise ValueError("Product source differs from selected main revision")
    # These files can alter test selection or the coverage denominator even when
    # every product byte matches main. Do not present such a run as main coverage.
    configuration = (
        "pyproject.toml",
        "pytest.toml",
        ".pytest.toml",
        "pytest.ini",
        ".pytest.ini",
        "tox.ini",
        "setup.cfg",
        ".coveragerc",
    )
    if git("diff", main_sha, "--", *configuration) or git(
        "ls-files", "--others", "--", *configuration
    ):
        raise ValueError("Test or coverage configuration differs from selected main revision")
    test_patch = git("diff", main_sha, "--", "tests")
    for name in git("ls-files", "--others", "--exclude-standard", "tests").decode().splitlines():
        test_patch += "".join(
            difflib.unified_diff(
                [],
                (repo / name).read_text().splitlines(keepends=True),
                fromfile="/dev/null",
                tofile="b/" + name,
            )
        ).encode()
    output.mkdir(parents=True, exist_ok=False)
    (output / "tests.patch").write_bytes(test_patch)
    argv = [
        "timeout",
        "--signal=INT",
        "--kill-after=10s",
        f"{timeout_seconds}s",
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "png_capture_completion",
        "--capture-completion-json=" + str(output / "completion.json"),
        "--cov=pa_core",
        "--cov=dashboard",
        "--cov-report=json:" + str(output / "coverage.json"),
        "--junitxml=" + str(output / "full-suite.xml"),
        "-m",
        "not slow and not live_llm",
        "tests/",
    ]
    metadata = {
        "argv": argv,
        "cwd": str(repo),
        "main_revision": main_sha,
        "requested_revision": revision,
        "head": git("rev-parse", "HEAD").decode().strip(),
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name) for name in ("pytest", "pytest-cov", "coverage")
        },
        "cli_sha256": hashlib.sha256((repo / "pa_core/cli.py").read_bytes()).hexdigest(),
        "test_patch_sha256": hashlib.sha256(test_patch).hexdigest(),
        "test_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in git("ls-files", "--cached", "--others", "--exclude-standard", "tests")
            .decode()
            .splitlines()
        },
        "completion_plugin_sha256": hashlib.sha256(
            Path(__file__).with_name("png_capture_completion.py").read_bytes()
        ).hexdigest(),
        "capture_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pyproject_sha256": hashlib.sha256((repo / "pyproject.toml").read_bytes()).hexdigest(),
        "environment": {
            "COVERAGE_FILE": str(output / ".coverage"),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(Path(__file__).resolve().parent),
                    *filter(None, [environment.get("PYTHONPATH")]),
                ]
            ),
        },
        "selection": "Exclude slow tests and preserve the default live_llm exclusion",
        "completed": False,
    }
    command_path = output / "command.json"

    def save_metadata() -> None:
        command_path.write_text(json.dumps(metadata, indent=2) + "\n")

    # Retain argv and input identity even if interrupted before pytest can report.
    save_metadata()
    started = time.monotonic()
    with (output / "full-suite.txt").open("wb") as log:
        result = subprocess.run(
            argv,
            cwd=repo,
            env={**environment, **metadata["environment"]},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    metadata.update(exit=result.returncode, elapsed_seconds=round(time.monotonic() - started, 2))
    save_metadata()
    try:
        report = summarize(output / "coverage.json", output / "full-suite.xml", result.returncode)
        completion = json.loads((output / "completion.json").read_text())
        if completion["selected"] is None or Counter(completion["selected"]) != Counter(
            completion["finished"]
        ):
            raise ValueError("Selected tests did not finish; partial suite evidence rejected")
        if completion["exit"] != result.returncode:
            raise ValueError("Completion recorder exit disagrees with pytest")
        metadata["test_selection"] = {
            "selected": len(completion["selected"]),
            "deselected": completion["deselected"],
            "finished": len(completion["finished"]),
        }
    except (ValueError, OSError) as exc:
        metadata["incomplete_reason"] = str(exc)
        save_metadata()
        return result.returncode or 1
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    metadata["completed"] = True
    save_metadata()
    totals = report["totals"]
    print(
        f"{totals['covered_lines']} / {totals['num_statements']} statements "
        f"({totals['percent_covered']}%); {report['counts']}"
    )
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="origin/main")
    parser.add_argument("--timeout", type=int, default=600, help="Full-suite timeout in seconds")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    raise SystemExit(capture(repo, args.output, args.revision, args.timeout))


if __name__ == "__main__":
    main()
