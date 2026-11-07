"""Reproduce the eight #2372 proofs in a disposable clone, never the working tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

MERGED = "e8967edb2aa1bdbc1a68ea58ed241cf205915d03"
PAGE = "dashboard/pages/3_Scenario_Wizard.py"
TEST = "tests/test_wizard_render_coverage.py"
ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/issue-2360-closer-repair"


def replay(output: Path) -> bool:
    """Retain full red/green output, exact edits, and restoration hashes."""
    plan = json.loads((EVIDENCE / "mutation-plan.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="issue2360-mutations-") as temporary:
        checkout = Path(temporary) / "checkout"
        subprocess.run(
            ["git", "clone", "--shared", "--no-checkout", str(ROOT), str(checkout)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", MERGED],
            check=True,
            capture_output=True,
        )
        page = checkout / PAGE
        original = page.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        for index, mutation in enumerate(plan, 1):
            old, new = mutation["before"], mutation["after"]
            source = original.decode()
            if source.count(old) != 1:
                raise ValueError(f"Mutation {index} must have exactly one matching location")
            # No inherited PYTHONPATH: imports must come from this historical checkout.
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            command = [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                "-q",
                "-m",
                "not slow and not live_llm",
                f"{TEST}::{mutation['node']}",
            ]
            receipts = {}
            try:
                page.write_text(source.replace(old, new), encoding="utf-8")
                for phase in ("red", "green"):
                    if phase == "green":
                        page.write_bytes(original)
                    completed = subprocess.run(
                        command,
                        cwd=checkout,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    log = output / f"{index:02d}-{phase}.txt"
                    output_text = completed.stdout + completed.stderr
                    log.write_text(
                        "\n".join(line.rstrip() for line in output_text.splitlines()) + "\n",
                        encoding="utf-8",
                    )
                    receipts[f"{phase}_exit"] = completed.returncode
                    receipts[f"{phase}_output"] = log.name
            finally:
                page.write_bytes(original)
            results.append(
                {
                    **mutation,
                    **receipts,
                    "command": shlex.join(command),
                    "restored_sha256": hashlib.sha256(page.read_bytes()).hexdigest(),
                }
            )
            print(
                f"{index}/8 {mutation['node']}: red={receipts['red_exit']} "
                f"green={receipts['green_exit']}",
                flush=True,
            )
        record = {
            "revision": MERGED,
            "wizard_sha256": digest,
            "checkout": "disposable clone; original bytes restored in finally after every case",
            "output_normalization": "Trailing whitespace removed from console lines only",
            "results": results,
        }
        (output / "mutation-replay.json").write_text(json.dumps(record, indent=2) + "\n")
    return all(row["red_exit"] == 1 and row["green_exit"] == 0 for row in results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if replay(args.output.resolve()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
