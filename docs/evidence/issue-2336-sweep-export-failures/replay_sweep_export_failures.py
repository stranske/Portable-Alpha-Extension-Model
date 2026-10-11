"""Capture real CLI handler mutations in an isolated committed source snapshot."""

import argparse
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE = "pa_core/cli.py"
TEST = "tests/test_cli_sweep_export_failures.py"
CASES = {
    "runtime": "except RuntimeError as e:",
    "import": "except (ImportError, ModuleNotFoundError) as e:",
    "module": "except (ImportError, ModuleNotFoundError) as e:",
    "value": "except (ValueError, TypeError, KeyError) as e:",
    "type": "except (ValueError, TypeError, KeyError) as e:",
    "key": "except (ValueError, TypeError, KeyError) as e:",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    caller = {name: digest((root / name).read_bytes()) for name in (SOURCE, TEST)}
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    archive = subprocess.check_output(["git", "-C", str(root), "archive", revision])
    controls = []
    with tempfile.TemporaryDirectory(prefix="pa-sweep-export-proof-") as tmp:
        tree = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(tree, filter="data")
        (tree / TEST).write_bytes((root / TEST).read_bytes())
        source = tree / SOURCE
        original = source.read_bytes()
        if digest(original) != caller[SOURCE]:
            raise ValueError("Committed source differs from caller; commit source changes first")
        text = original.decode()
        start = text.index("        # Handle packet export for parameter sweep mode")
        end = text.index("        # Sensitivity analysis can also", start)
        section = text[start:end]
        for case, before in CASES.items():
            if section.count(before) != 1:
                raise ValueError("Expected one scoped handler")
            mutant = (
                text[:start]
                + section.replace(before, "except FloatingPointError as e:")
                + text[end:]
            ).encode()
            node = f"{TEST}::test_sweep_packet_export_failure_is_truthful[{case}]"
            try:
                source.write_bytes(mutant)
                for phase, expected in (("red", 1), ("green", 0)):
                    if phase == "green":
                        source.write_bytes(original)
                    prefix = output / f"{case}-{phase}"
                    report = prefix.with_suffix(".xml")
                    env = dict(os.environ)
                    env["PYTEST_ADDOPTS"] = shlex.join(
                        ["--junitxml", str(report), "--override-ini", "addopts="]
                    )
                    command = [sys.executable, "-m", "pytest", node, "-q", "-m", "not slow"]
                    result = subprocess.run(
                        command,
                        cwd=tree,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                    prefix.with_suffix(".txt").write_text(result.stdout + result.stderr)
                    prefix.with_suffix(".json").write_text(
                        json.dumps(
                            {
                                "argv": command,
                                "cwd": str(tree),
                                "PYTEST_ADDOPTS": env["PYTEST_ADDOPTS"],
                                "exit": result.returncode,
                                "source_sha256": digest(source.read_bytes()),
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    suites = list(ET.parse(report).getroot().iter("testcase"))
                    if result.returncode != expected or len(suites) != 1:
                        raise ValueError(f"Incomplete {case}/{phase} proof")
                    node_result = suites[0]
                    if (
                        node_result.get("name")
                        != f"test_sweep_packet_export_failure_is_truthful[{case}]"
                        or node_result.find("error") is not None
                        or node_result.find("skipped") is not None
                    ):
                        raise ValueError("Wrong or skipped named node")
                    if (node_result.find("failure") is not None) != (phase == "red"):
                        raise ValueError("Unexpected named-node outcome")
            finally:
                source.write_bytes(original)
            controls.append(
                {
                    "case": case,
                    "before": before,
                    "after": "except FloatingPointError as e:",
                    "mutant_sha256": digest(mutant),
                    "restored_sha256": digest(source.read_bytes()),
                    "red_exit": 1,
                    "green_exit": 0,
                }
            )
    current = {name: digest((root / name).read_bytes()) for name in caller}
    if current != caller:
        raise ValueError("Caller changed during replay")
    (output / "controls.json").write_text(
        json.dumps(
            {
                "snapshot_commit": revision,
                "caller_before": caller,
                "caller_after": current,
                "cases": controls,
            },
            indent=2,
        )
        + "\n"
    )
    print("Six real production mutations RED; exact restorations GREEN; caller preserved.")


if __name__ == "__main__":
    main()
