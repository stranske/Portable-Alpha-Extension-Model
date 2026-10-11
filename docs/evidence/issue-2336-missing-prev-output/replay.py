"""Replay four missing-workbook mutations in a private tracked-file snapshot."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

NODE = "test_sweep_packet_prev_summary_empty_when_prev_output_missing"
EXPECTED = {
    f"{NODE}[{path}-{missing}]"
    for path in ("absolute", "relative")
    for missing in ("missing-file", "missing-parent")
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_junit(path, phase):
    nodes = ET.parse(path).getroot().findall(".//testcase")
    if len(nodes) != 4 or {node.get("name") for node in nodes} != EXPECTED:
        raise RuntimeError("Expected exactly the four missing-workbook JUnit nodes")
    for node in nodes:
        if node.find("error") is not None or node.find("skipped") is not None:
            raise RuntimeError("Collection error or skipped node: " + node.get("name"))
        if (node.find("failure") is not None) != (phase == "red"):
            raise RuntimeError("Unexpected JUnit phase outcome: " + node.get("name"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    # The caller's source and caches are never written, including on SIGTERM.
    with tempfile.TemporaryDirectory(prefix="missing-workbook-replay-") as temporary:
        private = Path(temporary) / "checkout"
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root, timeout=30)
        for raw in filter(None, tracked.split(b"\0")):
            relative = Path(raw.decode("utf-8"))
            target = private / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, target)
        replay(private, output, args.timeout)


def replay(root, output, timeout):
    source = root / "pa_core/cli.py"
    caller = root / "tests/test_cli_packet_diff.py"
    before = source.read_bytes()
    caller_before = caller.read_bytes()
    needle = b"if prev_out and Path(prev_out).exists():"
    if before.count(needle) != 1:
        raise RuntimeError("Mutation anchor is not unique")
    mutant = before.replace(needle, b"if prev_out:")
    case = {"mutant_sha256": digest(mutant), "phases": []}
    controls = {
        "source_sha256": digest(before),
        "test_sha256": digest(caller_before),
        "cases": [case],
        "complete": False,
    }
    try:
        for phase, content, expected in [("red", mutant, 1), ("green", before, 0)]:
            source.write_bytes(content)
            for cache in (root / "pa_core/__pycache__").glob("cli.*.pyc"):
                cache.unlink()
            junit = output / (phase + ".xml")
            argv = [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_cli_packet_diff.py::" + NODE,
                "-q",
                "-m",
                "not slow",
                "--junitxml=" + str(junit),
            ]
            (output / (phase + "-command.json")).write_text(json.dumps(argv, indent=2))
            with (output / ("00-" + phase + ".txt")).open("w") as stream:
                result = subprocess.run(
                    argv,
                    cwd=root,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
            validate_junit(junit, phase)
            if result.returncode != expected:
                raise RuntimeError("Unexpected pytest exit for " + phase)
            case["phases"].append(
                {"phase": phase, "exit": result.returncode, "nodes": sorted(EXPECTED)}
            )
        if caller.read_bytes() != caller_before:
            raise RuntimeError("Caller changed")
        controls["complete"] = True
    except subprocess.TimeoutExpired:
        controls["failure"] = "pytest timeout"
        raise
    finally:
        source.write_bytes(before)
        controls["restored_sha256"] = digest(source.read_bytes())
        (output / "controls.json").write_text(json.dumps(controls, indent=2))
    print(json.dumps(controls))


if __name__ == "__main__":
    main()
