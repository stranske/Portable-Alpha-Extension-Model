"""Replay six real CLI production mutations into a new immutable output directory."""

import argparse
import hashlib
import json
import subprocess
import shutil
import tempfile
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    # Copy tracked checkout bytes before any mutation. Each invocation owns its
    # private tree, so exceptions, SIGTERM and concurrent runs cannot mutate the
    # caller's source or caches. Temporary trees may survive an uncatchable kill.
    with tempfile.TemporaryDirectory(prefix="cli-mutation-replay-") as temporary:
        isolated = Path(temporary) / "checkout"
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
        for raw in filter(None, tracked):
            relative = Path(raw.decode("utf-8"))
            target = isolated / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, target)
        replay(isolated, output, args.python)


def replay(root, output, python):
    source = root / "pa_core/cli.py"
    test = root / "tests/test_cli_observability_boundaries.py"
    original = source.read_bytes()
    text = original.decode("utf-8")
    cases = [
        (
            "test_warning_snapshot_context_is_detached_from_collector[logging]",
            "return deepcopy(self.records)",
            "return [dict(rec) for rec in self.records]",
        ),
        (
            "test_warning_snapshot_context_is_detached_from_collector[warnings]",
            "return deepcopy(self.records)",
            "return [dict(rec) for rec in self.records]",
        ),
        (
            "test_enhanced_summary_preserves_monthly_returns_and_benchmark[no-benchmark]",
            "return summary_table(returns_map, benchmark=benchmark)",
            "return summary_table({k: -v for k, v in returns_map.items()}, benchmark=benchmark)",
        ),
        (
            "test_enhanced_summary_preserves_monthly_returns_and_benchmark[index-benchmark]",
            "return summary_table(returns_map, benchmark=benchmark)",
            "return summary_table(returns_map, benchmark=None)",
        ),
        (
            "test_config_snapshot_preserves_utf8_and_original_bytes",
            "return raw.decode(), raw",
            'return raw.decode(), raw.replace(b"\\r\\n", b"\\n")',
        ),
        (
            "test_run_timer_snapshot_uses_monotonic_elapsed_and_utc_wall_time",
            '"duration_seconds": self.elapsed(),',
            '"duration_seconds": 0.0,',
        ),
    ]
    controls = {
        "source_sha256": digest(original),
        "test_sha256": digest(test.read_bytes()),
        "cases": [],
    }
    try:
        for index, (name, before, after) in enumerate(cases):
            if text.count(before) != 1:
                raise RuntimeError("Mutation anchor is not unique: " + name)
            mutant = text.replace(before, after, 1).encode("utf-8")
            node = "tests/test_cli_observability_boundaries.py::" + name
            case = {
                "node": node,
                "mutant_sha256": digest(mutant),
                "before": before,
                "after": after,
                "phases": [],
            }
            controls["cases"].append(case)
            for phase, content, expected in [("red", mutant, 1), ("green", original, 0)]:
                source.write_bytes(content)
                # Avoid timestamp/size cache aliasing when a restored source has the same size.
                for cache in (root / "pa_core/__pycache__").glob("cli.*.pyc"):
                    cache.unlink()
                stem = f"{index:02d}-{phase}"
                junit = output / (stem + ".xml")
                argv = [
                    python,
                    "-m",
                    "pytest",
                    node,
                    "-m",
                    "not slow",
                    "-q",
                    "--junitxml=" + str(junit),
                ]
                started = time.time()
                with (output / (stem + ".txt")).open("w", encoding="utf-8") as stream:
                    proc = subprocess.run(
                        argv,
                        cwd=root,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=90,
                        check=False,
                    )
                record = {
                    "phase": phase,
                    "argv": argv,
                    "cwd": str(root),
                    "exit": proc.returncode,
                    "elapsed_seconds": time.time() - started,
                    "source_sha256": digest(source.read_bytes()),
                    "console": stem + ".txt",
                    "junit": stem + ".xml",
                }
                case["phases"].append(record)
                nodes = ET.parse(junit).getroot().findall(".//testcase")
                if len(nodes) != 1 or nodes[0].get("name") != name:
                    raise RuntimeError("Unexpected named JUnit case: " + node)
                result = nodes[0]
                if result.find("error") is not None or result.find("skipped") is not None:
                    raise RuntimeError("Collection error or skipped node: " + node)
                if (result.find("failure") is not None) != (phase == "red"):
                    raise RuntimeError("Unexpected JUnit phase outcome: " + node)
                if proc.returncode != expected:
                    raise RuntimeError("Unexpected exit: " + node)
    finally:
        source.write_bytes(original)
        for cache in (root / "pa_core/__pycache__").glob("cli.*.pyc"):
            cache.unlink()
        controls["restored_sha256"] = digest(source.read_bytes())
        (output / "controls.json").write_text(json.dumps(controls, indent=2), encoding="utf-8")
    if source.read_bytes() != original or digest(test.read_bytes()) != controls["test_sha256"]:
        raise RuntimeError("Source/test restoration failed")
    print("6 named nodes: actual production RED then byte-identical restored GREEN")


if __name__ == "__main__":
    main()
