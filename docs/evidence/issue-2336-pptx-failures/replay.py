"""Replay exact-node PPTX error-log mutations in a pinned private Git snapshot."""

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET

REVISION = "0e98dcaf6b43a6320ce5e3c74d764d182701d04a"
SOURCE_SHA256 = "95f0e38b1e2491ff23fe4f278c2d5c5d29a6293bfbf5ef14b64795a4c954906e"
TEST = "tests/test_cli_pptx_failures.py"
CASES = {
    "renderer": 'logger.error(f"PPTX export failed due to missing dependency: {e}")',
    "module": 'logger.error(f"PPTX export failed due to missing module: {e}")',
    "io": 'logger.error(f"PPTX export failed due to file system issue: {e}")',
    "permission": 'logger.error(f"PPTX export failed due to file system issue: {e}")',
    "value": 'logger.error(f"PPTX export failed due to data issue: {e}")',
    "type": 'logger.error(f"PPTX export failed due to data issue: {e}")',
    "runtime-renderer": 'logger.error(f"PPTX export failed: {e}")',
    "python-pptx": 'logger.error(f"PPTX export failed due to missing python-pptx: {e}")',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_junit_proof(junit_path, expected_test_name, phase, returncode, case):
    """Validate a single-node JUnit report and subprocess exit for RED/GREEN proof."""
    tests = ET.parse(junit_path).getroot().findall(".//testcase")
    if len(tests) != 1 or tests[0].get("name") != expected_test_name:
        raise RuntimeError(f"Wrong or incomplete JUnit node: {case}/{phase}")
    if tests[0].get("classname") != "tests.test_cli_pptx_failures":
        raise RuntimeError("Wrong JUnit module")
    if ET.parse(junit_path).getroot().findall(".//error"):
        raise RuntimeError("JUnit collection/setup error")
    failures = tests[0].findall("failure")
    if phase == "red":
        valid = (
            returncode == 1
            and len(failures) == 1
            and (
                failures[0].get("message", "").startswith("AssertionError:")
                or re.search(
                    r"^.+:\d+: AssertionError(?::|$)", failures[0].text or "", re.MULTILINE
                )
            )
        )
    else:
        valid = returncode == 0 and not failures
    if tests[0].find("error") is not None or tests[0].find("skipped") is not None or not valid:
        raise RuntimeError(f"Invalid proof outcome: {case}/{phase}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    caller_inputs = {p: (repo / p).read_bytes() for p in ("pa_core/cli.py", TEST)}
    git_env = os.environ.copy()
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    ):
        git_env.pop(key, None)
    archive = subprocess.check_output(
        ["git", "-C", str(repo), "archive", REVISION], env=git_env, timeout=60
    )
    (output / "original-source.py").write_bytes(caller_inputs["pa_core/cli.py"])
    (output / "test-input.py").write_bytes(caller_inputs[TEST])
    (output / "driver-input.py").write_bytes(Path(__file__).read_bytes())
    results = []
    with tempfile.TemporaryDirectory(prefix="pa-pptx-proof-") as temp:
        private = Path(temp)
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(private, filter="data")
        (private / TEST).write_bytes(caller_inputs[TEST])
        source = private / "pa_core/cli.py"
        original = source.read_bytes()
        if digest(original) != SOURCE_SHA256:
            raise RuntimeError("Pinned production source digest mismatch")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(private)
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for case, target in CASES.items():
            text = original.decode()
            if text.count(target) != 1:
                raise RuntimeError(f"Ambiguous mutation anchor: {case}")
            mutant = text.replace(
                target, target.replace("logger.error", "logger.debug"), 1
            ).encode()
            node = f"{TEST}::test_single_run_pptx_failure_preserves_results[{case}]"
            row = {
                "case": case,
                "node": node,
                "original_sha256": digest(original),
                "mutant_sha256": digest(mutant),
                "target": target,
            }
            (output / f"{case}-mutant-source.py").write_bytes(mutant)
            try:
                source.write_bytes(mutant)
                for phase in ("red", "green"):
                    if phase == "green":
                        source.write_bytes(original)
                    junit = output / f"{case}-{phase}.xml"
                    argv = [
                        sys.executable,
                        "-B",
                        "-m",
                        "pytest",
                        node,
                        "-q",
                        "--junitxml=" + str(junit),
                    ]
                    with (output / f"{case}-{phase}.txt").open("w") as log:
                        proc = subprocess.run(
                            argv,
                            cwd=private,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=180,
                        )
                    validate_junit_proof(
                        junit,
                        node.split("::")[-1],
                        phase,
                        proc.returncode,
                        case,
                    )
                    row[phase] = {
                        "argv": argv,
                        "exit": proc.returncode,
                        "source_sha256": digest(source.read_bytes()),
                        "cwd": str(private),
                    }
            finally:
                source.write_bytes(original)
            row["restored_sha256"] = digest(source.read_bytes())
            if row["restored_sha256"] != row["original_sha256"]:
                raise RuntimeError("Restoration digest mismatch")
            results.append(row)
    if any((repo / p).read_bytes() != content for p, content in caller_inputs.items()):
        raise RuntimeError("Caller input changed")
    (output / "results.json").write_text(
        json.dumps(
            {
                "revision": REVISION,
                "caller_unchanged": True,
                "inputs": {p: digest(content) for p, content in caller_inputs.items()},
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print("Eight exact assertion RED / byte-identical restored GREEN pairs; caller unchanged.")


if __name__ == "__main__":
    main()
