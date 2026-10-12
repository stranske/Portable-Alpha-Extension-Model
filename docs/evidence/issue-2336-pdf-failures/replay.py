"""Replay exact-node PDF diagnostic mutations in a pinned private Git snapshot."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET

REVISION = "395fcd394e185cd5042ca6f90569928f2f037844"
SOURCE_SHA256 = "95f0e38b1e2491ff23fe4f278c2d5c5d29a6293bfbf5ef14b64795a4c954906e"
TEST = "tests/test_cli_pdf_failures.py"
CASES = {
    "renderer": "❌ PDF export failed: Kaleido or Chrome/Chromium required",
    "module": "❌ PDF export failed due to missing dependency: {e}",
    "io": "❌ PDF export failed: Cannot write file - {e}",
    "permission": "❌ PDF export failed: Cannot write file - {e}",
    "value": "❌ PDF export failed: Invalid data - {e}",
    "type": "❌ PDF export failed: Invalid data - {e}",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    caller_inputs = {p: (repo / p).read_bytes() for p in ("pa_core/cli.py", TEST)}
    archive = subprocess.check_output(["git", "archive", REVISION], cwd=repo)
    results = []
    with tempfile.TemporaryDirectory(prefix="pa-pdf-proof-") as temp:
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
            mutant = text.replace(target, "❌ PDF diagnostic deliberately broken", 1).encode()
            node = f"{TEST}::test_single_run_pdf_failure_preserves_results[{case}]"
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
                    tests = ET.parse(junit).getroot().findall(".//testcase")
                    if len(tests) != 1 or tests[0].get("name") != node.split("::")[-1]:
                        raise RuntimeError(f"Wrong or incomplete JUnit node: {case}/{phase}")
                    failures = tests[0].findall("failure")
                    if phase == "red":
                        valid = (
                            proc.returncode == 1
                            and len(failures) == 1
                            and "AssertionError" in failures[0].get("message", "")
                        )
                    else:
                        valid = proc.returncode == 0 and not failures
                    if (
                        tests[0].find("error") is not None
                        or tests[0].find("skipped") is not None
                        or not valid
                    ):
                        raise RuntimeError(f"Invalid proof outcome: {case}/{phase}")
                    row[phase] = {
                        "argv": argv,
                        "exit": proc.returncode,
                        "source_sha256": digest(source.read_bytes()),
                        "cwd": str(private),
                    }
            finally:
                source.write_bytes(original)
            row["restored_sha256"] = digest(source.read_bytes())
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
    print("Six exact assertion RED / byte-identical restored GREEN pairs; caller unchanged.")


if __name__ == "__main__":
    main()
