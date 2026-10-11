"""Replay the existence-guard mutation in a disposable clean checkout."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
source = root / "pa_core/cli.py"
caller = root / "tests/test_cli_packet_diff.py"
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
before = source.read_bytes()
caller_before = caller.read_bytes()
needle = b"if prev_out and Path(prev_out).exists():"
assert before.count(needle) == 1
command = [
    sys.executable,
    "-m",
    "pytest",
    "tests/test_cli_packet_diff.py::test_sweep_packet_prev_summary_empty_when_prev_output_missing",
    "-q",
    "-m",
    "not slow",
]
exits = []
try:
    source.write_bytes(before.replace(needle, b"if prev_out:"))
    for phase in ("red", "green"):
        if phase == "green":
            source.write_bytes(before)
        argv = command + ["--junitxml=" + str(out / (phase + ".xml"))]
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True)
        (out / (phase + ".txt")).write_text(result.stdout + result.stderr)
        (out / (phase + "-command.json")).write_text(json.dumps(argv, indent=2))
        exits.append(result.returncode)
finally:
    source.write_bytes(before)
receipt = {
    "exits": exits,
    "restored": source.read_bytes() == before,
    "caller_unchanged": caller.read_bytes() == caller_before,
    "source_sha256": hashlib.sha256(before).hexdigest(),
    "caller_sha256": hashlib.sha256(caller_before).hexdigest(),
}
(out / "result.json").write_text(json.dumps(receipt, indent=2))
assert exits == [1, 0] and receipt["restored"] and receipt["caller_unchanged"], receipt
print(json.dumps(receipt))
