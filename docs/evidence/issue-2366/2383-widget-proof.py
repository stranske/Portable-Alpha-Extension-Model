import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out = Path(__file__).parent
files = ["dashboard/components/explain_results.py", "dashboard/components/comparison_llm.py"]
fixed = {f: (root / f).read_bytes() for f in files}
old = {
    f: subprocess.check_output(
        ["git", "-C", str(root), "show", "c81e14368ceca68ac6c95bfb98a41e12fa229bef:" + f]
    )
    for f in files
}
historical = json.loads((out / "2383-widget-causal.json").read_text())
assert all(
    hashlib.sha256(content).hexdigest() == historical["sources"][name]["old"]
    for name, content in old.items()
), "Pinned baseline hashes differ from preserved causal receipt"
cmd = [
    sys.executable,
    "-m",
    "pytest",
    "tests/test_dashboard_provider_widget_lifecycle.py",
]
try:
    for f in files:
        (root / f).write_bytes(old[f])
    with (out / "2383-widget-replay-red.log").open("w") as log:
        r = subprocess.run(
            cmd + ["--junitxml=" + str(out / "2383-widget-replay-red.xml")],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    assert r.returncode == 1, r.returncode
finally:
    for f in files:
        (root / f).write_bytes(fixed[f])
with (out / "2383-widget-replay-restored.log").open("w") as log:
    r = subprocess.run(
        cmd + ["--junitxml=" + str(out / "2383-widget-replay-restored.xml")],
        cwd=root,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
assert r.returncode == 0, r.returncode
(out / "2383-widget-replay-causal.json").write_text(
    json.dumps(
        {
            "command": cmd,
            "old_head": "c81e14368ceca68ac6c95bfb98a41e12fa229bef",
            "old_rc": 1,
            "restored_rc": 0,
            "test_sha256": hashlib.sha256(
                (root / "tests/test_dashboard_provider_widget_lifecycle.py").read_bytes()
            ).hexdigest(),
            "sources": {
                f: {
                    "old": hashlib.sha256(old[f]).hexdigest(),
                    "fixed": hashlib.sha256(fixed[f]).hexdigest(),
                    "restored": hashlib.sha256((root / f).read_bytes()).hexdigest(),
                }
                for f in files
            },
        },
        indent=2,
    )
)
print("REAL STREAMLIT RED2/RESTORED2 PASS")
