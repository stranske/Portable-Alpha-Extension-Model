import hashlib
import json
import subprocess
from pathlib import Path

root = Path(
    "/Users/teacher/.codex/automations/pd-workloop-resume/worktrees/portable-2366-credential-isolation-0900"
)
out = Path(__file__).parent
files = ["dashboard/components/explain_results.py", "dashboard/components/comparison_llm.py"]
fixed = {f: (root / f).read_bytes() for f in files}
old = {f: subprocess.check_output(["git", "-C", str(root), "show", "c81e143:" + f]) for f in files}
cmd = [
    "/Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python",
    "-m",
    "pytest",
    "tests/test_dashboard_provider_widget_lifecycle.py",
]
try:
    for f in files:
        (root / f).write_bytes(old[f])
    with (out / "2383-widget-red.log").open("w") as log:
        r = subprocess.run(
            cmd + ["--junitxml=" + str(out / "2383-widget-red.xml")],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    assert r.returncode == 1, r.returncode
finally:
    for f in files:
        (root / f).write_bytes(fixed[f])
with (out / "2383-widget-restored.log").open("w") as log:
    r = subprocess.run(
        cmd + ["--junitxml=" + str(out / "2383-widget-restored.xml")],
        cwd=root,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
assert r.returncode == 0, r.returncode
(out / "2383-widget-causal.json").write_text(
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
