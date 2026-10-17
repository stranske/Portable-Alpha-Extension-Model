import subprocess
import json
import hashlib
import xml.etree.ElementTree as E
from pathlib import Path

root = Path(
    "/Users/teacher/.codex/automations/pd-workloop-resume/worktrees/portable-2366-credential-isolation-0900"
)
out = Path(__file__).parent
files = [
    "dashboard/components/llm_settings.py",
    "dashboard/components/explain_results.py",
    "dashboard/components/comparison_llm.py",
    "dashboard/pages/3_Scenario_Wizard.py",
]
fixed = {f: (root / f).read_bytes() for f in files}
original = {
    f: subprocess.check_output(["git", "-C", str(root), "show", "origin/main:" + f]) for f in files
}
cmd = [
    "/Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python",
    "-m",
    "pytest",
    "tests/test_dashboard_llm_settings.py",
    "tests/test_dashboard_explain_results.py",
    "tests/test_dashboard_comparison_llm.py",
    "tests/test_llm_provider_missing_keys.py",
    "tests/test_wizard_config_chat_flow.py",
    "-k",
    "provider_defaults_switches or tracing_only or wizard_initial_provider or langsmith_is_never",
]
try:
    for f in files:
        (root / f).write_bytes(original[f])
    with (out / "2366-original-red.log").open("w") as log:
        r = subprocess.run(
            cmd + ["--junitxml=" + str(out / "2366-original-red.xml")],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    assert r.returncode == 1, r.returncode
finally:
    for f in files:
        (root / f).write_bytes(fixed[f])
with (out / "2366-restored.log").open("w") as log:
    r = subprocess.run(
        cmd + ["--junitxml=" + str(out / "2366-restored.xml")],
        cwd=root,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
assert r.returncode == 0, r.returncode
red = list(E.parse(out / "2366-original-red.xml").getroot().iter("testsuite"))[0]
assert red.get("errors") == "0" and int(red.get("failures")) > 0, red.attrib
proof = {
    "command": cmd,
    "original_source_ref": "origin/main at b6a4c1a",
    "red": red.attrib,
    "restored_returncode": 0,
    "sources": {
        f: {
            "original": hashlib.sha256(original[f]).hexdigest(),
            "fixed": hashlib.sha256(fixed[f]).hexdigest(),
            "restored": hashlib.sha256((root / f).read_bytes()).hexdigest(),
        }
        for f in files
    },
}
(out / "2366-causal.json").write_text(json.dumps(proof, indent=2))
print("CAUSAL", red.attrib)
