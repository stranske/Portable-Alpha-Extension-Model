"""Re-run the eight replay.py PNG mutation pairs against current-main pa_core/cli.py bytes.

replay.py stays byte-identical (its hashes are bound by current-manifest.json). This wrapper
only rebinds its pinned revision/source digest and then calls its main(). It also exercises
the failure and timeout paths: in both, the private snapshot's pa_core/cli.py is hashed after
replay.py's own `finally` has run, before the temporary directory is deleted.

Usage: python replay_current.py <fresh-output-dir>
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

REVISION = "5be77e15c925f5354ab67e2a1749fe6187d2038c"
SOURCE_SHA256 = "f3ffec1a90fe5888ca5ba357507e7542f4dc436117acea430348c87ab0b40e26"
HERE = Path(__file__).resolve().parent
REPLAY = HERE.parent / "replay.py"
ROOT = HERE.parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_replay():
    spec = importlib.util.spec_from_file_location("replay", REPLAY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REVISION = REVISION
    module.SOURCE_SHA256 = SOURCE_SHA256
    return module


def run_path(name: str, out: Path, inject) -> dict[str, object]:
    """Run replay.main() with an injected subprocess fault; hash the private source on exit."""
    replay = load_replay()
    observed: list[str] = []

    @contextmanager
    def keep_until_hashed(prefix=None):
        path = tempfile.mkdtemp(prefix=prefix)
        try:
            yield path
        finally:
            observed.append(sha256(Path(path) / "pa_core/cli.py"))
            shutil.rmtree(path)

    real_run = subprocess.run
    calls = {"n": 0}

    def faulty_run(argv, **kwargs):
        if "pytest" not in argv:  # git archive also goes through subprocess.run
            return real_run(argv, **kwargs)
        calls["n"] += 1
        return inject(real_run, argv, kwargs, calls["n"])

    real_tempdir = replay.tempfile.TemporaryDirectory
    replay.tempfile.TemporaryDirectory = keep_until_hashed
    replay.subprocess.run = faulty_run
    sys.argv = ["replay.py", "--output", str(out / name)]
    worktree_before = sha256(ROOT / "pa_core/cli.py")
    raised = None
    try:
        replay.main()
    except BaseException as exc:  # noqa: BLE001 - the raised type is the evidence
        raised = f"{type(exc).__name__}: {exc}"
    finally:
        replay.subprocess.run = real_run
        replay.tempfile.TemporaryDirectory = real_tempdir
    return {
        "path": name,
        "raised": raised,
        "private_cli_sha256_after_finally": list(observed),
        "restored_equals_baseline": observed == [SOURCE_SHA256],
        "worktree_cli_sha256_before": worktree_before,
        "worktree_cli_sha256_after": sha256(ROOT / "pa_core/cli.py"),
    }


def timeout_on_first_red(real_run, argv, kwargs, n):
    if n == 1:
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))
    return real_run(argv, **kwargs)


def fail_on_first_red(real_run, argv, kwargs, n):
    if n == 1:
        raise RuntimeError("injected failure while the first mutant is in place")
    return real_run(argv, **kwargs)


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        run_path("timeout-path", out, timeout_on_first_red),
        run_path("failure-path", out, fail_on_first_red),
    ]
    (out / "restoration-paths.json").write_text(json.dumps(paths, indent=2) + "\n")
    replay = load_replay()
    sys.argv = ["replay.py", "--output", str(out / "pairs")]
    replay.main()
    ok = all(p["raised"] and p["restored_equals_baseline"] for p in paths)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
