"""Lock Gate pull_request triggers without labeled (Workflows#4082).

``pr-00-gate.yml`` is create-only, so the WF#3983 template fix never synced.
Label bursts must not restart Gate (concurrency cancel-in-progress); the
workflow must still run on opened / synchronize / reopened / ready_for_review.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-00-gate.yml"

# Events that must still start Gate (code / review-state changes).
REQUIRED_PULL_REQUEST_TYPES = (
    "opened",
    "synchronize",
    "reopened",
    "ready_for_review",
)

# Label-only traffic must not start a new Gate run.
FORBIDDEN_PULL_REQUEST_TYPES = ("labeled", "unlabeled")


def _gate_pull_request_types() -> list[str]:
    document = yaml.safe_load(GATE_WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML 1.1 treats bare `on:` as boolean True.
    on_block = document.get("on", document.get(True))
    assert isinstance(on_block, dict), f"{GATE_WORKFLOW.name} missing on: block"
    pull_request = on_block["pull_request"]
    types = pull_request["types"]
    assert isinstance(types, list), f"expected list of trigger types, got {type(types)!r}"
    return [str(item) for item in types]


def test_gate_runs_on_opened_synchronize_reopened_ready_for_review() -> None:
    types = _gate_pull_request_types()
    missing = [event for event in REQUIRED_PULL_REQUEST_TYPES if event not in types]
    assert not missing, (
        f"{GATE_WORKFLOW.name} must trigger Gate on {REQUIRED_PULL_REQUEST_TYPES}; "
        f"missing={missing!r} actual={types!r}"
    )


def test_label_only_events_do_not_start_gate() -> None:
    types = _gate_pull_request_types()
    present = [event for event in FORBIDDEN_PULL_REQUEST_TYPES if event in types]
    assert not present, (
        f"{GATE_WORKFLOW.name} must not start Gate on label-only events; "
        f"forbidden present={present!r} actual={types!r}"
    )
