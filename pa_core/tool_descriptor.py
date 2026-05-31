"""Static tool descriptor for backplane and privacy-boundary discovery."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_DESCRIPTOR: dict[str, Any] = {
    "schema_version": "pa-tool-descriptor/v1",
    "tool": {
        "name": "portable-alpha-extension-model",
        "command": "pa",
        "description": (
            "Offline deterministic portable-alpha simulations with optional "
            "gated LLM dashboard helpers."
        ),
    },
    "network": {
        "deterministic": "offline",
        "llm": "gated-no-train",
    },
    "data_zones": [
        {
            "id": "deterministic",
            "name": "Deterministic local engine",
            "data_sensitivity": "real-or-proprietary",
            "network": "offline",
            "default_enabled": True,
            "boundary": (
                "Runs locally with the numpy backend and does not require network egress "
                "for simulation, validation, sweep, or calibration commands."
            ),
        },
        {
            "id": "llm",
            "name": "Optional LLM dashboard helpers",
            "data_sensitivity": "synthetic-or-redacted-only",
            "network": "gated-no-train",
            "default_enabled": False,
            "boundary": (
                "Disabled for proprietary data unless an authorized no-train endpoint "
                "with redaction is explicitly configured."
            ),
        },
        {
            "id": "filesystem",
            "name": "Declared filesystem I/O",
            "data_sensitivity": "depends-on-inputs",
            "network": "local-filesystem",
            "default_enabled": True,
            "boundary": (
                "Reads declared config/index files and writes run artifacts under "
                "declared output paths."
            ),
        },
    ],
    "permissions": {
        "reads": [
            "--config",
            "--index",
            "--input",
            "--returns",
            "scenario registry entries",
        ],
        "writes": [
            "Outputs.xlsx",
            "Sweep.xlsx",
            "runs/<run_id>/",
            ".pa_registry/",
            "asset_library.yaml",
        ],
        "network": {
            "deterministic": "offline",
            "llm": "requires explicit authorized no-train provider configuration and redaction",
        },
    },
}


def get_tool_descriptor() -> dict[str, Any]:
    """Return a mutable copy of the static tool descriptor."""

    return deepcopy(_DESCRIPTOR)
