"""Smoke tests for the static PA tool descriptor."""

from __future__ import annotations

import importlib
import json
import socket
import subprocess
import sys
from pathlib import Path

from pa_core.pa import main as pa_main
from pa_core.tool_descriptor import get_tool_descriptor


def test_describe_emits_zones(capsys) -> None:
    pa_main(["describe"])

    payload = json.loads(capsys.readouterr().out)

    assert {"network", "data_zones", "permissions"} <= payload.keys()
    zone_ids = {zone["id"] for zone in payload["data_zones"]}
    assert {"deterministic", "llm", "filesystem"} <= zone_ids
    llm_zone = next(zone for zone in payload["data_zones"] if zone["id"] == "llm")
    assert llm_zone["network"] == "gated-no-train"
    assert "no-train" in llm_zone["boundary"]
    assert payload["network"]["llm"] == "gated-no-train"


def test_llm_zone_consistent_with_no_network_guarantee(socket_connect_guard) -> None:
    attempts, blocked = socket_connect_guard
    descriptor = get_tool_descriptor()

    for name in list(sys.modules):
        if name == "pa_core.llm" or name.startswith("pa_core.llm."):
            sys.modules.pop(name)

    importlib.import_module("pa_core.llm")

    assert descriptor["network"]["deterministic"] == "offline"
    assert socket.socket.connect is blocked
    assert attempts == []


def test_describe_does_not_import_numpy_or_pandas() -> None:
    script = """
import sys
from pa_core.pa import main

main(["describe"])
if "numpy" in sys.modules or "pandas" in sys.modules:
    raise SystemExit("describe imported numpy or pandas")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert json.loads(result.stdout)["network"]["deterministic"] == "offline"
