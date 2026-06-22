"""Import-safety checks for pa_core.llm."""

from __future__ import annotations

import importlib
import socket
import sys


def test_import_pa_core_llm_no_network(socket_connect_guard):
    attempts, blocked = socket_connect_guard
    pa_core_pkg = sys.modules.get("pa_core")
    had_pa_core_llm_attr = hasattr(pa_core_pkg, "llm") if pa_core_pkg is not None else False
    original_pa_core_llm_attr = getattr(pa_core_pkg, "llm", None) if had_pa_core_llm_attr else None
    original_llm_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "pa_core.llm" or name.startswith("pa_core.llm.")
    }

    try:
        for name in list(sys.modules):
            if name == "pa_core.llm" or name.startswith("pa_core.llm."):
                sys.modules.pop(name)

        importlib.import_module("pa_core.llm")
    finally:
        for name in list(sys.modules):
            if name == "pa_core.llm" or name.startswith("pa_core.llm."):
                sys.modules.pop(name)
        sys.modules.update(original_llm_modules)
        for name, module in original_llm_modules.items():
            parent_name, _, child_name = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and child_name:
                setattr(parent, child_name, module)
        pa_core_pkg = sys.modules.get("pa_core")
        if pa_core_pkg is not None and not original_llm_modules:
            if had_pa_core_llm_attr:
                setattr(pa_core_pkg, "llm", original_pa_core_llm_attr)
            else:
                pa_core_pkg.__dict__.pop("llm", None)

    assert socket.socket.connect is blocked
    assert attempts == []
