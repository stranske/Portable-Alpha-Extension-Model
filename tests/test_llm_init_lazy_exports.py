"""Tests for lazy export behavior in pa_core.llm package init."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest


@pytest.fixture
def clear_llm_modules() -> None:
    for name in list(sys.modules):
        if name == "pa_core.llm" or name.startswith("pa_core.llm."):
            sys.modules.pop(name, None)


def test_import_tracing_submodule_does_not_require_pandas(
    monkeypatch: pytest.MonkeyPatch, clear_llm_modules
):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "pandas":
            raise AssertionError("pandas import should not be triggered")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    tracing = importlib.import_module("pa_core.llm.tracing")
    assert hasattr(tracing, "resolve_trace_url")


def test_package_reexport_resolve_trace_url_is_lazy(
    monkeypatch: pytest.MonkeyPatch, clear_llm_modules
):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "pandas":
            raise AssertionError("pandas import should not be triggered")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    llm = importlib.import_module("pa_core.llm")
    resolver = llm.resolve_trace_url
    assert callable(resolver)
    assert resolver("abc-123") == "https://smith.langchain.com/r/abc-123"


def test_package_unknown_attr_raises_attribute_error(clear_llm_modules):
    llm = importlib.import_module("pa_core.llm")
    with pytest.raises(AttributeError):
        _ = llm.not_a_real_export
