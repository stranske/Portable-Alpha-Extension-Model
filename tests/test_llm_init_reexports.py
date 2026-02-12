"""Tests for pa_core.llm package-level lazy re-exports."""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def clear_llm_modules() -> None:
    for name in list(sys.modules):
        if name == "pa_core.llm" or name.startswith("pa_core.llm."):
            sys.modules.pop(name, None)


def test_llm_getattr_resolves_known_lazy_export_identity(clear_llm_modules) -> None:
    llm = importlib.import_module("pa_core.llm")
    tracing = importlib.import_module("pa_core.llm.tracing")

    assert llm.resolve_trace_url is tracing.resolve_trace_url


def test_llm_getattr_unknown_attribute_error_includes_missing_name(clear_llm_modules) -> None:
    llm = importlib.import_module("pa_core.llm")
    missing_name = "not_a_real_export_name"

    with pytest.raises(AttributeError) as exc_info:
        getattr(llm, missing_name)

    assert missing_name in str(exc_info.value)


def test_llm_introspection_patterns_do_not_raise(clear_llm_modules) -> None:
    llm = importlib.import_module("pa_core.llm")

    names = dir(llm)
    has_resolver = hasattr(llm, "resolve_trace_url")
    wrapped = getattr(llm, "__wrapped__", None)

    assert "resolve_trace_url" in names
    assert has_resolver is True
    assert wrapped is None
