"""Shared test fixtures and utilities for test dependency setup.

This module provides utilities to eliminate code duplication in test module setup.
Rather than manually manipulating sys.path and sys.modules in individual test files,
tests should use these shared utilities or rely on proper PYTHONPATH setup.
"""

import runpy
import socket
import sys
from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def dashboard_module_loader(project_root: Path):
    """Factory fixture for loading dashboard modules via runpy.

    This replaces the common pattern of:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        module = runpy.run_path("dashboard/pages/1_Asset_Library.py", run_name="test_module")

    Usage:
        def test_something(dashboard_module_loader):
            module = dashboard_module_loader("dashboard/pages/1_Asset_Library.py")
            logger = module["logger"]
    """

    def _load_module(module_path: str, run_name: str = "test_module") -> Dict[str, Any]:
        """Load a dashboard module using runpy with proper path setup."""
        # Temporarily add project root to sys.path if not already there
        root_str = str(project_root)
        path_added = False
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
            path_added = True

        try:
            return runpy.run_path(module_path, run_name=run_name)
        finally:
            # Clean up sys.path if we added to it
            if path_added:
                sys.path.remove(root_str)

    return _load_module


def ensure_pa_core_importable() -> None:
    """Ensure pa_core module is importable without manual sys.path manipulation.

    This function should be used instead of manual module setup patterns.
    However, the preferred approach is to use proper PYTHONPATH setup
    when running tests (e.g., PYTHONPATH=$PWD python -m pytest).
    """
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)


ensure_pa_core_importable()


@pytest.fixture
def socket_connect_guard(monkeypatch: pytest.MonkeyPatch):
    """Block all socket.connect calls and track attempts.

    Returns a tuple of (attempts_list, blocked_function) so tests can assert
    that the guard was active and no connections were attempted.
    """
    attempts: list[object] = []

    def _blocked_connect(self, address):  # noqa: ANN001
        attempts.append(address)
        raise AssertionError("socket.connect should not be called during this test")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    return attempts, _blocked_connect


@pytest.fixture(autouse=True)
def _restore_viz_theme_globals():
    """Isolate the shared plotly theme so tests cannot leak palette state.

    ``pa_core.viz.theme`` keeps module-level globals (``TEMPLATE``, the colorway,
    fonts, backgrounds and thresholds) that ``reload_theme``/``reload_thresholds``
    -- and ``dashboard.app.apply_theme`` -- reassign in place. Tests such as
    ``test_apply_theme`` and ``test_theme_reload`` rewrite the colorway to a short
    custom palette and never restore it, which silently leaks into later tests that
    assert on colour assignment (e.g. the risk-return sleeve legend, which expects
    one distinct colour per sleeve). Snapshot the globals before each test and
    restore them afterwards so theme state stays isolated regardless of test order.
    """
    from pa_core.viz import theme

    attrs = ("TEMPLATE", "_COLORWAY", "_FONT", "_PAPER_BG", "_PLOT_BG", "THRESHOLDS")
    saved = {name: getattr(theme, name) for name in attrs if hasattr(theme, name)}
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(theme, name, value)
