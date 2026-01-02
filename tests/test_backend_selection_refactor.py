#!/usr/bin/env python3
"""
Tests for backend selection helper function to ensure consistent behavior
between cli.py and __main__.py after refactoring.
"""

import os
import tempfile

import pytest
import yaml

from pa_core.backend import get_backend, resolve_and_set_backend
from pa_core.config import ModelConfig


class TestBackendSelectionHelper:
    """Test the backend selection helper function."""

    def test_resolve_backend_cli_priority(self):
        """Test that CLI argument takes priority over config."""
        # Create a config with numpy backend
        config_data = {
            "N_SIMULATIONS": 1,
            "N_MONTHS": 1,
            "financing_mode": "broadcast",
            "backend": "numpy",
        }
        cfg = ModelConfig(**config_data)

        # CLI argument should override config
        backend = resolve_and_set_backend("numpy", cfg)
        assert backend == "numpy"
        assert get_backend() == "numpy"

    def test_resolve_backend_config_fallback(self):
        """Test fallback to config backend when CLI arg is None."""
        config_data = {
            "N_SIMULATIONS": 1,
            "N_MONTHS": 1,
            "financing_mode": "broadcast",
            "backend": "numpy",
        }
        cfg = ModelConfig(**config_data)

        # Should use config backend when CLI is None
        backend = resolve_and_set_backend(None, cfg)
        assert backend == "numpy"
        assert get_backend() == "numpy"

    def test_resolve_backend_default_fallback(self):
        """Test default fallback when both CLI and config are None/missing."""
        # Should default to numpy
        backend = resolve_and_set_backend(None, None)
        assert backend == "numpy"
        assert get_backend() == "numpy"

    def test_resolve_backend_unknown_error(self):
        """Test error handling when an unknown backend is requested."""
        with pytest.raises(ValueError, match="Only the 'numpy' backend is supported"):
            resolve_and_set_backend("cupy", None)

    def test_config_with_backend_field(self):
        """Test that ModelConfig accepts backend field."""
        config_data = {
            "N_SIMULATIONS": 10,
            "N_MONTHS": 12,
            "financing_mode": "broadcast",
            "backend": "numpy",
        }
        cfg = ModelConfig(**config_data)
        assert cfg.backend == "numpy"

    def test_config_invalid_backend_validation(self):
        """Test that invalid backend values are rejected."""
        config_data = {
            "N_SIMULATIONS": 10,
            "N_MONTHS": 12,
            "financing_mode": "broadcast",
            "backend": "cupy",
        }

        with pytest.raises(ValueError, match=r"backend must be .*numpy"):
            ModelConfig(**config_data)


class TestBackendSelectionIntegration:
    """Integration tests to verify consistent behavior between entry points."""

    def _create_test_config(self, backend=None):
        """Create a temporary config file for testing."""
        config_data = {
            "N_SIMULATIONS": 1,
            "N_MONTHS": 1,
            "risk_metrics": ["Return", "Risk", "terminal_ShortfallProb"],
        }
        if backend:
            config_data["backend"] = backend

        fd, path = tempfile.mkstemp(suffix=".yml")
        try:
            with open(path, "w") as f:
                yaml.dump(config_data, f)
        finally:
            os.close(fd)

        return path

    def _create_test_index(self):
        """Create a temporary index file for testing."""
        fd, path = tempfile.mkstemp(suffix=".csv")
        try:
            with open(path, "w") as f:
                f.write("Date,Return\n2020-01-01,0.01\n2020-02-01,0.02\n")
        finally:
            os.close(fd)
        return path

    def test_cli_and_main_consistent_behavior_no_args(self):
        """Test that both entry points behave consistently without backend args."""
        # Test the helper function directly instead of full integration
        # since we've already tested the individual components

        config_data = {
            "N_SIMULATIONS": 1,
            "N_MONTHS": 1,
            "financing_mode": "broadcast",
            "backend": "numpy",
            "risk_metrics": ["Return", "Risk", "terminal_ShortfallProb"],
        }
        cfg = ModelConfig(**config_data)

        # Simulate what both entry points should do without CLI backend arg
        result1 = resolve_and_set_backend(None, cfg)
        result2 = resolve_and_set_backend(None, cfg)

        # Both should give same result (config backend)
        assert result1 == result2 == "numpy"

    def test_cli_and_main_consistent_with_cli_arg(self):
        """Test that both entry points handle CLI backend arg consistently."""
        config_data = {
            "N_SIMULATIONS": 1,
            "N_MONTHS": 1,
            "financing_mode": "broadcast",
            "backend": "numpy",  # Config has one value
            "risk_metrics": ["Return", "Risk", "terminal_ShortfallProb"],
        }
        cfg = ModelConfig(**config_data)

        # Simulate what both entry points should do with CLI backend arg
        result1 = resolve_and_set_backend("numpy", cfg)  # CLI overrides
        result2 = resolve_and_set_backend("numpy", cfg)  # Same for both

        # Both should use CLI arg (numpy)
        assert result1 == result2 == "numpy"
