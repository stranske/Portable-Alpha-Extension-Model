"""Guided interface for running a full simulation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from dashboard.app import _DEF_THEME, _DEF_XLSX, apply_theme
from pa_core import cli as pa_cli


def _write_temp(uploaded: st.runtime.uploaded_file_manager.UploadedFile, suffix: str) -> str:
    """Write *uploaded* content to a temporary file with *suffix* and return path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        return tmp.name


def main() -> None:
    st.title("Scenario Wizard")
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)

    st.write("Upload a scenario configuration and index returns to run a simulation.")
    cfg = st.file_uploader("Scenario YAML", type=["yaml", "yml"])
    idx = st.file_uploader("Index CSV", type=["csv"])
    output = st.text_input("Output workbook", _DEF_XLSX)

    if st.button("Run"):
        if cfg is None or idx is None:
            st.warning("Please upload both files before running.")
            return

        cfg_path = _write_temp(cfg, ".yaml")
        idx_path = _write_temp(idx, ".csv")
        try:
            pa_cli.main(["--config", cfg_path, "--index", idx_path, "--output", output])
        except Exception as exc:  # pragma: no cover - runtime feedback
            st.error(str(exc))
        else:
            st.success(f"Run complete. Results written to {output}.")
            st.page_link("pages/4_Results.py", label="View results")
        finally:
            Path(cfg_path).unlink(missing_ok=True)
            Path(idx_path).unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
