"""Tests for export packet functionality."""

import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from pa_core.reporting.export_packet import create_export_packet


def create_test_data():
    """Create test data for export packet tests."""
    summary_df = pd.DataFrame(
        {
            "AnnReturn": [0.05, 0.07, 0.06],
            "AnnVol": [0.12, 0.15, 0.13],
            "ShortfallProb": [0.1, 0.15, 0.12],
            "VaR": [0.08, 0.12, 0.10],
        }
    )

    raw_returns_dict = {"Summary": summary_df}

    inputs_dict = {"N_SIMULATIONS": 1000, "N_MONTHS": 12, "analysis_mode": "returns"}

    # Create a simple test figure
    fig = go.Figure()
    fig.add_scatter(
        x=summary_df["AnnVol"],
        y=summary_df["AnnReturn"],
        mode="markers",
        name="Risk-Return",
    )
    fig.update_layout(
        title="Risk-Return Analysis", xaxis_title="Volatility", yaxis_title="Return"
    )

    return summary_df, raw_returns_dict, inputs_dict, fig


def test_export_packet_creates_files():
    """Test that export packet creates both PPTX and Excel files."""
    summary_df, raw_returns_dict, inputs_dict, fig = create_test_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            pptx_path, excel_path = create_export_packet(
                figs=[fig],
                summary_df=summary_df,
                raw_returns_dict=raw_returns_dict,
                inputs_dict=inputs_dict,
                base_filename=str(Path(tmpdir) / "test_packet"),
            )

            # Check that files were created
            assert Path(pptx_path).exists(), f"PPTX file not created: {pptx_path}"
            assert Path(excel_path).exists(), f"Excel file not created: {excel_path}"

            # Check file sizes (should be reasonable)
            pptx_size = Path(pptx_path).stat().st_size
            excel_size = Path(excel_path).stat().st_size

            assert pptx_size > 1000, f"PPTX file too small: {pptx_size} bytes"
            assert excel_size > 1000, f"Excel file too small: {excel_size} bytes"

        except RuntimeError as e:
            if "Chrome" in str(e) or "Chromium" in str(e) or "Kaleido" in str(e):
                pytest.skip("Chrome/Chromium not available in test environment")
            else:
                raise


def test_export_packet_with_alt_text():
    """Test that export packet works with alt text."""
    summary_df, raw_returns_dict, inputs_dict, fig = create_test_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            pptx_path, excel_path = create_export_packet(
                figs=[fig],
                summary_df=summary_df,
                raw_returns_dict=raw_returns_dict,
                inputs_dict=inputs_dict,
                base_filename=str(Path(tmpdir) / "test_packet_alt"),
                alt_texts=["Risk-return scatter plot for investment analysis"],
            )

            # Files should still be created successfully
            assert Path(pptx_path).exists()
            assert Path(excel_path).exists()

        except RuntimeError as e:
            if "Chrome" in str(e) or "Chromium" in str(e) or "Kaleido" in str(e):
                pytest.skip("Chrome/Chromium not available in test environment")
            else:
                raise


def test_export_packet_handles_empty_data():
    """Test that export packet handles edge cases gracefully."""
    # Empty summary
    empty_summary = pd.DataFrame()

    inputs_dict = {"test": "value"}
    raw_returns_dict = {"Empty": empty_summary}

    fig = go.Figure()
    fig.update_layout(title="Empty Chart")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            pptx_path, excel_path = create_export_packet(
                figs=[fig],
                summary_df=empty_summary,
                raw_returns_dict=raw_returns_dict,
                inputs_dict=inputs_dict,
                base_filename=str(Path(tmpdir) / "test_empty"),
            )

            # Should still create files even with empty data
            assert Path(pptx_path).exists()
            assert Path(excel_path).exists()

        except RuntimeError as e:
            if "Chrome" in str(e) or "Chromium" in str(e) or "Kaleido" in str(e):
                pytest.skip("Chrome/Chromium not available in test environment")
            else:
                raise


if __name__ == "__main__":
    # Run tests directly if executed as script
    try:
        test_export_packet_creates_files()
        test_export_packet_with_alt_text()
        test_export_packet_handles_empty_data()
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
