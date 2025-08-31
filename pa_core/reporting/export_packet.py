"""Committee-ready export packet generator.

Creates comprehensive PPTX and Excel reports for committee presentation,
combining executive summary, charts, detailed data, and appendices.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd
import plotly.graph_objects as go
from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from .excel import export_to_excel

__all__ = ["create_export_packet"]


def _create_title_slide(pres: Presentation, title: str = "Portable Alpha Analysis") -> None:
    """Create a professional title slide."""
    slide_layout = pres.slide_layouts[0]  # Title slide layout
    slide = pres.slides.add_slide(slide_layout)
    
    title_shape = slide.shapes.title
    subtitle_shape = slide.placeholders[1]
    
    title_shape.text = title
    subtitle_shape.text = "Investment Committee Report"
    
    # Style the title
    title_shape.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    title_shape.text_frame.paragraphs[0].font.size = Pt(36)
    title_shape.text_frame.paragraphs[0].font.bold = True


def _create_executive_summary_slide(
    pres: Presentation, 
    summary_df: pd.DataFrame,
    inputs_dict: Dict[str, Any],
) -> None:
    """Create executive summary slide with key metrics."""
    slide_layout = pres.slide_layouts[1]  # Title and Content layout
    slide = pres.slides.add_slide(slide_layout)
    
    title_shape = slide.shapes.title
    title_shape.text = "Executive Summary"
    
    # Content placeholder
    content_placeholder = slide.placeholders[1]
    tf = content_placeholder.text_frame
    tf.clear()
    
    # Add key metrics
    p = tf.paragraphs[0]
    p.text = "Key Results:"
    p.font.bold = True
    p.font.size = Pt(18)
    
    # Extract key metrics from summary
    if not summary_df.empty:
        first_row = summary_df.iloc[0]
        
        metrics = []
        if "AnnReturn" in first_row:
            metrics.append(f"• Annualized Return: {first_row['AnnReturn']:.2%}")
        if "AnnVol" in first_row:
            metrics.append(f"• Annualized Volatility: {first_row['AnnVol']:.2%}")
        if "ShortfallProb" in first_row:
            metrics.append(f"• Shortfall Probability: {first_row['ShortfallProb']:.2%}")
        if "VaR" in first_row:
            metrics.append(f"• Value at Risk (95%): {first_row['VaR']:.2%}")
        
        for metric in metrics:
            p = tf.add_paragraph()
            p.text = metric
            p.font.size = Pt(14)
            p.level = 0
    
    # Add key parameters
    p = tf.add_paragraph()
    p.text = "\nKey Parameters:"
    p.font.bold = True
    p.font.size = Pt(18)
    
    key_params = ["N_SIMULATIONS", "N_MONTHS", "analysis_mode"]
    for param in key_params:
        if param in inputs_dict:
            p = tf.add_paragraph()
            p.text = f"• {param}: {inputs_dict[param]}"
            p.font.size = Pt(14)
            p.level = 0


def _create_appendix_slide(pres: Presentation, inputs_dict: Dict[str, Any]) -> None:
    """Create appendix slide with detailed parameters."""
    slide_layout = pres.slide_layouts[1]  # Title and Content layout
    slide = pres.slides.add_slide(slide_layout)
    
    title_shape = slide.shapes.title
    title_shape.text = "Appendix: Model Parameters"
    
    # Content placeholder
    content_placeholder = slide.placeholders[1]
    tf = content_placeholder.text_frame
    tf.clear()
    
    # Add parameters in groups
    param_groups = {
        "Simulation Settings": ["N_SIMULATIONS", "N_MONTHS", "analysis_mode"],
        "Return Assumptions": ["mu_H", "mu_E", "mu_M"],
        "Volatility Assumptions": ["sigma_H", "sigma_E", "sigma_M"],
        "Correlation Assumptions": ["rho_idx_H", "rho_idx_E", "rho_idx_M", "rho_H_E", "rho_H_M", "rho_E_M"],
    }
    
    p = tf.paragraphs[0]
    p.text = ""  # Clear default text
    
    for group_name, params in param_groups.items():
        # Group header
        p = tf.add_paragraph()
        p.text = f"{group_name}:"
        p.font.bold = True
        p.font.size = Pt(14)
        
        # Parameters in this group
        for param in params:
            if param in inputs_dict:
                value = inputs_dict[param]
                if isinstance(value, float):
                    display_value = f"{value * 100:.2f}%"
                else:
                    display_value = str(value)
            else:
                display_value = "N/A"

            p = tf.add_paragraph()
            p.text = f"  {param}: {display_value}"
            p.font.size = Pt(12)
            p.level = 1


def _create_comprehensive_pptx(
    figs: list[go.Figure],
    summary_df: pd.DataFrame,
    inputs_dict: Dict[str, Any],
    path: str | Path,
    *,
    alt_texts: Sequence[str] | None = None,
) -> None:
    """Create a comprehensive PPTX with title, executive summary, charts, and appendix."""
    pres = Presentation()
    
    # Title slide
    _create_title_slide(pres)
    
    # Executive summary slide
    _create_executive_summary_slide(pres, summary_df, inputs_dict)
    
    # Chart slides - reuse existing functionality but with enhanced error handling
    alt_iter = iter(alt_texts) if alt_texts is not None else None
    
    for fig in figs:
        slide = pres.slides.add_slide(pres.slide_layouts[5])  # Blank layout
        
        try:
            img_bytes = fig.to_image(format="png")
            pic = slide.shapes.add_picture(io.BytesIO(img_bytes), Inches(0), Inches(0))
            
            # Add title from figure
            if fig.layout.title and fig.layout.title.text:
                # Add title textbox
                title_box = slide.shapes.add_textbox(Inches(0), Inches(0), Inches(10), Inches(1))
                title_frame = title_box.text_frame
                title_frame.text = str(fig.layout.title.text)
                title_frame.paragraphs[0].font.size = Pt(20)
                title_frame.paragraphs[0].font.bold = True
                title_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
            
            # Add alt text
            alt = next(alt_iter, None) if alt_iter else None
            if not alt and fig.layout.title and fig.layout.title.text:
                alt = str(fig.layout.title.text)
            
            if alt:
                el = pic._element.xpath("./p:nvPicPr/p:cNvPr")[0]
                el.set("descr", alt)
                
        except Exception as e:
            # Create an error slide instead of failing silently
            slide = pres.slides.add_slide(pres.slide_layouts[1])  # Title and Content
            title_shape = slide.shapes.title
            title_shape.text = "Chart Export Error"
            
            content_placeholder = slide.placeholders[1]
            tf = content_placeholder.text_frame
            tf.text = f"Unable to export chart: {str(e)}\n\nThis may be due to missing Chrome/Chromium installation."
    
    # Appendix slide
    _create_appendix_slide(pres, inputs_dict)
    
    pres.save(str(path))


def _create_enhanced_excel(
    inputs_dict: Dict[str, Any],
    summary_df: pd.DataFrame,
    raw_returns_dict: Dict[str, Any],
    filename: str,
    *,
    pivot: bool = False,
) -> None:
    """Create enhanced Excel file with additional committee-ready formatting."""
    # Use existing excel export as base
    export_to_excel(inputs_dict, summary_df, raw_returns_dict, filename, pivot=pivot)
    
    # Add additional formatting and sheets for committee presentation
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
        
        wb = openpyxl.load_workbook(filename)
        
        # Enhance Summary sheet formatting
        if "Summary" in wb.sheetnames:
            ws = wb["Summary"]
            
            # Style headers
            header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            
            for cell in ws[1]:  # First row (headers)
                cell.font = Font(bold=True, size=12, color="FFFFFF")
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center")
            
            # Auto-adjust column widths
            for column_cells in ws.columns:
                length = max(len(str(cell.value) or "") for cell in column_cells)
                ws.column_dimensions[get_column_letter(column_cells[0].column)].width = min(length + 2, 50)
        
        # Create Executive Summary sheet
        if "Executive Summary" not in wb.sheetnames:
            exec_ws = wb.create_sheet("Executive Summary", 0)  # Insert as first sheet
            
            exec_ws["A1"] = "Portable Alpha Analysis - Executive Summary"
            exec_ws["A1"].font = Font(bold=True, size=16)
            exec_ws.merge_cells("A1:D1")
            
            row = 3
            exec_ws[f"A{row}"] = "Key Metrics"
            exec_ws[f"A{row}"].font = Font(bold=True, size=14)
            row += 1
            
            if not summary_df.empty:
                first_row = summary_df.iloc[0]
                
                metrics = [
                    ("Annualized Return", "AnnReturn", "0.00%"),
                    ("Annualized Volatility", "AnnVol", "0.00%"),
                    ("Shortfall Probability", "ShortfallProb", "0.00%"),
                    ("Value at Risk (95%)", "VaR", "0.00%"),
                ]
                
                for label, col, fmt in metrics:
                    if col in first_row:
                        exec_ws[f"A{row}"] = label
                        exec_ws[f"B{row}"] = first_row[col]
                        exec_ws[f"B{row}"].number_format = fmt
                        row += 1
            
            row += 2
            exec_ws[f"A{row}"] = "Model Configuration"
            exec_ws[f"A{row}"].font = Font(bold=True, size=14)
            row += 1
            
            key_params = [
                ("Simulations", "N_SIMULATIONS"),
                ("Time Horizon (Months)", "N_MONTHS"),
                ("Analysis Mode", "analysis_mode"),
            ]
            
            for label, param in key_params:
                if param in inputs_dict:
                    exec_ws[f"A{row}"] = label
                    exec_ws[f"B{row}"] = inputs_dict[param]
                    row += 1
        
        wb.save(filename)
        
    except Exception:
        # If enhanced formatting fails, at least we have the basic export
        pass


def create_export_packet(
    figs: list[go.Figure],
    summary_df: pd.DataFrame,
    raw_returns_dict: Dict[str, Any],
    inputs_dict: Dict[str, Any],
    base_filename: str = "committee_packet",
    *,
    alt_texts: Sequence[str] | None = None,
    pivot: bool = False,
) -> tuple[str, str]:
    """Create a comprehensive committee-ready export packet.
    
    Parameters
    ----------
    figs : list[go.Figure]
        Plotly figures to include in the PPTX.
    summary_df : pd.DataFrame
        Summary metrics table.
    raw_returns_dict : Dict[str, Any]
        Raw returns data for Excel export.
    inputs_dict : Dict[str, Any]
        Input parameters for documentation.
    base_filename : str, optional
        Base filename (without extension). Default: "committee_packet".
    alt_texts : Sequence[str] | None, optional
        Alt text descriptions for figures.
    pivot : bool, optional
        Whether to use pivot format in Excel export.
    
    Returns
    -------
    tuple[str, str]
        Paths to the created PPTX and Excel files.
    
    Raises
    ------
    RuntimeError
        If Chrome/Chromium is not available for image generation.
    """
    pptx_path = f"{base_filename}.pptx"
    excel_path = f"{base_filename}.xlsx"
    
    # Create enhanced PPTX
    try:
        _create_comprehensive_pptx(figs, summary_df, inputs_dict, pptx_path, alt_texts=alt_texts)
    except Exception as e:
        if "Chrome" in str(e) or "Kaleido" in str(e) or "Chromium" in str(e):
            raise RuntimeError(
                "📷 Export packet requires Chrome/Chromium for chart generation.\n"
                "Install with: sudo apt-get install chromium-browser\n"
                "Or on macOS: brew install chromium\n"
                "On Windows: Download from https://www.google.com/chrome/"
            ) from e
        else:
            raise RuntimeError(f"Failed to create PPTX export: {str(e)}") from e
    
    # Create enhanced Excel
    try:
        _create_enhanced_excel(inputs_dict, summary_df, raw_returns_dict, excel_path, pivot=pivot)
    except Exception as e:
        raise RuntimeError(f"Failed to create Excel export: {str(e)}") from e
    
    return pptx_path, excel_path