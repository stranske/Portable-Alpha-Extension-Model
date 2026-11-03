"""Focused coverage for the Results page's executable UI branches (issue #2381)."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pandas as pd


class _Container:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []

    def __enter__(self) -> _Container:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def metric(self, label: str, value: str, **_: object) -> None:
        self.metrics.append((label, value))

    def download_button(self, label: str, *_: object, **kwargs: object) -> None:
        self.downloads.append((label, str(kwargs.get("file_name"))))


class _ResultsStreamlit:
    def __init__(self) -> None:
        self.sidebar = self
        self.session_state: dict[str, object] = {}
        self.titles: list[str] = []
        self.subheaders: list[str] = []
        self.frames: list[pd.DataFrame] = []
        self.errors: list[str] = []
        self.charts: list[object] = []
        self.downloads: list[tuple[str, str]] = []
        self.column_containers: list[list[_Container]] = []
        self.tab_containers: list[list[_Container]] = []

    def title(self, value: str) -> None:
        self.titles.append(value)

    def subheader(self, value: str) -> None:
        self.subheaders.append(value)

    def columns(self, count: int) -> list[_Container]:
        containers = [_Container() for _ in range(count)]
        self.column_containers.append(containers)
        return containers

    def tabs(self, names: list[str]) -> list[_Container]:
        containers = [_Container() for _ in names]
        self.tab_containers.append(containers)
        return containers

    def slider(self, *_: object) -> int:
        return 1

    def multiselect(self, _: str, options: list[str], __: list[str]) -> list[str]:
        return options

    def number_input(self, *_: object, **__: object) -> float:
        return 0.0

    def checkbox(self, _: str) -> bool:
        return False

    def plotly_chart(self, figure: object, **_: object) -> None:
        self.charts.append(figure)

    def dataframe(self, value: pd.DataFrame) -> None:
        self.frames.append(value)

    def download_button(self, label: str, *_: object, **kwargs: object) -> None:
        self.downloads.append((label, str(kwargs.get("file_name"))))

    def caption(self, _: str) -> None:
        return None

    def info(self, _: str) -> None:
        return None

    def warning(self, _: str) -> None:
        return None

    def error(self, value: str) -> None:
        self.errors.append(value)

    def stop(self) -> None:
        raise AssertionError("stop should not be called for an existing workbook")


def _load_results_module() -> dict[str, Any]:
    """Load the Results page module and register it for coverage tracking."""
    import sys
    import types
    
    module_name = "dashboard.pages.page_4_Results"
    module_path = str(Path("dashboard/pages/4_Results.py"))
    module_dict = runpy.run_path(module_path, run_name=module_name)
    
    # Register the module in sys.modules so coverage tools can track it
    # Only register if not already present to avoid overwriting
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        module.__dict__.update(module_dict)
        sys.modules[module_name] = module
    
    return module_dict


class _CallRecorder:
    """Record function calls for verification."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls.append((self.__class__.__name__, args, kwargs))


def test_results_main_renders_metrics_plots_and_download_for_available_workbook(tmp_path) -> None:
    module = _load_results_module()
    fake_st = _ResultsStreamlit()
    workbook = tmp_path / "results.xlsx"
    workbook.write_bytes(b"workbook")
    summary = pd.DataFrame(
        {
            "Config": ["Base"],
            "monthly_TE": [0.02],
            "monthly_CVaR": [-0.01],
            "monthly_BreachProb": [0.03],
            "Agent": ["Total"],
            "terminal_AnnReturn": [0.04],
        }
    )
    paths = pd.DataFrame([[0.01]], columns=["Base"])
    
    # Create call recorders for collaborators
    render_explain_recorder = _CallRecorder()
    render_comparison_recorder = _CallRecorder()
    render_png_recorder = _CallRecorder()
    render_async_recorder = _CallRecorder()
    
    globals_ = module["main"].__globals__
    globals_.update(
        {
            "st": fake_st,
            "PLOTS": {"Headline": "headline", "Paths": "paths"},
            "render_settings_sidebar": lambda _: (str(workbook), "theme.yaml"),
            "apply_theme": lambda _: None,
            "load_data": lambda _: (summary, paths),
            "_get_plot_fn": lambda _: lambda frame: frame,
            "_render_explain_results": render_explain_recorder,
            "_render_comparison_panel": render_comparison_recorder,
            "render_png_download": render_png_recorder,
            "render_async_export_button": render_async_recorder,
        }
    )

    module["main"]()

    # Verify top-level rendering
    assert fake_st.titles == ["Results"]
    assert fake_st.subheaders == ["Key Metrics"]
    assert fake_st.frames == [summary]
    assert fake_st.errors == []
    
    # Verify charts were rendered
    assert len(fake_st.charts) == 2
    assert fake_st.charts[0] is summary
    assert list(fake_st.charts[1].columns) == ["Base"]
    
    # Verify XLSX download button
    assert ("Download XLSX", "results.xlsx") in fake_st.downloads
    
    # Verify containers were retained from columns() calls
    # There are two calls: st.columns(4) for metrics and st.columns(3) for export buttons
    assert len(fake_st.column_containers) == 2
    
    # The first call is for the 4 metric columns
    metric_col_containers = fake_st.column_containers[0]
    assert len(metric_col_containers) == 4  # Four columns for metrics
    
    # The second call is for the 3 export button columns
    export_col_containers = fake_st.column_containers[1]
    assert len(export_col_containers) == 3  # Three columns for export buttons
    
    # Verify metric labels in the metric column containers
    all_metrics = []
    for container in metric_col_containers:
        all_metrics.extend(container.metrics)
    
    # Check that all expected metrics are present with correct labels
    metric_labels = [m[0] for m in all_metrics]
    assert "Tracking Error" in metric_labels
    assert "monthly_CVaR" in metric_labels
    assert "Breach Prob" in metric_labels
    assert "Overlay Total" in metric_labels
    
    # Verify collaborator calls were recorded
    assert len(render_explain_recorder.calls) == 1
    assert len(render_comparison_recorder.calls) == 1
    assert len(render_png_recorder.calls) >= 1
    assert len(render_async_recorder.calls) >= 1


def test_overlay_total_return_computes_from_summary_with_agent_total() -> None:
    """Test _overlay_total_return extracts total return from summary DataFrame."""
    module = _load_results_module()
    overlay_fn = module["_overlay_total_return"]
    
    # Test with Agent=Total row present
    summary_with_total = pd.DataFrame(
        {
            "Agent": ["Base", "Total"],
            "terminal_AnnReturn": [0.05, 0.08],
        }
    )
    result = overlay_fn(summary_with_total)
    assert result == 0.08
    
    # Test with only Total row
    summary_only_total = pd.DataFrame(
        {
            "Agent": ["Total"],
            "terminal_AnnReturn": [0.10],
        }
    )
    result = overlay_fn(summary_only_total)
    assert result == 0.10
    
    # Test with no Total row
    summary_no_total = pd.DataFrame(
        {
            "Agent": ["Base", "H"],
            "terminal_AnnReturn": [0.05, 0.07],
        }
    )
    result = overlay_fn(summary_no_total)
    assert result is None
    
    # Test with missing columns
    summary_missing_cols = pd.DataFrame({"Config": ["Base"]})
    result = overlay_fn(summary_missing_cols)
    assert result is None
