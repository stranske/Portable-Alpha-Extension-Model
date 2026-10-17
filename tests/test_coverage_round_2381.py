"""Focused coverage for the Results page's executable UI branches (issue #2381)."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pandas as pd


class _Container:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, str]] = []
        self.downloads: list[str] = []

    def __enter__(self) -> _Container:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def metric(self, label: str, value: str, **_: object) -> None:
        self.metrics.append((label, value))

    def download_button(self, label: str, *_: object, **__: object) -> None:
        self.downloads.append(label)


class _ResultsStreamlit:
    def __init__(self) -> None:
        self.sidebar = self
        self.session_state: dict[str, object] = {}
        self.titles: list[str] = []
        self.subheaders: list[str] = []
        self.frames: list[pd.DataFrame] = []
        self.errors: list[str] = []

    def title(self, value: str) -> None:
        self.titles.append(value)

    def subheader(self, value: str) -> None:
        self.subheaders.append(value)

    def columns(self, count: int) -> list[_Container]:
        return [_Container() for _ in range(count)]

    def tabs(self, names: list[str]) -> list[_Container]:
        return [_Container() for _ in names]

    def slider(self, *_: object) -> int:
        return 1

    def multiselect(self, _: str, options: list[str], __: list[str]) -> list[str]:
        return options

    def number_input(self, *_: object, **__: object) -> float:
        return 0.0

    def checkbox(self, _: str) -> bool:
        return False

    def plotly_chart(self, *_: object, **__: object) -> None:
        return None

    def dataframe(self, value: pd.DataFrame) -> None:
        self.frames.append(value)

    def download_button(self, *_: object, **__: object) -> None:
        return None

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
    return runpy.run_path(str(Path("dashboard/pages/4_Results.py")))


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
    globals_ = module["main"].__globals__
    globals_.update(
        {
            "st": fake_st,
            "PLOTS": {"Headline": "headline", "Paths": "paths"},
            "render_settings_sidebar": lambda _: (str(workbook), "theme.yaml"),
            "apply_theme": lambda _: None,
            "load_data": lambda _: (summary, paths),
            "_get_plot_fn": lambda _: lambda frame: frame,
            "_render_explain_results": lambda **_: None,
            "_render_comparison_panel": lambda **_: None,
            "render_png_download": lambda *_args, **_kwargs: None,
            "render_async_export_button": lambda *_args, **_kwargs: None,
        }
    )

    module["main"]()

    assert fake_st.titles == ["Results"]
    assert fake_st.subheaders == ["Key Metrics"]
    assert fake_st.frames == [summary]
