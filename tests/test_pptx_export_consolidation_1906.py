"""Regression tests for issue #1906: consolidate the two PPTX exporters.

`pa_core/viz/pptx_export.py` and `pa_core/reporting/export_packet.py` used to be
near-duplicate PPTX implementations with divergent kaleido error handling (one
silently swallowed render failures, the other raised). Both must now share a
single render/slide code path with consistent, actionable error handling so a
missing renderer can never silently drop chart slides from a board pack.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest
from pptx import Presentation

from pa_core.reporting import export_packet
from pa_core.viz import pptx_export


class _BoomFig:
    """Stand-in figure whose static render always fails (no kaleido)."""

    layout = None

    def to_image(self, *args: object, **kwargs: object) -> bytes:
        raise ValueError("Kaleido/Chromium not installed")


def _without_ci_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the real-render branch (CI/pytest normally short-circuit to a
    # placeholder image to avoid spawning kaleido).
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)


def test_export_packet_delegates_to_shared_helper() -> None:
    # Guard against re-divergence: the committee packet must reuse the viz
    # exporter rather than reimplementing chart rendering.
    assert export_packet.add_chart_slide is pptx_export.add_chart_slide


def test_render_chart_png_uses_placeholder_under_pytest() -> None:
    # Under pytest PYTEST_CURRENT_TEST is set, so no kaleido subprocess runs.
    assert pptx_export.render_chart_png(_BoomFig()) == pptx_export._ONE_PX_PNG


def test_save_raises_actionable_error_without_renderer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    # `save` no longer silently swallows render failures.
    _without_ci_placeholder(monkeypatch)
    with pytest.raises(RuntimeError, match="static image renderer"):
        pptx_export.save([_BoomFig()], tmp_path / "out.pptx")  # type: ignore[operator]


def test_packet_chart_slide_raises_same_error_without_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The packet path raises the identical actionable error (shared code path).
    _without_ci_placeholder(monkeypatch)
    prs = Presentation()
    with pytest.raises(RuntimeError, match="static image renderer"):
        export_packet._add_chart_slide(prs, _BoomFig())


def test_save_writes_slides_with_alt_text(tmp_path: object) -> None:
    fig = go.Figure()
    out = tmp_path / "out.pptx"  # type: ignore[operator]
    pptx_export.save([fig], out, alt_texts=["my chart"])
    assert out.exists()
    pres = Presentation(out)
    # Layout 5 carries a title placeholder, so scan all shapes for the picture.
    descrs = [
        node.get("descr")
        for shape in pres.slides[0].shapes
        for node in shape._element.xpath("./p:nvPicPr/p:cNvPr")
    ]
    assert "my chart" in descrs
