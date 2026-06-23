from __future__ import annotations

import sys
import asyncio

import pytest

from pa_core.viz import export_backend


class _Figure:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def to_json(self) -> str:
        return '{"data":[{"type":"scatter","x":[1],"y":[2]}],"layout":{"title":"T"}}'

    def to_image(self, **kwargs: object) -> bytes:
        self.calls.append(kwargs)
        return b"kaleido-png"


def test_module_exposes_figure_to_png_bytes() -> None:
    assert callable(export_backend.figure_to_png_bytes)


def test_server_branch_uses_kaleido(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    fig = _Figure()

    assert export_backend.figure_to_png_bytes(fig, scale=2) == b"kaleido-png"
    assert fig.calls == [{"format": "png", "engine": "kaleido", "scale": 2}]


def test_browser_branch_uses_plotlyjs_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "emscripten")
    fig = _Figure()
    calls: list[tuple[object, dict[str, object]]] = []

    async def fake_bridge(figure: object, **opts: object) -> bytes:
        calls.append((figure, opts))
        return b"plotlyjs-png"

    monkeypatch.setattr(export_backend, "_plotlyjs_bridge_png_bytes", fake_bridge)

    cache = asyncio.run(export_backend.prerender_png_cache([fig], scale=3))
    with export_backend.use_png_cache(cache):
        assert export_backend.figure_to_png_bytes(fig, scale=3) == b"plotlyjs-png"

    assert calls == [(fig, {"scale": 3})]
    assert fig.calls == []
