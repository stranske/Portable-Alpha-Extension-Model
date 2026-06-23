from __future__ import annotations

import sys
import asyncio
import base64
import zipfile
from pathlib import Path

import pytest

from pa_core.viz import export_backend
from pa_core.viz import pptx_export


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


def test_browser_pptx_save_prerenders_without_manual_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "platform", "emscripten")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("PA_PPTX_PLACEHOLDER", raising=False)
    fig = _Figure()
    calls: list[tuple[object, dict[str, object]]] = []
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ"
        "/pLvAAAAAElFTkSuQmCC"
    )

    async def fake_bridge(figure: object, **opts: object) -> bytes:
        calls.append((figure, opts))
        return png

    monkeypatch.setattr(export_backend, "_plotlyjs_bridge_png_bytes", fake_bridge)

    out = tmp_path / "browser-flow.pptx"
    asyncio.run(pptx_export.save_async([fig], out, alt_texts=["browser chart"]))

    assert calls == [(fig, {})]
    assert fig.calls == []
    with zipfile.ZipFile(out) as zf:
        media = [name for name in zf.namelist() if name.startswith("ppt/media/")]
        assert media
        assert any(zf.read(name) == png for name in media)
