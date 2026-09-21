from __future__ import annotations

import sys
import asyncio
import base64
import tomllib
import types
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

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


def test_plotly_dependency_excludes_unverified_major() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    dependencies = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    for requirements in (
        dependencies,
        (root / "requirements.txt").read_text().splitlines(),
        (root / "requirements-dev.txt").read_text().splitlines(),
    ):
        plotly = next(Requirement(dep) for dep in requirements if dep.startswith("plotly"))
        assert any(spec.operator == ">=" and spec.version == "6.7.0" for spec in plotly.specifier)
        assert "7.0.0" not in plotly.specifier
        assert any(spec.operator == "<" and spec.version == "7" for spec in plotly.specifier)


def test_server_branch_uses_kaleido(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    fig = _Figure()

    assert export_backend.figure_to_png_bytes(fig, scale=2) == b"kaleido-png"
    assert fig.calls == [{"format": "png", "scale": 2}]


def test_server_static_exports_accept_plotly7_signature(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The removed engine keyword must not reach either Plotly export method."""
    monkeypatch.setattr(sys, "platform", "darwin")

    class Plotly7Figure:
        def to_image(self, *, format: str, scale: int) -> bytes:
            assert (format, scale) == ("png", 2)
            return b"png-bytes"

        def write_image(self, path: str | Path, *, format: str, scale: int) -> None:
            assert (format, scale) == ("png", 2)
            Path(path).write_bytes(b"png-bytes")

    fig = Plotly7Figure()
    assert export_backend.figure_to_png_bytes(fig, scale=2, engine="kaleido") == b"png-bytes"
    output = tmp_path / "plotly7.png"
    export_backend.write_figure_image(fig, output, format="png", scale=2, engine="kaleido")
    assert output.read_bytes() == b"png-bytes"


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


def test_plotlyjs_bridge_posts_string_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    fig = _Figure()
    calls: list[tuple[object, dict[str, object]]] = []
    scripts: list[str] = []

    def fake_run_js(script: str) -> object:
        scripts.append(script)

        def requester(fig_arg: object, opts: dict[str, object]) -> str:
            calls.append((fig_arg, opts))
            return (
                "data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAA"
                "MBAQDJ/pLvAAAAAElFTkSuQmCC"
            )

        return requester

    pyodide_module = types.ModuleType("pyodide")
    pyodide_code_module = types.ModuleType("pyodide.code")
    pyodide_code_module.run_js = fake_run_js  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyodide", pyodide_module)
    monkeypatch.setitem(sys.modules, "pyodide.code", pyodide_code_module)

    png = asyncio.run(export_backend._plotlyjs_bridge_png_bytes(fig, scale=3, width=640))

    assert png
    assert len(calls) == 1
    fig_arg, opts = calls[0]
    assert isinstance(fig_arg, str)
    assert fig_arg == fig.to_json()
    assert opts == {"scale": 3, "width": 640, "height": None}
    assert "BroadcastChannel" in scripts[0]
    assert "document" not in scripts[0]
    assert "window" not in scripts[0]
    assert "Plotly" not in scripts[0]


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
