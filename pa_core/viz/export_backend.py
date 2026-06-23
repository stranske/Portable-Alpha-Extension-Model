from __future__ import annotations

import base64
import contextlib
import contextvars
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, cast

ImageCache = dict[str, bytes]

_PNG_CACHE: contextvars.ContextVar[ImageCache | None] = contextvars.ContextVar(
    "pa_core_plotly_png_cache",
    default=None,
)


def is_browser_runtime() -> bool:
    return sys.platform == "emscripten"


def figure_image_cache_key(fig: Any, *, format: str = "png", **opts: Any) -> str:
    payload = {
        "format": format,
        "opts": {key: opts[key] for key in sorted(opts)},
        "figure": json.loads(fig.to_json()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@contextlib.contextmanager
def use_png_cache(cache: ImageCache | None) -> Iterator[None]:
    token = _PNG_CACHE.set(cache)
    try:
        yield
    finally:
        _PNG_CACHE.reset(token)


def seed_png_cache(fig: Any, png_bytes: bytes, **opts: Any) -> str:
    cache = _PNG_CACHE.get()
    if cache is None:
        raise RuntimeError("No Plotly PNG export cache is active.")
    key = figure_image_cache_key(fig, format="png", **opts)
    cache[key] = png_bytes
    return key


def figure_to_png_bytes(fig: Any, **opts: Any) -> bytes:
    """Return PNG bytes for a Plotly figure.

    Server Python keeps using Plotly/Kaleido synchronously. In stlite/Pyodide,
    callers must first populate the cache with :func:`prerender_png_cache`
    because Plotly.js image rendering is Promise-based while PPTX/Excel assembly
    remains synchronous.
    """
    return figure_to_image_bytes(fig, format="png", **opts)


def figure_to_pdf_bytes(fig: Any, **opts: Any) -> bytes:
    return figure_to_image_bytes(fig, format="pdf", **opts)


def figure_to_image_bytes(fig: Any, *, format: str = "png", **opts: Any) -> bytes:
    clean_opts = _without_engine(opts)
    if is_browser_runtime():
        if format != "png":
            raise RuntimeError(
                f"Browser Plotly export only supports cached PNG bytes; got {format!r}."
            )
        cache = _PNG_CACHE.get()
        key = figure_image_cache_key(fig, format="png", **clean_opts)
        if cache is not None and key in cache:
            return cache[key]
        raise RuntimeError(
            "Browser Plotly PNG export was requested before async pre-render completed. "
            "Call pa_core.viz.export_backend.prerender_png_cache at the export action boundary."
        )
    return cast(bytes, fig.to_image(format=format, engine="kaleido", **clean_opts))


def write_figure_image(
    fig: Any,
    path: str | Path,
    *,
    format: str | None = None,
    **opts: Any,
) -> None:
    clean_opts = _without_engine(opts)
    if not is_browser_runtime():
        write_opts: dict[str, Any] = {"engine": "kaleido", **clean_opts}
        if format is not None:
            write_opts["format"] = format
        fig.write_image(path, **write_opts)
        return
    image_format = format or Path(path).suffix.lstrip(".") or "png"
    Path(path).write_bytes(figure_to_image_bytes(fig, format=image_format, **clean_opts))


async def prerender_png_cache(figs: Any, **opts: Any) -> ImageCache:
    """Render Plotly figures to PNG bytes in the browser and return a cache."""
    clean_opts = _without_engine(opts)
    figures = list(figs)
    cache: ImageCache = {}
    if not is_browser_runtime():
        for fig in figures:
            key = figure_image_cache_key(fig, format="png", **clean_opts)
            cache[key] = figure_to_png_bytes(fig, **clean_opts)
        return cache
    for fig in figures:
        key = figure_image_cache_key(fig, format="png", **clean_opts)
        cache[key] = await _plotlyjs_bridge_png_bytes(fig, **clean_opts)
    return cache


async def _plotlyjs_bridge_png_bytes(fig: Any, **opts: Any) -> bytes:
    """Render a Plotly figure with Plotly.js inside Pyodide/stlite."""
    figjson = json.loads(fig.to_json())
    scale = opts.get("scale", 2)
    width = opts.get("width")
    height = opts.get("height")
    data_url = _run_plotlyjs_to_image(figjson, scale=scale, width=width, height=height)
    if inspect.isawaitable(data_url):
        data_url = await data_url
    return _decode_data_url(str(data_url))


def _run_plotlyjs_to_image(
    figjson: Mapping[str, Any],
    *,
    scale: Any = 2,
    width: Any = None,
    height: Any = None,
) -> Any:
    from pyodide.code import run_js  # type: ignore[import-not-found]

    render = run_js("""
        async (figjson, opts) => {
          const div = document.createElement("div");
          div.style.position = "fixed";
          div.style.left = "-10000px";
          div.style.top = "-10000px";
          div.style.width = `${opts.width || 960}px`;
          div.style.height = `${opts.height || 540}px`;
          document.body.appendChild(div);
          try {
            await Plotly.newPlot(div, figjson.data || [], figjson.layout || {}, {});
            return await Plotly.toImage(div, {
              format: "png",
              scale: opts.scale || 2,
              width: opts.width || undefined,
              height: opts.height || undefined,
            });
          } finally {
            Plotly.purge(div);
            div.remove();
          }
        }
        """)
    return render(figjson, {"scale": scale, "width": width, "height": height})


def _decode_data_url(data_url: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise RuntimeError("Plotly.js did not return a PNG data URL.")
    return base64.b64decode(data_url[len(prefix) :])


def _without_engine(opts: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(opts)
    clean.pop("engine", None)
    return clean
