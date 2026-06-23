from __future__ import annotations

import base64
import contextlib
import contextvars
import hashlib
import inspect
import io
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, TypeVar, cast

ImageCache = dict[str, bytes]
T = TypeVar("T")

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
        if format not in {"png", "pdf"}:
            raise RuntimeError(
                f"Browser Plotly export only supports cached PNG/PDF bytes; got {format!r}."
            )
        cache = _PNG_CACHE.get()
        key = figure_image_cache_key(fig, format="png", **clean_opts)
        if cache is not None and key in cache:
            png_bytes = cache[key]
            if format == "pdf":
                return _png_to_pdf_bytes(png_bytes)
            return png_bytes
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


async def run_with_browser_png_cache(
    figs: Any,
    render: Callable[[], T],
    **opts: Any,
) -> T:
    """Run ``render`` with a populated Plotly PNG cache in browser runtimes.

    Server Python keeps the existing synchronous Kaleido path. Browser/Pyodide
    callers must await this at the export action boundary because Plotly.js image
    rendering is async while PPTX/Excel/PDF assembly APIs are synchronous.
    """
    if not is_browser_runtime():
        return render()
    cache = await prerender_png_cache(figs, **opts)
    with use_png_cache(cache):
        return render()


async def _plotlyjs_bridge_png_bytes(fig: Any, **opts: Any) -> bytes:
    """Request Plotly.js PNG rendering from the stlite main thread."""
    scale = opts.get("scale", 2)
    width = opts.get("width")
    height = opts.get("height")
    data_url = _run_plotlyjs_to_image(
        fig.to_json(),
        scale=scale,
        width=width,
        height=height,
    )
    if inspect.isawaitable(data_url):
        data_url = await data_url
    return _decode_data_url(str(data_url))


def _run_plotlyjs_to_image(
    fig_json_str: str,
    *,
    scale: Any = 2,
    width: Any = None,
    height: Any = None,
) -> Any:
    from pyodide.code import run_js  # type: ignore[import-not-found]

    requester = run_js("""
        (figStr, opts) => new Promise((resolve, reject) => {
          let bc;
          try {
            bc = new BroadcastChannel("pa-render");
          } catch (e) {
            reject(new Error("no BroadcastChannel in worker: " + e));
            return;
          }
          const id = Math.random().toString(36).slice(2);
          const timer = setTimeout(() => {
            try {
              bc.close();
            } catch (e) {}
            reject(new Error("plotly render timeout"));
          }, 30000);
          bc.onmessage = (e) => {
            const m = e.data;
            if (!m || m.kind !== "response" || m.id !== id) return;
            clearTimeout(timer);
            bc.close();
            if (m.error) reject(new Error(m.error));
            else resolve(m.url);
          };
          bc.postMessage({ kind: "request", id, figStr, opts });
        })
        """)
    return requester(fig_json_str, {"scale": scale, "width": width, "height": height})


def _decode_data_url(data_url: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise RuntimeError("Plotly.js did not return a PNG data URL.")
    return base64.b64decode(data_url[len(prefix) :])


def _png_to_pdf_bytes(png_bytes: bytes) -> bytes:
    try:
        from PIL import Image
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - optional dep
        raise RuntimeError("Browser PDF export requires Pillow to wrap cached PNG bytes.") from exc

    with Image.open(io.BytesIO(png_bytes)) as img:
        pdf_img: Any = img
        if img.mode in {"RGBA", "LA", "P"}:
            pdf_img = img.convert("RGB")
        out = io.BytesIO()
        pdf_img.save(out, format="PDF")
        return out.getvalue()


def _without_engine(opts: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(opts)
    clean.pop("engine", None)
    return clean
