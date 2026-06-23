from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pa_core.viz.export_backend import (
    figure_to_png_bytes,
    is_browser_runtime,
    prerender_png_cache,
    use_png_cache,
)


async def _render_png_browser(fig: Any, **opts: Any) -> bytes:
    cache = await prerender_png_cache([fig], **opts)
    with use_png_cache(cache):
        return figure_to_png_bytes(fig, **opts)


def render_png_download(
    st: Any,
    *,
    key: str,
    label: str,
    fig: Any,
    file_name: str,
    mime: str = "image/png",
    **opts: Any,
) -> None:
    if not is_browser_runtime():
        data = figure_to_png_bytes(fig, **opts)
        st.download_button(label, data, file_name=file_name, mime=mime)
        return

    task_key = f"_browser_png_export_task_{key}"
    task = st.session_state.get(task_key)
    if task is not None and task.done():
        try:
            data = task.result()
        except Exception as exc:
            st.session_state.pop(task_key, None)
            st.error(f"PNG export failed: {exc}")
            return
        st.download_button(label, data, file_name=file_name, mime=mime, key=f"{key}_download")
        return

    if task is not None:
        st.info("Preparing image export...")
        return

    if st.button(f"Prepare {label}", key=f"{key}_prepare"):
        st.session_state[task_key] = asyncio.create_task(_render_png_browser(fig, **opts))
        st.info("Preparing image export...")
        st.rerun()


def render_async_export_button(
    st: Any,
    *,
    key: str,
    label: str,
    factory: Callable[[], Awaitable[Any]],
    on_ready: Callable[[Any], None],
    pending_message: str = "Preparing export...",
) -> None:
    if not is_browser_runtime():
        if st.button(label, key=key):
            result = factory()
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            on_ready(result)
        return

    task_key = f"_browser_async_export_task_{key}"
    task = st.session_state.get(task_key)
    if task is not None and task.done():
        try:
            result = task.result()
        except Exception as exc:
            st.session_state.pop(task_key, None)
            st.error(f"Export failed: {exc}")
            return
        on_ready(result)
        return

    if task is not None:
        st.info(pending_message)
        return

    if st.button(label, key=key):
        st.session_state[task_key] = asyncio.create_task(factory())
        st.info(pending_message)
        st.rerun()
