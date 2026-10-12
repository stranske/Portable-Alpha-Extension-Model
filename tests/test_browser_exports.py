from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from typing import Any

import pytest

from dashboard import browser_exports


class _FakeSt:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.download_calls: list[dict[str, Any]] = []
        self.button_handlers: dict[str, bool] = {}
        self.info_messages: list[str] = []
        self.error_messages: list[str] = []
        self.rerun_count = 0

    def download_button(
        self,
        label: str,
        data: bytes,
        file_name: str,
        mime: str = "image/png",
        key: str | None = None,
    ) -> None:
        self.download_calls.append(
            {
                "label": label,
                "data": data,
                "file_name": file_name,
                "mime": mime,
                "key": key,
            }
        )

    def button(self, label: str, key: str) -> bool:
        return self.button_handlers.get(key, False)

    def info(self, message: str) -> None:
        self.info_messages.append(message)

    def error(self, message: str) -> None:
        self.error_messages.append(message)

    def rerun(self) -> None:
        self.rerun_count += 1


class _DoneTask:
    def __init__(self, result: Any) -> None:
        self._result = result

    def done(self) -> bool:
        return True

    def result(self) -> Any:
        return self._result


class _PendingTask:
    def done(self) -> bool:
        return False


class _FailedTask:
    def done(self) -> bool:
        return True

    def result(self) -> Any:
        raise RuntimeError("render failed")


def test_render_png_download_server_uses_kaleido_path(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _FakeSt()
    fig = object()
    rendered: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(browser_exports, "is_browser_runtime", lambda: False)

    def fake_png(f: Any, **opts: Any) -> bytes:
        rendered.append((f, opts))
        return b"png-bytes"

    monkeypatch.setattr(browser_exports, "figure_to_png_bytes", fake_png)

    # The Scenario Grid page passes scale=2; the export must render at the scale it asked for.
    browser_exports.render_png_download(
        st,
        key="chart",
        label="Download chart",
        fig=fig,
        file_name="chart.png",
        scale=2,
    )

    assert rendered == [(fig, {"scale": 2})]

    assert st.download_calls == [
        {
            "label": "Download chart",
            "data": b"png-bytes",
            "file_name": "chart.png",
            "mime": "image/png",
            "key": None,
        }
    ]


def test_render_png_download_browser_prepares_async_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = _FakeSt()
    fig = object()
    cache = {"k": b"cached"}
    calls: list[tuple[str, Any, dict[str, Any]]] = []
    monkeypatch.setattr(browser_exports, "is_browser_runtime", lambda: True)

    async def fake_prerender(figs: list[Any], **opts: Any) -> dict[str, bytes]:
        calls.append(("prerender", figs, opts))
        return cache

    active: list[dict[str, bytes]] = []

    @contextlib.contextmanager
    def fake_use_cache(used: dict[str, bytes]) -> Iterator[None]:
        calls.append(("use_cache", used, {}))
        active.append(used)
        try:
            yield
        finally:
            active.pop()

    def fake_png(f: Any, **opts: Any) -> bytes:
        calls.append(("render", f, opts))
        # Only a render made inside the prepared cache's context gets the cached bytes.
        return active[-1]["k"] if active else b"uncached"

    monkeypatch.setattr(browser_exports, "prerender_png_cache", fake_prerender)
    monkeypatch.setattr(browser_exports, "use_png_cache", fake_use_cache)
    monkeypatch.setattr(browser_exports, "figure_to_png_bytes", fake_png)
    st.button_handlers["chart_prepare"] = True

    async def exercise() -> None:
        browser_exports.render_png_download(
            st,
            key="chart",
            label="Download chart",
            fig=fig,
            file_name="chart.png",
            scale=2,
        )
        task_key = "_browser_png_export_task_chart"
        assert task_key in st.session_state
        task = st.session_state[task_key]
        assert isinstance(task, asyncio.Task)
        assert await task == b"cached"

    asyncio.run(exercise())
    # Both rendering stages see the requested options, and the render reads the cache that the
    # prerender built with those same options.
    assert calls == [
        ("prerender", [fig], {"scale": 2}),
        ("use_cache", cache, {}),
        ("render", fig, {"scale": 2}),
    ]
    assert st.info_messages == ["Preparing image export..."]
    assert st.rerun_count == 1


def test_render_png_download_browser_serves_ready_download(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _FakeSt()
    st.session_state["_browser_png_export_task_chart"] = _DoneTask(b"ready-png")
    monkeypatch.setattr(browser_exports, "is_browser_runtime", lambda: True)

    browser_exports.render_png_download(
        st,
        key="chart",
        label="Download chart",
        fig=object(),
        file_name="chart.png",
    )

    assert st.download_calls == [
        {
            "label": "Download chart",
            "data": b"ready-png",
            "file_name": "chart.png",
            "mime": "image/png",
            "key": "chart_download",
        }
    ]


def test_render_png_download_browser_surfaces_render_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = _FakeSt()
    st.session_state["_browser_png_export_task_chart"] = _FailedTask()
    monkeypatch.setattr(browser_exports, "is_browser_runtime", lambda: True)

    browser_exports.render_png_download(
        st,
        key="chart",
        label="Download chart",
        fig=object(),
        file_name="chart.png",
    )

    assert "_browser_png_export_task_chart" not in st.session_state
    assert st.error_messages == ["PNG export failed: render failed"]


def test_render_png_download_browser_shows_pending_state(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _FakeSt()
    st.session_state["_browser_png_export_task_chart"] = _PendingTask()
    monkeypatch.setattr(browser_exports, "is_browser_runtime", lambda: True)

    browser_exports.render_png_download(
        st,
        key="chart",
        label="Download chart",
        fig=object(),
        file_name="chart.png",
    )

    assert st.info_messages == ["Preparing image export..."]
    assert st.download_calls == []


def test_render_async_export_button_server_invokes_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _FakeSt()
    st.button_handlers["export"] = True
    ready: list[str] = []
    monkeypatch.setattr(browser_exports, "is_browser_runtime", lambda: False)

    async def factory() -> str:
        return "payload"

    browser_exports.render_async_export_button(
        st,
        key="export",
        label="Export workbook",
        factory=factory,
        on_ready=lambda value: ready.append(value),
    )

    assert ready == ["payload"]


def test_render_async_export_button_server_runs_sync_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = _FakeSt()
    st.button_handlers["export"] = True
    ready: list[int] = []
    monkeypatch.setattr(browser_exports, "is_browser_runtime", lambda: False)

    browser_exports.render_async_export_button(
        st,
        key="export",
        label="Export workbook",
        factory=lambda: 7,
        on_ready=lambda value: ready.append(value),
    )

    assert ready == [7]


def test_render_async_export_button_browser_schedules_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = _FakeSt()
    st.button_handlers["export"] = True
    monkeypatch.setattr(browser_exports, "is_browser_runtime", lambda: True)

    async def factory() -> str:
        await asyncio.sleep(0)
        return "async-payload"

    async def exercise() -> None:
        browser_exports.render_async_export_button(
            st,
            key="export",
            label="Export workbook",
            factory=factory,
            on_ready=lambda _: None,
            pending_message="Working...",
        )
        task_key = "_browser_async_export_task_export"
        assert task_key in st.session_state
        task = st.session_state[task_key]
        assert isinstance(task, asyncio.Task)
        await task

    asyncio.run(exercise())
    assert st.info_messages == ["Working..."]
    assert st.rerun_count == 1


def test_render_async_export_button_browser_delivers_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = _FakeSt()
    st.session_state["_browser_async_export_task_export"] = _DoneTask("done-value")
    delivered: list[str] = []
    monkeypatch.setattr(browser_exports, "is_browser_runtime", lambda: True)

    browser_exports.render_async_export_button(
        st,
        key="export",
        label="Export workbook",
        factory=lambda: "ignored",
        on_ready=lambda value: delivered.append(value),
    )

    assert delivered == ["done-value"]


def test_render_async_export_button_browser_shows_pending_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = _FakeSt()
    st.session_state["_browser_async_export_task_export"] = _PendingTask()
    monkeypatch.setattr(browser_exports, "is_browser_runtime", lambda: True)

    browser_exports.render_async_export_button(
        st,
        key="export",
        label="Export workbook",
        factory=lambda: "ignored",
        on_ready=lambda _: None,
        pending_message="Still working...",
    )

    assert st.info_messages == ["Still working..."]


def test_render_async_export_button_browser_reports_factory_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = _FakeSt()
    st.session_state["_browser_async_export_task_export"] = _FailedTask()
    monkeypatch.setattr(browser_exports, "is_browser_runtime", lambda: True)

    browser_exports.render_async_export_button(
        st,
        key="export",
        label="Export workbook",
        factory=lambda: "ignored",
        on_ready=lambda _: None,
    )

    assert "_browser_async_export_task_export" not in st.session_state
    assert st.error_messages == ["Export failed: render failed"]


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None
